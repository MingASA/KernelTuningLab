"""Scale and adversarial contract checks with synthetic, CPU-only fixtures."""

from dataclasses import replace
import math
import random
from types import SimpleNamespace

import pytest

from kernel_tuning_lab.backends.mock import MockAdapter
from kernel_tuning_lab.hardware import get_hardware_profile
from kernel_tuning_lab.l3 import PerformanceMatrix, ShapeDispatcher, greedy, greedy_local_swap, oracle
from kernel_tuning_lab.l3.solver import OracleLimitError
from kernel_tuning_lab.models import (
    HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec, RecordStatus,
    SearchBudget, ShapeSpec, WorkloadDistribution, canonical_json,
)
from kernel_tuning_lab.operators import operator_space, operator_spec
from kernel_tuning_lab.replay import run_replay
from kernel_tuning_lab.runner import BenchmarkRunner
from kernel_tuning_lab.search import SEARCHER_NAMES, SearchHistory, make_searcher
from kernel_tuning_lab.space import ConfigSpace
from kernel_tuning_lab.storage import ExperimentStore
from kernel_tuning_lab.tuning import tune


def _record(config, *, shape=None, latency=1.0, split="test", experiment="synthetic-fixture"):
    return MeasurementRecord(HardwareSpec("MOCK-scale-fixture", architecture="mock"),
                             OperatorSpec("categorical-fixture", kind="probe", implementation="probe"),
                             shape or ShapeSpec({"M": 17}), config, RecordStatus.SUCCESS,
                             runtime_samples_ms=(latency,), repetitions=1, wall_time_s=0.01,
                             provenance="mock", split=split, experiment_id=experiment,
                             environment={"backend": "fixture", "version": "1"},
                             metadata={"protocol": {"cache": "hot"}, "mutable_fixture": {"value": 1}})


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_batched_gemm_requires_independent_batches_and_rejects_unsupported_schema(dtype):
    hardware = get_hardware_profile("H200")
    operator = operator_spec("batched_gemm", dtype=dtype)
    space = operator_space("batched_gemm", small=True)
    config = next(space.iter_configs())
    assert space.validate(config, hardware, operator, ShapeSpec({"Batch": 1, "M": 1, "N": 129, "K": 65})).valid
    assert space.validate(config, hardware, operator, ShapeSpec({"Batch": 19, "M": 127, "N": 129, "K": 65})).valid
    for shape in (ShapeSpec({"M": 127, "N": 129, "K": 65}),
                  ShapeSpec({"Batch": 2, "M": 127, "N": 129, "K": 65, "BatchB": 1}),
                  ShapeSpec({"batch": 2, "M": 127, "N": 129, "K": 65})):
        assert not space.validate(config, hardware, operator, shape).valid
    shape = ShapeSpec({"Batch": 2, "M": 127, "N": 129, "K": 65})
    for changed in (replace(operator, dtype="float32"), replace(operator, dtype="int8"),
                    replace(operator, parameters={"transpose_b": True}),
                    replace(operator, parameters={"broadcast_batch": True})):
        assert not space.validate(config, hardware, changed, shape).valid


@pytest.mark.parametrize("name", ("softmax", "layernorm"))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "float32"))
def test_reduction_tail_and_maximum_row_boundaries(name, dtype):
    hardware = get_hardware_profile("4090")
    operator = operator_spec(name, dtype=dtype)
    shapes = (ShapeSpec({"M": 1, "N": 1}), ShapeSpec({"M": 7, "N": 193}),
              ShapeSpec({"M": 33, "N": 1025}), ShapeSpec({"M": 1, "N": 65536}))
    space = operator_space(name, shapes=shapes)
    for shape in shapes:
        block = 1 << (shape.get("N") - 1).bit_length()
        config = KernelConfig({"BLOCK_SIZE": block, "num_warps": 4, "num_stages": 1})
        assert space.validate(config, hardware, operator, shape).valid
    config = KernelConfig({"BLOCK_SIZE": 65536, "num_warps": 8, "num_stages": 2})
    assert not space.validate(config, hardware, operator, ShapeSpec({"M": 1, "N": 65537})).valid
    assert not space.validate(config, hardware, replace(operator, dtype="float64"), shapes[-1]).valid
    assert not space.validate(config, hardware, operator, ShapeSpec({"M": 7, "N": 193, "Batch": 2})).valid


@pytest.mark.parametrize("parameters", ({"axis": 0}, {"axis": True}, {"axis": "last"},
                                        {"affine": True}, {"elementwise_affine": True},
                                        {"eps": 0}, {"eps": -1}, {"eps": math.nan},
                                        {"eps": math.inf}, {"eps": True}, {"unknown": 1}))
def test_layernorm_rejects_unsupported_or_ambiguous_parameters(parameters):
    operator = replace(operator_spec("layernorm"), parameters=parameters)
    space = operator_space("layernorm", shapes=(ShapeSpec({"M": 7, "N": 193}),))
    assert not space.validate(next(space.iter_configs()), get_hardware_profile("H800"), operator,
                              ShapeSpec({"M": 7, "N": 193})).valid


def test_jsonl_large_malformed_tail_and_multi_split_conflicts_do_not_partly_import(tmp_path):
    original = _record(KernelConfig({"x": 0}), split="train")
    new_rows = [_record(KernelConfig({"x": i}), shape=ShapeSpec({"M": i + 32}),
                        split=("train", "validation", "test")[i % 3]) for i in range(128)]
    source = tmp_path / "bad-tail.jsonl"
    source.write_text("\n".join(canonical_json(record.to_dict()) for record in new_rows) + "\n{broken\n")
    with ExperimentStore(tmp_path / "atomic.sqlite") as store:
        store.add(original)
        before = {table: store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                  for table in ("hardware", "operators", "shapes", "configs", "measurements")}
        with pytest.raises(ValueError):
            store.import_jsonl(source)
        assert [row.id for row in store.records()] == [original.id]
        assert before == {table: store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                          for table in before}
        conflict = replace(new_rows[-1], shape=new_rows[0].shape, split="test")
        source.write_text("\n".join(canonical_json(row.to_dict()) for row in (*new_rows, conflict)) + "\n")
        with pytest.raises(ValueError, match="split"):
            store.import_jsonl(source)
        assert [row.id for row in store.records()] == [original.id]
        independent = replace(original, split="test", experiment_id="independent-experiment")
        source.write_text("\n".join(canonical_json(row.to_dict()) for row in (*new_rows, independent)) + "\n")
        assert store.import_jsonl(source) == {"read": 129, "added": 129, "duplicates": 0}
        assert {row.split for row in store.records()} == {"train", "validation", "test"}
        assert len(store.records(experiment_id="independent-experiment", split="test")) == 1
        assert store.import_jsonl(source) == {"read": 129, "added": 0, "duplicates": 129}


def test_export_preserves_existing_target_when_reading_fails_midstream(tmp_path, monkeypatch):
    target = tmp_path / "existing.jsonl"
    original_bytes = b"previous successful export\n"
    target.write_bytes(original_bytes)
    row = _record(KernelConfig({"x": 1}))
    with ExperimentStore(tmp_path / "journal.sqlite") as store:
        store.add(row)

        def interrupted_records(**filters):
            yield row
            raise OSError("simulated interrupted source read")

        monkeypatch.setattr(store, "iter_records", interrupted_records)
        with pytest.raises(OSError, match="interrupted"):
            store.export_jsonl(target)
        assert target.read_bytes() == original_bytes
    assert {path.name for path in tmp_path.iterdir()} == {"existing.jsonl", "journal.sqlite"}


def test_add_many_rolls_back_when_the_input_iterator_fails(tmp_path):
    original = _record(KernelConfig({"x": 0}))

    def interrupted_batch():
        for index in range(1, 33):
            yield _record(KernelConfig({"x": index}), shape=ShapeSpec({"M": index + 100}))
        raise RuntimeError("synthetic interrupted batch")

    with ExperimentStore(tmp_path / "batch.sqlite") as store:
        store.add(original)
        with pytest.raises(RuntimeError, match="interrupted batch"):
            store.add_many(interrupted_batch())
        assert [record.id for record in store.records()] == [original.id]
        assert store.connection.execute("SELECT COUNT(*) FROM configs").fetchone()[0] == 1
        assert store.connection.execute("SELECT COUNT(*) FROM shapes").fetchone()[0] == 1


def test_export_failure_to_replace_preserves_target_and_removes_temporary_file(tmp_path, monkeypatch):
    import kernel_tuning_lab.storage as implementation

    target = tmp_path / "archive.jsonl"
    target.write_bytes(b"previous archive\n")

    def failed_replace(source, destination):
        assert source.parent == destination.parent == tmp_path
        assert source.is_file()
        raise OSError("synthetic failed atomic replace")

    with ExperimentStore(tmp_path / "journal.sqlite") as store:
        store.add(_record(KernelConfig({"x": 1})))
        monkeypatch.setattr(implementation.os, "replace", failed_replace)
        with pytest.raises(OSError, match="atomic replace"):
            store.export_jsonl(target)
        assert target.read_bytes() == b"previous archive\n"
    assert {path.name for path in tmp_path.iterdir()} == {"archive.jsonl", "journal.sqlite"}


def test_export_cannot_overwrite_the_sqlite_journal(tmp_path):
    database = tmp_path / "journal.sqlite"
    row = _record(KernelConfig({"x": 1}))
    with ExperimentStore(database) as store:
        store.add(row)
        with pytest.raises(ValueError, match="overwrite"):
            store.export_jsonl(database)
        assert [record.id for record in store.records()] == [row.id]
    with ExperimentStore(database, read_only=True) as store:
        assert [record.id for record in store.records()] == [row.id]


def test_read_only_store_exports_and_filters_without_accepting_writes(tmp_path):
    database = tmp_path / "readonly.sqlite"
    row = _record(KernelConfig({"x": 1}))
    with ExperimentStore(database) as store:
        store.add(row)
    original_bytes = database.read_bytes()
    exchange = tmp_path / "readonly.jsonl"
    with ExperimentStore(database, read_only=True) as store:
        assert list(store.iter_records(split="train")) == []
        assert [record.id for record in store.iter_records(split="test")] == [row.id]
        assert store.export_jsonl(exchange) == 1
        for write in (lambda: store.add(row), lambda: store.add_many((row,)),
                      lambda: store.import_jsonl(exchange)):
            with pytest.raises(PermissionError, match="read-only"):
                write()
    assert database.read_bytes() == original_bytes
    missing = tmp_path / "absent" / "never-created.sqlite"
    with pytest.raises(FileNotFoundError):
        ExperimentStore(missing, read_only=True)
    assert not missing.parent.exists()


@pytest.mark.parametrize("algorithm", SEARCHER_NAMES)
def test_unrequested_latency_perturbations_cannot_change_search_trajectory(algorithm):
    space = ConfigSpace({"x": tuple(range(32))})
    rows = tuple(_record(config, latency=1 + index / 32) for index, config in enumerate(space.iter_configs()))
    first = run_replay(rows, make_searcher(algorithm, seed=23), space, SearchBudget(4))
    revealed = {step.config.id for step in first.trace}
    perturbed = tuple(record if record.config.id in revealed else replace(record, runtime_samples_ms=(0.0001,))
                      for record in rows)
    second = run_replay(perturbed, make_searcher(algorithm, seed=23), space, SearchBudget(4))
    assert [step.config.id for step in second.trace] == [step.config.id for step in first.trace]
    assert [step.latency_ms for step in second.trace] == [step.latency_ms for step in first.trace]
    assert second.best_latency_ms == first.best_latency_ms
    assert second.oracle_best_latency_ms == 0.0001
    assert second.regret > first.regret


def test_replay_history_mutations_do_not_modify_source_trace_or_next_history():
    space = ConfigSpace({"x": (0, 1, 2)})
    rows = tuple(_record(config, latency=3 - index) for index, config in enumerate(space.iter_configs()))
    before = tuple(canonical_json(record.to_dict()) for record in rows)

    class MutatingSearcher:
        name = "history-mutation-fixture"

        def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
            index = len(history)
            assert len(history.attempted_config_ids) == index
            for observed_index, record in enumerate(history):
                assert record.config.params == {"x": observed_index}
                assert record.metadata["mutable_fixture"]["value"] == 1
                record.config.params["x"] = 999
                record.metadata["mutable_fixture"]["value"] = 999
            hardware.metadata["injected"] = True
            operator.parameters["injected"] = True
            shape.dims["M"] = 999
            return [KernelConfig({"x": index})]

    result = run_replay(rows, MutatingSearcher(), space, SearchBudget(3))
    assert result.evaluations == 3, result.stop_reason
    assert result.stop_reason == "evaluation_budget"
    assert [step.config.params for step in result.trace] == [{"x": 0}, {"x": 1}, {"x": 2}]
    assert result.best_config.params == {"x": 2}
    assert tuple(canonical_json(record.to_dict()) for record in rows) == before


class CountingSpace(ConfigSpace):
    """Permit bounded grid scanning, but detect eager Cartesian enumeration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated = 0
        self.validations = 0

    def iter_configs(self):
        for config in super().iter_configs():
            self.generated += 1
            if self.generated > 4096:
                raise AssertionError("candidate generation traversed the Cartesian space")
            yield config

    def legal_configs(self, hardware, operator, shape):
        raise AssertionError("large spaces must not materialize every legal configuration")

    def validate(self, config, hardware, operator, shape):
        self.validations += 1
        if self.validations > 8192:
            raise AssertionError("legality checks exceeded bounded candidate work")
        return super().validate(config, hardware, operator, shape)


def test_million_configuration_sampling_is_reproducible_unique_and_exclusion_aware():
    space = CountingSpace({"x": tuple(range(1000)), "y": tuple(range(1000))})
    assert len(space) == 1_000_000
    first = space.sample_configs(random.Random(37), 256)
    second = space.sample_configs(random.Random(37), 256)
    assert first == second and len(first) == 256
    assert len({config.id for config in first}) == 256
    excluded = {config.id for config in first[:192]}
    remaining = space.sample_configs(random.Random(37), 256, exclude_ids=excluded)
    assert len(remaining) == len({config.id for config in remaining}) == 256
    assert not excluded & {config.id for config in remaining}
    assert all(0 <= config.get("x") < 1000 and 0 <= config.get("y") < 1000 for config in remaining)
    assert space.generated == 0
    assert space.sample_configs(random.Random(1), 0) == []
    with pytest.raises(ValueError):
        space.sample_configs(random.Random(1), -1)


def test_sampling_returns_only_the_unexcluded_remainder_of_small_categorical_space():
    space = ConfigSpace({"tile": (8, 16), "layout": ("row", "col"), "fused": (False, True)})
    configs = tuple(space.iter_configs())
    allowed = configs[-1]
    result = space.sample_configs(random.Random(91), 100, exclude_ids={config.id for config in configs[:-1]})
    assert result == [allowed]
    assert space.sample_configs(random.Random(91), 100, exclude_ids={config.id for config in configs}) == []


@pytest.mark.parametrize("algorithm", SEARCHER_NAMES)
def test_searchers_use_bounded_candidates_on_a_million_configuration_space(algorithm):
    space = CountingSpace({"x": tuple(range(1000)), "y": tuple(range(1000))})
    context = _record(KernelConfig({"x": 0, "y": 0}))
    searcher = make_searcher(algorithm, seed=31, candidate_pool_size=64)
    history = []
    attempted = set()
    assert searcher.suggest(context.hardware, context.operator, context.shape, space, SearchHistory(),
                            SearchBudget(0)) == []
    for index in range(8):
        candidates = searcher.suggest(context.hardware, context.operator, context.shape, space,
                                      SearchHistory(tuple(history), frozenset(attempted)),
                                      SearchBudget(8 - index), batch_size=1)
        assert len(candidates) == 1
        config = candidates[0]
        assert config.id not in attempted
        assert space.validate(config, context.hardware, context.operator, context.shape).valid
        attempted.add(config.id)
        history.append(_record(config, latency=1 + index / 8))
    assert len(attempted) == 8
    assert space.generated <= 4096 and space.validations <= 8192


def test_partial_replay_respects_budget_without_materializing_million_candidate_table():
    space = CountingSpace({"x": tuple(range(1000)), "y": tuple(range(1000))})
    rows = tuple(_record(KernelConfig({"x": 0, "y": index}), latency=1 + index / 100) for index in range(32))
    result = run_replay(rows, make_searcher("random", seed=8, candidate_pool_size=64), space, SearchBudget(8))
    assert result.evaluations == len(result.trace) == 8
    assert result.source_records == 32
    assert result.failed_evaluations == 0
    assert all(step.status in {"success", "missing"} for step in result.trace)
    assert len({step.config.id for step in result.trace}) == 8
    assert space.generated <= 4096 and space.validations <= 8192


def _gemm_context():
    hardware = get_hardware_profile("4090")
    operator = operator_spec("gemm")
    shape = ShapeSpec({"M": 127, "N": 129, "K": 65})
    space = operator_space("gemm", small=True)
    return hardware, operator, shape, space


def test_online_invalid_and_repeated_candidates_are_bounded_and_never_recommended(tmp_path):
    hardware, operator, shape, space = _gemm_context()
    valid = next(space.iter_configs())

    class InvalidThenRepeated:
        name = "invalid-and-repeated-fixture"

        def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
            index = len(history)
            return ["wrong type" if index == 0 else KernelConfig({"unknown": 1}) if index == 1 else valid]

    with ExperimentStore(tmp_path / "attempts.sqlite") as store:
        runner = BenchmarkRunner(MockAdapter(0, 0, 0), store, repetitions=2)
        result = tune(runner, hardware, operator, shape, space, InvalidThenRepeated(), SearchBudget(16))
        assert result.evaluations == result.attempts == 3
        assert result.cache_hits == 0 and result.stop_reason == "repeated_candidate"
        assert [row.status for row in result.records] == [RecordStatus.INVALID_CONFIG, RecordStatus.INVALID_CONFIG,
                                                        RecordStatus.SUCCESS]
        assert result.best_config.id == valid.id
        assert len(store.records()) == 3


def test_online_repeated_cached_candidate_stops_and_zero_budget_reveals_nothing(tmp_path):
    hardware, operator, shape, space = _gemm_context()
    config = next(space.iter_configs())

    class AlwaysSame:
        name = "repeat-cache-fixture"
        calls = 0

        def suggest(self, *args, **kwargs):
            self.calls += 1
            return [config]

    with ExperimentStore(tmp_path / "cache.sqlite") as store:
        runner = BenchmarkRunner(MockAdapter(0, 0, 0), store, repetitions=2)
        original = runner.measure(hardware, operator, shape, config, space)
        searcher = AlwaysSame()
        zero = tune(runner, hardware, operator, shape, space, searcher, SearchBudget(0))
        assert zero.records == () and zero.cache_hits == zero.evaluations == 0
        assert searcher.calls == 0
        repeated = tune(runner, hardware, operator, shape, space, searcher, SearchBudget(16))
        assert repeated.attempts == repeated.cache_hits == 1 and repeated.evaluations == 0
        assert repeated.records[0].id == original.id
        assert repeated.stop_reason == "repeated_candidate"
        assert len(store.records()) == 1


def test_online_does_not_measure_after_searcher_spends_wall_budget(monkeypatch):
    import kernel_tuning_lab.tuning as implementation

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(implementation, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    hardware, operator, shape, space = _gemm_context()

    class SlowSuggestion:
        name = "budget-clock-fixture"

        def suggest(self, *args, **kwargs):
            clock.now = 2.0
            return [next(space.iter_configs())]

    class NoMeasurementAllowed:
        def measure(self, *args, **kwargs):
            raise AssertionError("measurement admitted after the wall budget was spent")

    result = tune(NoMeasurementAllowed(), hardware, operator, shape, space, SlowSuggestion(), SearchBudget(10, 1.0))
    assert result.evaluations == result.attempts == 0
    assert result.stop_reason == "wall_time_budget" and result.wall_time_s == 2.0


def test_sixty_four_shape_forty_eight_config_l3_curve_and_dispatch_use_observed_objective():
    rng = random.Random(73)
    shapes = tuple(ShapeSpec({"M": index + 1, "N": 129}) for index in range(64))
    weights = tuple(0.0 if index == 63 else (index % 7) + 1 for index in range(64))
    workload = WorkloadDistribution(shapes, weights, name="synthetic-64-shape-workload")
    configs = tuple(KernelConfig({"variant": index}) for index in range(48))
    latencies = tuple(tuple(0.5 + 0.15 * abs(row % 16 - col % 16) + 0.03 * (col // 16)
                            + rng.random() * 0.01 for col in range(48)) for row in range(64))
    matrix = PerformanceMatrix(workload, configs, latencies, provenance="mock")
    exact_one = oracle(matrix, 1, max_combinations=48)
    independent_one = min(math.fsum(weight * row[column] for weight, row in zip(workload.normalized_weights, latencies))
                          for column in range(48))
    assert exact_one.expected_latency_ms == pytest.approx(independent_one)
    with pytest.raises(OracleLimitError):
        oracle(matrix, 8, max_combinations=1000)
    previous_greedy = math.inf
    hardware, operator = HardwareSpec("MOCK-l3-scale"), OperatorSpec("variant-fixture", kind="probe")
    space = ConfigSpace({"variant": tuple(range(48))})
    for k in (1, 2, 4, 8, 16):
        baseline = greedy(matrix, k)
        improved = greedy_local_swap(matrix, k, max_passes=8)
        assert baseline.expected_latency_ms <= previous_greedy + 1e-12
        assert improved.expected_latency_ms <= baseline.expected_latency_ms + 1e-12
        previous_greedy = baseline.expected_latency_ms
        for result in (baseline, improved):
            assert result.coverage == 1 and not result.exact
            assert result.variant_count <= k
            assert len(set(result.selected_indices)) == result.variant_count
            independent = math.fsum(weight * min(row[index] for index in result.selected_indices)
                                    for weight, row in zip(workload.normalized_weights, latencies))
            assert result.expected_latency_ms == pytest.approx(independent)
            assert result.relative_loss == pytest.approx(independent / matrix.reference_latency_ms - 1)
            dispatcher = ShapeDispatcher.from_result(matrix, result, hardware=hardware, operator=operator, space=space)
            for row_index, shape in enumerate(shapes):
                config = dispatcher(shape)
                assert config.id in {selected.id for selected in result.configs}
                assert latencies[row_index][config.get("variant")] == min(latencies[row_index][index]
                                                                          for index in result.selected_indices)
