"""Contract regressions for the result journal and budgeted plugin runner."""

from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest

from kernel_tuning_lab.models import (
    HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec,
    RecordStatus, SearchBudget, ShapeSpec, WorkloadDistribution, canonical_json,
)
from kernel_tuning_lab.runner import BenchmarkRunner
from kernel_tuning_lab.space import ConfigSpace, ValidationResult
from kernel_tuning_lab.storage import ExperimentStore


class ProbeAdapter:
    """An observable plugin with independent, explicit outcomes for every phase."""

    provenance = "mock"
    measurement_mode = "contract_probe"
    name = "contract_probe"

    def __init__(self, *, compile_clock=None):
        self.compiles = []
        self.measures = []
        self.compile_clock = compile_clock

    def environment(self):
        return {"plugin": "contract_probe", "version": "1"}

    def protocol_metadata(self):
        return {"synthetic": True}

    def validate(self, hardware, operator, shape, config):
        if config.get("mode") == "invalid":
            return ValidationResult(False, ("explicitly unsupported configuration",))
        return ValidationResult(True)

    def prepare(self, hardware, operator, shape, seed):
        return shape

    def compile(self, prepared, config):
        self.compiles.append(config.get("mode"))
        if self.compile_clock is not None:
            self.compile_clock.now += 2.0
        if config.get("mode") == "compile":
            raise RuntimeError("deliberate compilation failure")
        return config

    def check_correctness(self, compiled):
        return compiled.get("mode") != "correctness", {"reference": "contract_fixture"}

    def warmup(self, compiled, iterations):
        pass

    def measure(self, compiled, repetitions):
        self.measures.append(compiled.get("mode"))
        if compiled.get("mode") == "runtime":
            raise RuntimeError("deliberate runtime failure")
        value = math.nan if compiled.get("mode") == "nan" else 1234.0
        return tuple(value for _ in range(repetitions))


def case():
    return HardwareSpec("contract-hardware"), OperatorSpec("probe", kind="probe"), ShapeSpec({"M": 64})


@pytest.mark.parametrize("mode,status", [
    ("success", RecordStatus.SUCCESS),
    ("invalid", RecordStatus.INVALID_CONFIG),
    ("compile", RecordStatus.COMPILE_ERROR),
    ("correctness", RecordStatus.CORRECTNESS_ERROR),
    ("runtime", RecordStatus.RUNTIME_ERROR),
])
def test_runner_persists_distinct_success_and_failure_outcomes(tmp_path, mode, status):
    hardware, operator, shape = case()
    adapter = ProbeAdapter()
    config = KernelConfig({"mode": mode})
    failure_log = tmp_path / "failures.jsonl"
    with ExperimentStore(tmp_path / "measurements.sqlite") as store:
        runner = BenchmarkRunner(adapter, store, repetitions=3, failure_log_path=failure_log)
        result = runner.measure(hardware, operator, shape, config, ConfigSpace({"mode": (mode,)}))
        persisted = store.records()
        assert result.status == status
        assert len(persisted) == 1 and persisted[0].id == result.id
        assert result.provenance == "mock"
        if status == RecordStatus.SUCCESS:
            # Slow but correct performance is still a successful observation.
            assert result.median_ms == 1234.0
            assert not failure_log.exists()
        else:
            assert result.failure_reason and result.median_ms is None
            logged = json.loads(failure_log.read_text())
            assert logged["status"] == status.value and logged["record_id"] == result.id
        if status in {RecordStatus.INVALID_CONFIG, RecordStatus.COMPILE_ERROR, RecordStatus.CORRECTNESS_ERROR}:
            assert not adapter.measures
        if status == RecordStatus.INVALID_CONFIG:
            assert not adapter.compiles


def test_bad_runtime_sample_is_a_persisted_runtime_failure(tmp_path):
    hardware, operator, shape = case()
    with ExperimentStore(tmp_path / "bad-sample.sqlite") as store:
        runner = BenchmarkRunner(ProbeAdapter(), store, repetitions=2)
        result = runner.measure(hardware, operator, shape, KernelConfig({"mode": "nan"}), ConfigSpace({"mode": ("nan",)}))
        assert result.status == RecordStatus.RUNTIME_ERROR
        assert result.runtime_samples_ms == ()
        assert "finite positive" in result.failure_reason
        assert store.records(status=RecordStatus.RUNTIME_ERROR)[0].id == result.id
        # Failure remains exportable as strict JSON, including the traceback.
        assert store.export_jsonl(tmp_path / "bad-sample.jsonl") == 1


def test_resume_returns_records_without_repeating_physical_measurements(tmp_path):
    hardware, operator, shape = case()
    adapter = ProbeAdapter()
    space = ConfigSpace({"mode": ("a", "b", "c")})
    with ExperimentStore(tmp_path / "resume.sqlite") as store:
        runner = BenchmarkRunner(adapter, store, repetitions=2)
        first = runner.run(hardware, operator, (shape,), space, SearchBudget(3))
        assert len(adapter.measures) == 3
        resumed = runner.run(hardware, operator, (shape,), space, SearchBudget(3))
        assert [record.id for record in resumed] == [record.id for record in first]
        assert len(adapter.measures) == 3 and len(adapter.compiles) == 3
        assert runner.last_run_stats["evaluations"] == 0
        assert runner.last_run_stats["cache_hits"] == 3
        assert len(store.records()) == 3


def test_tightening_same_constraint_rejects_cached_success(tmp_path):
    hardware, operator, shape = case()
    restriction = {"reject": False}

    def mutable_constraint(config, hardware, operator, shape):
        return "new hardware restriction" if restriction["reject"] else None

    # The callable name and domains remain identical across the change.
    space = ConfigSpace({"mode": ("success",)}, constraints=(mutable_constraint,))
    adapter = ProbeAdapter()
    with ExperimentStore(tmp_path / "constraints.sqlite") as store:
        runner = BenchmarkRunner(adapter, store, repetitions=2)
        original = runner.measure(hardware, operator, shape, KernelConfig({"mode": "success"}), space)
        assert original.successful
        restriction["reject"] = True
        tightened = runner.measure(hardware, operator, shape, original.config, space)
        assert tightened.status == RecordStatus.INVALID_CONFIG
        assert "new hardware restriction" in tightened.failure_reason
        assert not runner.last_cache_hit
        assert len(adapter.measures) == 1
        assert {record.status for record in store.records()} == {RecordStatus.SUCCESS, RecordStatus.INVALID_CONFIG}


def test_budget_zero_count_and_exact_time_boundaries(tmp_path):
    hardware, operator, shape = case()
    budget = SearchBudget(2, 5.0)
    assert not budget.exhausted(1, 4.999)
    assert budget.exhausted(2, 0.0)
    assert budget.exhausted(0, 5.0)
    assert budget.remaining(1, 2.0) == SearchBudget(1, 3.0)
    assert budget.remaining(8, 9.0) == SearchBudget(0, 0.0)
    adapter = ProbeAdapter()
    space = ConfigSpace({"mode": ("invalid", "compile", "runtime", "success")})
    with ExperimentStore(tmp_path / "budgets.sqlite") as store:
        runner = BenchmarkRunner(adapter, store, repetitions=2)
        assert runner.run(hardware, operator, (shape,), space, SearchBudget(0)) == []
        assert runner.run(hardware, operator, (shape,), space, SearchBudget(10, 0.0)) == []
        assert not adapter.compiles and not store.records()
        results = runner.run(hardware, operator, (shape,), space, SearchBudget(2))
        assert [record.status for record in results] == [RecordStatus.INVALID_CONFIG, RecordStatus.COMPILE_ERROR]
        assert runner.last_run_stats["evaluations"] == 2
        assert runner.last_run_stats["budget_exhausted"]
        assert len(store.records()) == 2


def test_running_compilation_may_overrun_time_admission_budget(tmp_path, monkeypatch):
    import kernel_tuning_lab.runner as runner_module

    class Clock:
        now = 0.0

        def read(self):
            return self.now

    clock = Clock()
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(perf_counter=clock.read))
    hardware, operator, shape = case()
    adapter = ProbeAdapter(compile_clock=clock)
    with ExperimentStore(tmp_path / "time-budget.sqlite") as store:
        runner = BenchmarkRunner(adapter, store, repetitions=2)
        results = runner.run(hardware, operator, (shape,), ConfigSpace({"mode": ("a", "b")}), SearchBudget(10, 1.0))
        assert len(results) == 1 and results[0].successful
        assert results[0].compile_time_ms == 2000.0
        assert len(adapter.measures) == 1
        assert runner.last_run_stats["time_budget_overrun"]
        assert runner.last_run_stats["budget_exhausted"]


def test_jsonl_roundtrip_dedup_and_case_split_conflicts_are_atomic(tmp_path):
    hardware, operator, shape = case()
    record = MeasurementRecord(hardware, operator, shape, KernelConfig({"mode": "a"}),
                               RecordStatus.SUCCESS, runtime_samples_ms=(1.0, 2.0, 3.0),
                               repetitions=3, provenance="mock", split="train")
    exchange = tmp_path / "records.jsonl"
    with ExperimentStore(tmp_path / "source.sqlite") as source:
        assert source.add(record)
        assert not source.add(record)
        assert source.export_jsonl(exchange) == 1
    with ExperimentStore(tmp_path / "destination.sqlite") as destination:
        assert destination.import_jsonl(exchange) == {"read": 1, "added": 1, "duplicates": 0}
        assert destination.import_jsonl(exchange) == {"read": 1, "added": 0, "duplicates": 1}
        assert canonical_json(destination.records()[0].to_dict()) == canonical_json(record.to_dict())
        conflict = replace(record, config=KernelConfig({"mode": "b"}), split="test")
        with pytest.raises(ValueError, match="split"):
            destination.add(conflict)
        unrelated = replace(record, shape=ShapeSpec({"M": 128}))
        conflicting_file = tmp_path / "conflicting.jsonl"
        conflicting_file.write_text(canonical_json(unrelated.to_dict()) + "\n" + canonical_json(conflict.to_dict()) + "\n")
        with pytest.raises(ValueError, match="split"):
            destination.import_jsonl(conflicting_file)
        assert [saved.id for saved in destination.records()] == [record.id]


def test_huge_finite_weights_normalize_without_overflow():
    shapes = tuple(ShapeSpec({"M": i}) for i in (1, 2, 3))
    distribution = WorkloadDistribution(shapes, (1e308, 0.0, 1e308))
    assert distribution.normalized_weights == (0.5, 0.0, 0.5)
    assert math.fsum(distribution.normalized_weights) == 1.0
    assert all(math.isfinite(weight) for weight in distribution.normalized_weights)
