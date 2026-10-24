"""Regression contracts for replay evidence, bounded suggestions and cache recovery."""

import sqlite3

import pytest

from kernel_tuning_lab.models import HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec, RecordStatus, SearchBudget, ShapeSpec
from kernel_tuning_lab.replay import run_replay
from kernel_tuning_lab.search import GridSearch
from kernel_tuning_lab.space import ConfigSpace
from kernel_tuning_lab.storage import ExperimentStore
from kernel_tuning_lab.tuning import tune


def _case():
    hardware = HardwareSpec("CPU-test")
    operator = OperatorSpec("replay-test")
    shape = ShapeSpec({"M": 8})
    space = ConfigSpace({"x": (1, 2, 3, 4)})
    rows = []
    for x in range(1, 5):
        rows.append(MeasurementRecord(
            hardware, operator, shape, KernelConfig({"x": x}), RecordStatus.SUCCESS,
            runtime_samples_ms=(float(x + 1),), provenance="mock", split="test",
            metadata={"correctness_verified": x == 4,
                      "correctness": {"status": "verified" if x == 4 else "declared",
                                      "comparison_performed": x == 4}}))
    return hardware, operator, shape, space, tuple(rows)


@pytest.mark.parametrize("value", [True, "2.5"])
def test_search_budget_rejects_non_numeric_wall_time_as_value_error(value):
    with pytest.raises(ValueError):
        SearchBudget(max_wall_time_s=value)


def test_top_level_submitter_claim_origin_never_counts_as_verified_success():
    hardware, operator, shape, _, _ = _case()
    record = MeasurementRecord(hardware, operator, shape, KernelConfig({"x": 1}), RecordStatus.SUCCESS,
                               runtime_samples_ms=(1.0,), metadata={"correctness_origin": "submitter_claim"})
    assert not record.correctness_verified


class _Ordered:
    name = "ordered"
    def __init__(self, configs): self.configs = configs
    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        if budget.exhausted(0, 0): return []
        seen = set(history.attempted_config_ids)
        return [config for config in self.configs if config.id not in seen][:batch_size]


@pytest.mark.parametrize("status", [RecordStatus.COMPILE_ERROR, RecordStatus.CORRECTNESS_ERROR,
                                    RecordStatus.RUNTIME_ERROR, RecordStatus.CORRECTNESS_UNKNOWN])
def test_replay_preserves_failure_evidence_but_never_scores_it_as_best(status):
    _, _, _, space, rows = _case()
    failed = rows[0].__class__(
        rows[0].hardware, rows[0].operator, rows[0].shape, rows[0].config, status,
        runtime_samples_ms=(0.0001,), failure_reason="captured failure", wall_time_s=0.25,
        provenance="mock", split="test", metadata={"correctness_verified": False})
    result = run_replay((failed, *rows[1:]), _Ordered([failed.config, rows[-1].config]), space, SearchBudget(2))
    assert result.trace[0].status == status.value
    assert result.trace[0].failure_reason == "captured failure"
    assert result.failed_evaluations == 1
    assert result.best_config == rows[-1].config
    assert result.oracle_best_latency_ms == rows[-1].median_ms
    assert result.known_measurement_cost_s == 0.25
    assert result.cost_coverage == 0.5


def test_unverified_success_is_revealed_but_excluded_from_replay_optimum():
    _, _, _, space, rows = _case()
    result = run_replay(rows, _Ordered([rows[0].config, rows[-1].config]), space, SearchBudget(2))
    assert result.trace[0].status == "missing"
    assert result.oracle_best_latency_ms == rows[-1].median_ms
    assert result.best_latency_ms == rows[-1].median_ms
    assert result.best_config == rows[-1].config


def test_replay_and_tune_reject_overproducing_generators_without_consuming_excess():
    _, _, _, space, rows = _case()
    class ExplodingGenerator:
        def suggest(self, *args, **kwargs):
            def proposals():
                yield rows[0].config
                yield rows[1].config
                raise AssertionError("third proposal must not be read")
            return proposals()
    replay = run_replay(rows, ExplodingGenerator(), space, SearchBudget(1))
    assert replay.evaluations == 0
    assert replay.stop_reason == "searcher_error:ValueError:searcher exceeded requested batch_size=1"

    class FakeRunner:
        last_cache_hit = False
        def measure(self, *args, **kwargs):
            raise AssertionError("overproduced suggestions must not be evaluated")

    online = tune(FakeRunner(), rows[0].hardware, rows[0].operator, rows[0].shape,
                  space, ExplodingGenerator(), SearchBudget(1))
    assert online.attempts == 0
    assert online.stop_reason == "searcher_error:ValueError:searcher exceeded requested batch_size=1"


def test_cache_ignores_interrupted_latest_row_and_keeps_completed_record(tmp_path):
    _, _, _, _, rows = _case()
    completed = rows[-1]
    interrupted = completed.__class__(**{**completed.__dict__, "metadata": {"terminated_by_parent": True}})
    with ExperimentStore(tmp_path / "cache.sqlite") as store:
        store.add(completed)
        store.add(interrupted)
        assert store.get_cached(completed.cache_key).id == completed.id
        only_interrupted = rows[-2].__class__(**{**rows[-2].__dict__, "metadata": {"terminated_by_parent": True}})
        store.add(only_interrupted)
        assert store.get_cached(only_interrupted.cache_key) is None
        assert any(row.id == only_interrupted.id for row in store.records())


@pytest.mark.parametrize("schema_sql", [
    "CREATE TABLE schema_info(version INTEGER PRIMARY KEY); INSERT INTO schema_info VALUES (2);",
    "CREATE TABLE unrelated(value TEXT);",
])
def test_readonly_store_rejects_unknown_database_without_mutation(tmp_path, schema_sql):
    path = tmp_path / "foreign.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(schema_sql)
    connection.commit()
    before = connection.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    connection.close()
    with pytest.raises(ValueError):
        ExperimentStore(path, read_only=True)
    connection = sqlite3.connect(path)
    after = connection.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    connection.close()
    assert after == before
