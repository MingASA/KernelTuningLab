"""CPU integration checks spanning collection, holdouts, replay, storage and CLI.

All generated timings are explicitly synthetic.  Hardware profiles here exercise
portable contracts and do not stand in for execution on those physical GPUs.
"""

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import pytest

from kernel_tuning_lab.backends.mock import MockAdapter
from kernel_tuning_lab.datasets import split_records, validate_split_isolation
from kernel_tuning_lab.experiments import example_spec
from kernel_tuning_lab.hardware import get_hardware_profile
from kernel_tuning_lab.l3 import PerformanceMatrix, ShapeDispatcher, greedy, oracle
from kernel_tuning_lab.models import SearchBudget, WorkloadDistribution, canonical_json
from kernel_tuning_lab.operators import OPERATOR_NAMES, operator_demo_shapes, operator_spec
from kernel_tuning_lab.replay import compare_searchers
from kernel_tuning_lab.runner import BenchmarkRunner
from kernel_tuning_lab.search import SEARCHER_NAMES
from kernel_tuning_lab.space import ConfigSpace, gemm_constraint, reduction_constraint
from kernel_tuning_lab.storage import ExperimentStore


PROFILES = ("H200", "H800", "4090")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("operator_name", OPERATOR_NAMES)
def test_eight_evaluation_smoke_verifies_every_shape_on_each_profile(tmp_path, profile, operator_name):
    spec = example_spec("mock", operator_name)
    hardware = get_hardware_profile(profile)
    with ExperimentStore(tmp_path / "smoke.sqlite") as store:
        runner = BenchmarkRunner(MockAdapter(0, 0, 0), store, warmup=1, repetitions=3,
                                 seed=13, experiment_id=f"smoke-{operator_name}-{profile}")
        records = runner.run(hardware, spec.operator, spec.shapes, spec.config_space(), SearchBudget(8))
        assert len(records) == 8
        assert runner.last_run_stats["evaluations"] == 8
        assert all(record.successful for record in records)
        assert {record.shape.id for record in records} == {shape.id for shape in spec.shapes}
        assert set(Counter(record.shape.id for record in records).values()) == {4}
        assert len(store.records()) == 8
        assert all(record.hardware.id == hardware.id and record.provenance == "mock" for record in records)
        for record in records:
            assert record.warmup == 1 and record.repetitions == 3
            assert len(record.runtime_samples_ms) == 3
            assert record.compile_time_ms >= 0 and record.wall_time_s >= 0
            assert record.metadata["protocol"]["synthetic"] is True
            assert record.metadata["correctness"]["verified_on_gpu"] is False
            assert record.quantile_ms(0.1) <= record.median_ms <= record.quantile_ms(0.9)


def _integration_space(operator_name):
    if operator_name in {"gemm", "batched_gemm"}:
        return ConfigSpace({"BLOCK_M": (16, 32, 64), "BLOCK_N": (32, 64), "BLOCK_K": (32,),
                            "num_warps": (4,), "num_stages": (2,)}, constraints=(gemm_constraint,))
    return ConfigSpace({"BLOCK_SIZE": (2048,), "num_warps": (4, 8), "num_stages": (1, 2)},
                       constraints=(reduction_constraint,))


def test_four_operator_three_profile_dataset_runs_through_holdout_replay_and_l3(tmp_path):
    records = []
    for profile in PROFILES:
        hardware = get_hardware_profile(profile)
        for name in OPERATOR_NAMES:
            runner = BenchmarkRunner(MockAdapter(0, 0, 0), warmup=1, repetitions=3, seed=21,
                                     experiment_id="multi-operator-integration")
            records.extend(runner.run(hardware, operator_spec(name), operator_demo_shapes(name),
                                      _integration_space(name)))
    assert len(records) == 480
    assert all(record.successful and record.provenance == "mock" for record in records)
    partitions = split_records(records, seed=37, holdout="hardware")
    assert partitions.manifest() == split_records(reversed(records), seed=37, holdout="hardware").manifest()
    rows = tuple(record for split in (partitions.train, partitions.validation, partitions.test) for record in split)
    validate_split_isolation(rows, holdout="hardware")
    assert all(getattr(partitions, split) for split in ("train", "validation", "test"))
    with ExperimentStore(tmp_path / "dataset.sqlite") as store:
        assert store.add_many(rows) == 480
        assert store.add_many(rows) == 0
        assert store.export_jsonl(tmp_path / "dataset.jsonl") == 480
        assert len(store.records(split="test")) == 160
    with ExperimentStore(tmp_path / "restored.sqlite") as store:
        assert store.import_jsonl(tmp_path / "dataset.jsonl") == {"read": 480, "added": 480, "duplicates": 0}
        restored = store.records()
    assert [canonical_json(row.to_dict()) for row in restored] == [canonical_json(row.to_dict()) for row in rows]

    trials = []
    for name in OPERATOR_NAMES:
        operator_rows = tuple(row for row in restored if row.operator.name == name)
        space = _integration_space(name)
        results = compare_searchers(operator_rows, space, budgets=(2, 4), seeds=(0, 3))
        assert len(results) == 8 * len(SEARCHER_NAMES) * 2 * 2
        assert {row.algorithm for row in results} == set(SEARCHER_NAMES)
        assert all(row.split == "test" and row.provenance == "mock" for row in results)
        assert all(row.evaluations == row.budget for row in results)
        assert all(row.oracle_coverage == 1 and row.best_config is not None for row in results)
        assert all(row.regret is not None and row.regret >= 0 for row in results)
        assert all(len({step.config.id for step in row.trace}) == row.evaluations for row in results)
        assert len({row.hardware_id for row in results}) == 1
        trials.extend(results)

        held_out = tuple(row for row in operator_rows if row.split == "test")
        shapes = operator_demo_shapes(name)
        workload = WorkloadDistribution(shapes, tuple(range(1, 9)), name=f"synthetic-{name}")
        configs = tuple(space.iter_configs())
        matrix = PerformanceMatrix.from_records(workload, configs, held_out)
        assert matrix.complete and matrix.provenance == "mock"
        greedy_result = greedy(matrix, 2)
        exact = oracle(matrix, 2, max_combinations=100)
        assert exact.exact
        assert exact.expected_latency_ms <= greedy_result.expected_latency_ms * (1 + 1e-12)
        dispatcher = ShapeDispatcher.from_result(matrix, exact, hardware=held_out[0].hardware,
                                                 operator=held_out[0].operator, space=space)
        assert len(dispatcher.selected_configs) <= 2
        for shape, index in zip(shapes, exact.assignment_indices):
            assert dispatcher(shape).id == configs[index].id
    assert len(trials) == 768


def _cli(*arguments):
    # -S removes site packages, so this is a real command with no pytest,
    # matplotlib, Torch or Triton installed in its Python import path.
    return subprocess.run([sys.executable, "-S", "-m", "kernel_tuning_lab", *map(str, arguments)],
                          cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=45)


@pytest.mark.parametrize("operator_name,profile", tuple(zip(OPERATOR_NAMES, ("H200", "H800", "4090", "H200"))))
def test_cli_cpu_only_smoke_writes_complete_artifacts(tmp_path, operator_name, profile):
    spec_path = tmp_path / f"{operator_name}.json"
    written = _cli("write-spec", "--operator", operator_name, "--output", spec_path)
    assert written.returncode == 0, written.stderr
    output = tmp_path / "run"
    completed = _cli("benchmark", "--spec", spec_path, "--hardware", profile,
                     "--max-evaluations", 8, "--require-success", "--output", output)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["backend"] == "mock" and summary["provenance"] == "mock"
    assert summary["stats"]["evaluations"] == 8
    assert summary["requested_shapes"] == summary["successful_shapes"] == 2
    expected = ("experiment.json", "measurements.sqlite", "measurements.jsonl", "run_summary.json", "report/report.md")
    assert all((output / path).is_file() for path in expected)
    assert json.loads((output / "run_summary.json").read_text()) == summary
    saved_spec = json.loads((output / "experiment.json").read_text())
    assert saved_spec["operator"]["kind"] == operator_name
    assert saved_spec["hardware"]["name"] == get_hardware_profile(profile).name
    records = [json.loads(line) for line in (output / "measurements.jsonl").read_text().splitlines()]
    assert len(records) == 8 and all(row["provenance"] == "mock" for row in records)
    report = (output / "report/report.md").read_text()
    assert "MOCK / SYNTHETIC DATA" in report
    assert len(list((output / "report").glob("*.svg"))) >= 7
    assert not list((output / "report").glob("*.png"))
    if any(row["status"] != "success" for row in records):
        assert (output / "failures.jsonl").is_file()


def test_cli_online_budget_is_per_shape_and_exchange_is_deduplicated(tmp_path):
    output = tmp_path / "online"
    result = _cli("benchmark", "--operator", "batched_gemm", "--hardware", "4090", "--searcher", "grid",
                  "--max-evaluations", 2, "--output", output)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["stats"]["budget_scope"] == "per_shape"
    assert summary["stats"]["evaluations"] == 4
    tuning = json.loads((output / "tuning.json").read_text())
    assert len(tuning) == 2 and all(row["evaluations"] == 2 for row in tuning)
    exchange = tmp_path / "exchange.jsonl"
    exported = _cli("export", "--database", output / "measurements.sqlite", "--output", exchange)
    assert exported.returncode == 0, exported.stderr
    assert json.loads(exported.stdout)["exported"] == 4
    target = tmp_path / "copy.sqlite"
    imported = _cli("import-jsonl", "--database", target, "--input", exchange)
    repeated = _cli("import-jsonl", "--database", target, "--input", exchange)
    assert imported.returncode == repeated.returncode == 0, (imported.stderr, repeated.stderr)
    assert json.loads(imported.stdout) == {"read": 4, "added": 4, "duplicates": 0}
    assert json.loads(repeated.stdout) == {"read": 4, "added": 0, "duplicates": 4}


def test_cli_require_success_fails_after_saving_inspectable_artifacts(tmp_path):
    output = tmp_path / "insufficient-budget"
    completed = _cli("benchmark", "--operator", "softmax", "--max-evaluations", 1,
                     "--require-success", "--output", output)
    assert completed.returncode == 2
    error = json.loads(completed.stderr)
    assert error["error"] == "ValueError" and "smoke validation failed" in error["message"]
    assert completed.stdout == ""
    assert all((output / path).is_file() for path in
               ("experiment.json", "measurements.sqlite", "measurements.jsonl", "run_summary.json", "report/report.md"))
    summary = json.loads((output / "run_summary.json").read_text())
    assert summary["requested_shapes"] == 2 and summary["successful_shapes"] <= 1
    assert summary["stats"]["evaluations"] == 1


def test_cli_missing_gpu_dependency_returns_structured_error_without_installing_it(tmp_path):
    result = _cli("benchmark", "--backend", "triton", "--operator", "layernorm", "--output", tmp_path / "gpu")
    assert result.returncode == 2
    error = json.loads(result.stderr)
    assert error["error"] == "BackendUnavailableError"
    assert "torch" in error["message"].lower() or "triton" in error["message"].lower()
    assert result.stdout == ""


def test_cli_read_command_does_not_create_a_missing_journal(tmp_path):
    database = tmp_path / "missing" / "dataset.sqlite"
    output = tmp_path / "report"
    result = _cli("report", "--database", database, "--output", output)
    assert result.returncode == 2
    error = json.loads(result.stderr)
    assert error["error"] == "FileNotFoundError" and "does not exist" in error["message"]
    assert result.stdout == ""
    assert not database.exists() and not database.parent.exists()
    assert not output.exists()
