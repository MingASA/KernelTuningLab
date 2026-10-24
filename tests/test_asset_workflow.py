"""End-to-end extension acceptance through external SDK-only plugins and CLI."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _write_plugin(path: Path, *, runtime="python", bad_output=False, bad_algorithm=False,
                  slow=False, claim=False):
    launch_body = ('''ctx.outputs["out"][0][0] = float("nan")''' if bad_output else
                   '''ctx.outputs["out"][0][0] = ctx.inputs["x"][0][0]''')
    suggest_body = ('''if budget.exhausted(0, 0): return []
        return [KernelConfig({"tile": 99})]''' if bad_algorithm else
                    '''if budget.exhausted(0, 0): return []
        seen = {record.config.id for record in history} | set(getattr(history, "attempted_config_ids", ()))
        result = []
        for config in space.iter_configs():
            if config.id in seen or not space.validate(config, hardware, operator, shape).valid:
                continue
            result.append(config)
            seen.add(config.id)
            if len(result) >= batch_size: break
        return result''')
    slow_body = '''import time; time.sleep(30)''' if slow else '''pass'''
    claim_block = '''
from dataclasses import replace
from kernel_tuning_lab.sdk import declare_correctness
claim = declare_correctness(kernel, operator, id="example-claim", source="test submitter",
    evidence=["external-fixture"], scope={"runtimes": ["python"], "dtypes": ["float32"],
    "architectures": ["cpu"], "shapes": "*", "configs": "*", "parameters": "*"})
kernel = replace(kernel, claims=(claim,))
''' if claim else "\n"
    path.write_text(f'''from kernel_tuning_lab.sdk import (
    KernelDefinition, OperatorDefinition, ParameterSpec, SearchAlgorithmDefinition,
    SearchCapabilities, ShapeSpec, SpaceDefinition, KernelConfig,
)

def generate_inputs(ctx, seed):
    {slow_body}
    ctx.allocate_output("out", (ctx.shape.get("M"), ctx.shape.get("N")), dtype="float32")
    return {{"x": [[1.0 for _ in range(ctx.shape.get("N"))]
                   for _ in range(ctx.shape.get("M"))]}}

def reference(ctx):
    return {{"out": ctx.inputs["x"]}}

operator = OperatorDefinition(id="external_toy", version="1", shape_dims=("M", "N"),
    dtypes=("float32",), generate_inputs=generate_inputs, reference=reference,
    tolerances={{"float32": (0.0, 0.0)}},
    validation_shapes=(ShapeSpec({{"M": 1, "N": 1}}),))

def build(ctx, tile=1):
    def launch():
        {launch_body}
    return launch

kernel = KernelDefinition(id="external_toy/python", version="1", operator=operator.ref,
    build=build, space=SpaceDefinition({{"tile": ParameterSpec((1, 2))}}),
    runtime={runtime!r}, dtypes=("float32",), architectures=("cpu",) if {runtime!r} == "python" else ("sm_*",))
{claim_block}
class Searcher:
    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        {suggest_body}

def make_searcher(seed=0):
    return Searcher()

algorithm = SearchAlgorithmDefinition("external_first", "1", make_searcher,
                                       SearchCapabilities())
algorithm_v2 = SearchAlgorithmDefinition("external_first", "2", make_searcher,
                                          SearchCapabilities())
ASSETS = [operator, kernel, algorithm, algorithm_v2]
''', encoding="utf-8")


def _spec(tmp_path: Path, plugin: Path, *, preflight_seconds=10.0, hardware=None,
          validation_seconds=5.0, kernel_ref="external_toy/python@1", claim=False,
          budget_evaluations=1):
    value = {
        "schema_version": 2,
        "name": "external-sdk-workflow",
        "operator": "external_toy@1",
        "kernels": [{"asset": kernel_ref}],
        "algorithms": [{"asset": "external_first@1"}],
        "shapes": [{"M": 1, "N": 1}],
        "plugins": [str(plugin.resolve())],
        "dtype": "float32",
        "hardware": hardware,
        "budget": {"max_evaluations": budget_evaluations, "max_wall_time_s": 5.0},
        "validation": {"max_configs": 2, "max_cases": 1,
                       "max_wall_time_s": validation_seconds, "trust_claims": claim},
        "preflight_seconds": preflight_seconds,
        "seed": 11,
        "warmup": 0,
        "repetitions": 1,
    }
    target = tmp_path / "experiment.json"
    target.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return target


def _cli(action: str, spec: Path, output: Path, timeout=15):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, "-m", "kernel_tuning_lab", action,
                           "--spec", str(spec), "--output", str(output)],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)


def _fixture(tmp_path, **options):
    plugin = tmp_path / "external_assets.py"
    preflight_seconds = options.pop("preflight_seconds", 10.0)
    _write_plugin(plugin, **options)
    spec = _spec(tmp_path, plugin, preflight_seconds=preflight_seconds)
    return plugin, spec


def test_sdk_only_external_plugin_cpu_run_archives_complete_measurement_flow(tmp_path):
    plugin, spec = _fixture(tmp_path)
    output = tmp_path / "run"
    result = _cli("run", spec, output)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "run_summary.json").read_text())
    assert summary["status"] == "passed"
    for name in ("preflight.json", "acceptance.json", "assets.lock.json", "measurements.sqlite",
                 "measurements.jsonl", "trials.json"):
        assert (output / name).is_file(), name
    records = [json.loads(line) for line in (output / "measurements.jsonl").read_text().splitlines() if line]
    assert records
    assert all(record["provenance"] == "measured" for record in records)
    assert all(record["environment"].get("runtime") == "python" for record in records)
    assert all(record["metadata"]["correctness"]["verified_on_cpu"] is True for record in records)
    assert all(record["metadata"]["correctness"]["verified_on_gpu"] is False for record in records)


def test_archived_experiment_is_self_contained_for_a_followup_run(tmp_path):
    _, spec = _fixture(tmp_path)
    first = tmp_path / "first-run"
    result = _cli("run", spec, first)
    assert result.returncode == 0, result.stdout + result.stderr
    archived = first / "experiment.json"
    archived_spec = json.loads(archived.read_text())
    assert Path(archived_spec["plugins"][0]).is_absolute()
    second = tmp_path / "reproduced-run"
    reproduced = _cli("run", archived, second)
    assert reproduced.returncode == 0, reproduced.stdout + reproduced.stderr
    summary = json.loads((second / "run_summary.json").read_text())
    assert summary["status"] == "passed"
    assert json.loads((second / "source_experiment.json").read_text())["plugins"] == archived_spec["plugins"]


def test_check_runs_static_and_synthetic_algorithm_acceptance_without_research_measurements(tmp_path):
    _, spec = _fixture(tmp_path)
    output = tmp_path / "check"
    result = _cli("check", spec, output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "preflight.json").is_file()
    assert (output / "acceptance.json").is_file()
    assert (output / "assets.lock.json").is_file()
    assert (output / "measurements.sqlite").is_file()
    assert (output / "measurements.jsonl").is_file()
    assert json.loads((output / "trials.json").read_text()) == []
    summary = json.loads((output / "run_summary.json").read_text())
    assert summary["check_only"] is True
    assert summary["evaluations"] == 0


def test_same_kernel_multiple_algorithm_versions_reuse_compatible_measurements(tmp_path):
    plugin, spec = _fixture(tmp_path)
    value = json.loads(spec.read_text())
    value["algorithms"].append({"asset": "external_first@2"})
    spec.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "multi-algorithm"
    result = _cli("run", spec, output)
    assert result.returncode == 0, result.stdout + result.stderr
    trials = json.loads((output / "trials.json").read_text())
    assert len(trials) == 2
    assert trials[0]["algorithm_fingerprint"] != trials[1]["algorithm_fingerprint"]
    assert trials[1]["cache_hits"] >= 1
    assert all(trial["observations"] <= 1 for trial in trials)
    assert all(trial["physical_evaluations"] + trial["cache_hits"] <= 1 for trial in trials)
    assert len(set(trials[0]["record_ids"]) & set(trials[1]["record_ids"])) >= 1


def test_zero_research_budget_performs_no_search_evaluations(tmp_path):
    plugin = tmp_path / "zero_budget_assets.py"
    _write_plugin(plugin)
    spec = _spec(tmp_path, plugin, budget_evaluations=0)
    output = tmp_path / "zero-budget"
    result = _cli("run", spec, output)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "run_summary.json").read_text())
    trials = json.loads((output / "trials.json").read_text())
    assert summary["evaluations"] == 0
    assert trials
    assert all(trial["evaluations"] == 0 for trial in trials)
    assert all(trial["stop_reason"] == "evaluation_budget" for trial in trials)


def test_wrong_output_is_archived_as_failed_instead_of_success(tmp_path):
    _, spec = _fixture(tmp_path, bad_output=True)
    output = tmp_path / "wrong-output"
    result = _cli("run", spec, output)
    assert result.returncode == 2, result.stdout + result.stderr
    summary = json.loads((output / "run_summary.json").read_text())
    assert summary["status"] == "failed"
    records = [json.loads(line) for line in (output / "measurements.jsonl").read_text().splitlines() if line]
    assert records
    assert all(record["status"] != "success" for record in records)


def test_incompatible_algorithm_proposals_are_reported_by_check(tmp_path):
    _, spec = _fixture(tmp_path, bad_algorithm=True)
    output = tmp_path / "bad-algorithm"
    result = _cli("check", spec, output)
    assert result.returncode == 2, result.stdout + result.stderr
    acceptance = json.loads((output / "acceptance.json").read_text())
    assert acceptance["status"] == "failed"
    assert acceptance.get("reason")
    assert acceptance.get("preflight_evidence") or acceptance.get("error")
    preflight = json.loads((output / "preflight.json").read_text())
    failures = [check for check in preflight["checks"] if check.get("status") == "failed"]
    assert failures
    assert "outside its domain" in json.dumps(failures).lower()


def test_declared_correctness_is_archived_as_claim_not_verified_oracle(tmp_path):
    _, spec = _fixture(tmp_path, claim=True)
    # Claims are trusted only because the experiment explicitly opts in.
    value = json.loads(spec.read_text())
    value["validation"]["trust_claims"] = True
    spec.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "claimed"
    result = _cli("run", spec, output)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in (output / "measurements.jsonl").read_text().splitlines() if line]
    assert records
    trial = json.loads((output / "trials.json").read_text())[0]
    assert trial["correctness_origin"] == "submitter_claim"
    assert trial["recommendation_verified"] is False
    assert trial["best_verified_record_id"] is None
    for record in records:
        correctness = record["metadata"]["correctness"]
        assert correctness["correctness_origin"] == "submitter_claim"
        assert correctness["comparison_performed"] is False
        assert correctness["verified_on_cpu"] is False
        assert correctness["verified_on_gpu"] is False
        assert correctness.get("oracle_eligible", False) is False


def test_missing_cuda_is_deferred_and_keeps_static_and_algorithm_evidence(tmp_path):
    plugin = tmp_path / "gpu_assets.py"
    _write_plugin(plugin, runtime="cuda")
    spec = _spec(tmp_path, plugin, hardware="h200-sxm")
    output = tmp_path / "gpu-deferred"
    result = _cli("run", spec, output)
    assert result.returncode == 3, result.stdout + result.stderr
    summary = json.loads((output / "run_summary.json").read_text())
    assert summary["status"] == "deferred"
    assert (output / "preflight.json").is_file()
    assert (output / "acceptance.json").is_file()
    assert (output / "assets.lock.json").is_file()


def test_hard_preflight_timeout_archives_worker_evidence_and_removes_stale_success(tmp_path):
    plugin = tmp_path / "slow_assets.py"
    _write_plugin(plugin, slow=True)
    spec = _spec(tmp_path, plugin, preflight_seconds=0.2)
    output = tmp_path / "timeout"
    output.mkdir()
    # A stale completion marker from an earlier attempt must never survive a timeout.
    (output / "run_summary.json").write_text('{"status":"passed"}\n', encoding="utf-8")
    result = _cli("run", spec, output, timeout=12)
    assert result.returncode == 124, result.stdout + result.stderr
    summary = json.loads((output / "run_summary.json").read_text())
    assert summary["status"] == "timeout"
    assert (output / "worker.log").is_file()
    assert (output / "preflight.json").is_file()
    assert (output / "acceptance.json").is_file()


def test_validation_timeout_archives_interrupted_sample_and_configuration(tmp_path):
    plugin = tmp_path / "slow_validation_assets.py"
    _write_plugin(plugin, slow=True)
    spec = _spec(tmp_path, plugin, validation_seconds=0.2)
    output = tmp_path / "validation-timeout"
    result = _cli("run", spec, output, timeout=12)
    assert result.returncode == 124, result.stdout + result.stderr
    summary = json.loads((output / "run_summary.json").read_text())
    interrupted = json.loads((output / "interrupted_attempt.json").read_text())
    assert summary["status"] == "timeout"
    assert summary["phase"] == "validation"
    assert interrupted["status"] == "timeout"
    assert interrupted["progress"]["phase"] == "validation"
    assert interrupted["progress"]["config"]
    assert (output / "worker.log").is_file()

