"""Lifecycle and input-boundary contracts for asset experiments."""

import json
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from kernel_tuning_lab.asset_experiments import AssetExperiment
import kernel_tuning_lab.asset_runner as asset_runner
from kernel_tuning_lab.run_lifecycle import supervised_worker


ROOT = Path(__file__).resolve().parents[1]


def _spec_dict(plugin: Path, **updates):
    value = {
        "schema_version": 2, "name": "lifecycle", "operator": "external_toy@1",
        "kernels": [{"asset": "external_toy/python@1"}],
        "algorithms": [{"asset": "external_first@1"}],
        "shapes": [{"M": 1, "N": 1}], "plugins": [str(plugin.resolve())],
        "dtype": "float32", "hardware": None,
        "budget": {"max_evaluations": 1, "max_wall_time_s": 5.0},
        "validation": {"max_configs": 2, "max_cases": 1,
                       "max_wall_time_s": 5.0, "trust_claims": True},
        "preflight_seconds": 10.0, "seed": 3, "warmup": 0, "repetitions": 1,
    }
    value.update(updates)
    return value


def _plugin(path: Path, *, delay=0.0):
    path.write_text(f'''import time
from kernel_tuning_lab.sdk import KernelDefinition, OperatorDefinition, ParameterSpec, SearchAlgorithmDefinition, SearchCapabilities, ShapeSpec, SpaceDefinition, KernelConfig
def generate_inputs(ctx, seed):
    time.sleep({delay!r})
    ctx.allocate_output("out", (1, 1), dtype="float32")
    return {{"x": [[1.0]]}}
def reference(ctx): return {{"out": ctx.inputs["x"]}}
operator = OperatorDefinition(id="external_toy", version="1", shape_dims=("M", "N"), dtypes=("float32",), generate_inputs=generate_inputs, reference=reference, tolerances={{"float32": (0.0, 0.0)}}, validation_shapes=(ShapeSpec({{"M": 1, "N": 1}}),))
def build(ctx, tile=1):
    def launch(): ctx.outputs["out"][0][0] = ctx.inputs["x"][0][0]
    return launch
kernel = KernelDefinition(id="external_toy/python", version="1", operator=operator.ref, build=build, space=SpaceDefinition({{"tile": ParameterSpec((1, 2))}}), runtime="python", dtypes=("float32",), architectures=("cpu",))
class Searcher:
    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        if budget.exhausted(0, 0): return []
        if any(record.config.get("tile") == 1 for record in history): return []
        return [KernelConfig({{"tile": 1}})]
def make_searcher(seed=0): return Searcher()
algorithm = SearchAlgorithmDefinition("external_first", "1", make_searcher, SearchCapabilities())
ASSETS = [operator, kernel, algorithm]
''', encoding="utf-8")


def _launch(spec: Path, output: Path, timeout=20):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, "-m", "kernel_tuning_lab", "run", "--spec", str(spec), "--output", str(output)],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)


def _known_good_fixture(tmp_path, *, slow=False):
    # Reuse the established public SDK fixture so lifecycle assertions don't
    # accidentally test a second, subtly different plugin contract.
    fixture_path = ROOT / "tests" / "test_asset_workflow.py"
    module_spec = importlib.util.spec_from_file_location("asset_workflow_fixture", fixture_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module._fixture(tmp_path, slow=slow)


@pytest.mark.parametrize("bad", [
    "duplicate",
    "NaN",
    "Infinity",
    "-Infinity",
])
def test_asset_experiment_load_rejects_non_strict_json(tmp_path, bad):
    path = tmp_path / "bad.json"
    plugin = tmp_path / "plugin.py"
    raw = json.dumps(_spec_dict(plugin))
    if bad == "duplicate":
        raw = raw.replace('"name": "lifecycle"', '"name": "lifecycle", "name": "lifecycle"', 1)
        message = "duplicate"
    else:
        raw = raw.replace('"dtype": "float32"', f'"dtype": "float32", "parameters": {{"scale": {bad}}}', 1)
        message = "nonfinite"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        AssetExperiment.load(path)


@pytest.mark.parametrize("change", [
    lambda d: d.update(budget=[]),
    lambda d: d.update(validation=[]),
    lambda d: d.update(hardware=7),
    lambda d: d.update(unexpected=True),
    lambda d: d["kernels"][0].update(unexpected=True),
    lambda d: d["algorithms"][0].update(unexpected=True),
    lambda d: d.update(selection="not-a-selection"),
])
def test_asset_experiment_rejects_malformed_or_unknown_configuration(tmp_path, change):
    plugin = tmp_path / "plugin.py"
    value = _spec_dict(plugin)
    change(value)
    with pytest.raises((ValueError, TypeError)):
        AssetExperiment.from_dict(value)


def test_asset_experiment_snapshots_nested_input_mappings(tmp_path):
    plugin = tmp_path / "plugin.py"
    value = _spec_dict(plugin)
    experiment = AssetExperiment.from_dict(value)
    value["kernels"][0]["asset"] = "modified"
    value["algorithms"][0]["asset"] = "modified"
    value["plugins"].append("modified")
    value["shapes"][0]["M"] = 99
    assert experiment.kernels[0].asset == "external_toy/python@1"
    assert experiment.algorithms[0].asset == "external_first@1"
    assert experiment.plugins == (str(plugin.resolve()),)
    assert experiment.shapes[0].get("M") == 1


def test_parent_passes_worker_an_immutable_spec_snapshot(tmp_path, monkeypatch):
    plugin = tmp_path / "plugin.py"
    value = _spec_dict(plugin)
    # Relative plugin references are resolved against the source spec directory.
    value["plugins"] = ["plugin.py"]
    source = tmp_path / "input.json"
    source.write_text(json.dumps(value), encoding="utf-8")
    captured = {}

    class FinishedProcess:
        returncode = 0
        def poll(self): return 0

    def fake_popen(command, **kwargs):
        spec_index = command.index("--spec") + 1
        worker_spec = Path(command[spec_index])
        captured["worker_spec"] = worker_spec
        captured["content"] = json.loads(worker_spec.read_text(encoding="utf-8"))
        source.write_text('{"name":"changed while worker starts"}', encoding="utf-8")
        return FinishedProcess()

    monkeypatch.setattr(asset_runner.subprocess, "Popen", fake_popen)
    asset_runner.execute_asset_experiment(source, tmp_path / "output")
    assert captured["worker_spec"] != source
    assert captured["content"]["name"] == "lifecycle"
    assert captured["content"]["plugins"] == [str(plugin.resolve())]


def test_concurrent_same_output_is_rejected_without_overwriting_active_run(tmp_path):
    _, spec = _known_good_fixture(tmp_path, slow=True)
    output = tmp_path / "run"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    first = subprocess.Popen([sys.executable, "-m", "kernel_tuning_lab", "run", "--spec", str(spec), "--output", str(output)],
                             cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (output / "phase.json").exists():
            if first.poll() is not None:
                pytest.fail("first run exited before acquiring/recording its run phase")
            time.sleep(0.03)
        marker = (output / "run_summary.json").read_bytes()
        second = _launch(spec, output)
        assert second.returncode != 0
        assert "OutputDirectoryBusy" in (second.stdout + second.stderr)
        assert (output / "run_summary.json").read_bytes() == marker
    finally:
        if first.poll() is None:
            first.terminate()
        first.communicate(timeout=10)


def test_repeated_run_archives_previous_run_metadata(tmp_path):
    _, spec = _known_good_fixture(tmp_path)
    output = tmp_path / "run"
    first = _launch(spec, output)
    assert first.returncode == 0, first.stdout + first.stderr
    first_summary = json.loads((output / "run_summary.json").read_text())
    second = _launch(spec, output)
    assert second.returncode == 0, second.stdout + second.stderr
    archives = list((output / "attempts").glob("*/run_summary.json"))
    assert archives
    assert any(json.loads(path.read_text()) == first_summary for path in archives)


def test_terminating_parent_releases_output_lease_for_immediate_recovery(tmp_path):
    slow_root, fast_root = tmp_path / "slow-run", tmp_path / "fast-run"
    slow_root.mkdir()
    fast_root.mkdir()
    _, slow_spec = _known_good_fixture(slow_root, slow=True)
    _, fast_spec = _known_good_fixture(fast_root)
    output = tmp_path / "recoverable"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    first = subprocess.Popen([sys.executable, "-m", "kernel_tuning_lab", "run", "--spec", str(slow_spec), "--output", str(output)],
                             cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (output / "phase.json").exists():
            if first.poll() is not None:
                pytest.fail("first run exited before recording its active phase")
            time.sleep(0.03)
        first.send_signal(signal.SIGTERM)
        first.communicate(timeout=10)
        recovered = _launch(fast_spec, output)
        assert recovered.returncode == 0, recovered.stdout + recovered.stderr
        summary = json.loads((output / "run_summary.json").read_text())
        assert summary["status"] == "passed"
        assert list((output / "attempts").glob("*/run_summary.json"))
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate(timeout=10)


@pytest.mark.skipif(os.name != "posix", reason="process group cleanup is POSIX-specific")
def test_supervisor_kills_worker_children_even_after_worker_exits(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"open({str(child_pid_path)!r},'w').write(str(child.pid))"
    )
    worker = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    with supervised_worker(worker):
        worker.wait(timeout=5)
    child_pid = int(child_pid_path.read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            stat = Path(f"/proc/{child_pid}/stat").read_text()
        except FileNotFoundError:
            break
        if stat.split()[2] == "Z":  # killed and awaiting init reaping
            break
        time.sleep(0.03)
    else:
        pytest.fail("worker child survived process-group cleanup")


def test_partial_validation_cannot_be_reported_as_passed(tmp_path):
    _, spec = _known_good_fixture(tmp_path)
    value = json.loads(spec.read_text())
    value["validation"]["max_configs"] = 0
    spec.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "partial-validation"
    completed = _launch(spec, output)
    summary = json.loads((output / "run_summary.json").read_text())
    acceptance = json.loads((output / "acceptance.json").read_text())
    assert acceptance["status"] == "partial"
    assert summary["status"] != "passed"
    assert completed.returncode != 0
