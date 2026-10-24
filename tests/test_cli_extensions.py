"""Public CLI paths for extensibility and cached upstream data access."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from kernel_tuning_lab.adapters import bundled_sample_path, load_kernel_tuner_json
from kernel_tuning_lab.adapters.benchmark_hub import BenchmarkHubEntry, HUB_COMMIT
from kernel_tuning_lab.cli import main
from kernel_tuning_lab.experiments import ExperimentSpec, example_spec
from kernel_tuning_lab.scalability import run_scale_demo
from kernel_tuning_lab.storage import ExperimentStore


def test_portable_spec_preserves_grid_traversal_order(tmp_path):
    spec = example_spec("mock", "gemm")
    path = tmp_path / "experiment.json"
    spec.save(path)
    loaded = ExperimentSpec.load(path)
    assert [c.id for c in loaded.config_space().iter_configs()] == [c.id for c in spec.config_space().iter_configs()]
    assert len(path.read_text().splitlines()) > 10  # Editable, human-readable JSON.


def test_local_plugin_runs_via_real_cpu_only_cli(tmp_path):
    spec = replace(example_spec("mock"), name="plugin-contract", backend="deterministic_mock")
    spec_path = tmp_path / "plugin.json"
    spec.save(spec_path)
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-S", "-m", "kernel_tuning_lab", "--plugin",
                             str(project / "examples/plugins/deterministic_mock.py"), "benchmark", "--spec", str(spec_path),
                             "--require-success", "--output", str(tmp_path / "output")],
                            cwd=project, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["backend"] == "deterministic_mock"
    assert payload["provenance"] == "mock" and payload["successful_shapes"] == 2


def test_hub_cli_import_keeps_unknown_and_deduplicates(tmp_path, monkeypatch, capsys):
    from kernel_tuning_lab.adapters import benchmark_hub
    rows = load_kernel_tuner_json(bundled_sample_path())
    entry = BenchmarkHubEntry("convolution_milo", "A100", "cachefiles/convolution_milo/A100.json.gz",
                              f"https://raw.githubusercontent.com/AutoTuningAssociation/benchmark_hub/{HUB_COMMIT}/cachefiles/convolution_milo/A100.json.gz",
                              1, "0" * 40)
    monkeypatch.setattr(benchmark_hub, "discover_benchmark_hub", lambda **kwargs: (entry,))
    monkeypatch.setattr(benchmark_hub, "import_benchmark_hub", lambda *args, **kwargs: rows)
    arguments = ["hub", "import", "--kernel", entry.kernel, "--gpu", entry.gpu,
                 "--cache-dir", str(tmp_path / "hub"), "--database", str(tmp_path / "data.sqlite")]
    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["added"] == 24 and first["verified_successes"] == 0
    assert main(arguments) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["added"] == 0 and second["duplicates"] == 24
    with ExperimentStore(tmp_path / "data.sqlite", read_only=True) as store:
        assert not any(record.successful for record in store.iter_records())


def test_scale_demo_saves_explicitly_synthetic_evidence(tmp_path):
    result = run_scale_demo(tmp_path / "scale", budget=2, candidate_pool_size=16, matrix_shapes=8, matrix_configs=6)
    assert result["space_cardinality"] == 139968
    assert result["online_trials"] == 24 and result["evaluations"] <= 48
    assert result["provenance"] == "mock" and result["l3_runs"] == 15
    rows = json.loads((tmp_path / "scale/online_results.json").read_text())
    assert all(row["regret"] is None and row["candidate_pool_approximate"] for row in rows)
    assert (tmp_path / "scale/report.md").exists()
