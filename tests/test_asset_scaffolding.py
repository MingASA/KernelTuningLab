"""Contributor scaffolds are executable and preserve correctness evidence."""

import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import pytest

from kernel_tuning_lab.claims import declare_correctness
from kernel_tuning_lab.kernel_runtime import AssetKernelAdapter
from kernel_tuning_lab.models import (HardwareSpec, KernelConfig, MeasurementRecord,
                                      OperatorSpec, RecordStatus, ShapeSpec)
from tests.test_asset_contracts import _kernel, _operator


ROOT = Path(__file__).resolve().parents[1]


def _cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, "-m", "kernel_tuning_lab", *map(str, args)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("operator", ["gemm@1.0.0", "batched_gemm@1.0.0", "softmax@1.0.0", "layernorm@1.0.0"])
def test_python_kernel_scaffold_generates_executable_run_and_refuses_overwrite(tmp_path, operator):
    folder = tmp_path / operator.split("@")[0]
    made = _cli("scaffold", "kernel", "--runtime", "python", "--operator", operator, "--output", folder)
    assert made.returncode == 0, made.stdout + made.stderr
    for name in ("assets.py", "experiment.json", "README.md"):
        assert (folder / name).is_file()
    checked = _cli("check", "--spec", folder / "experiment.json", "--output", folder / "check")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    ran = _cli("run", "--spec", folder / "experiment.json", "--output", folder / "run")
    assert ran.returncode == 0, ran.stdout + ran.stderr
    summary = json.loads((folder / "run" / "run_summary.json").read_text())
    assert summary["status"] == "passed"
    rerun = _cli("scaffold", "kernel", "--runtime", "python", "--operator", operator, "--output", folder)
    assert rerun.returncode != 0
    assert "refuses to overwrite" in rerun.stderr


def test_python_algorithm_scaffold_generates_executable_run(tmp_path):
    folder = tmp_path / "algorithm"
    made = _cli("scaffold", "algorithm", "--runtime", "python", "--output", folder)
    assert made.returncode == 0, made.stdout + made.stderr
    checked = _cli("check", "--spec", folder / "experiment.json", "--output", folder / "check")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    ran = _cli("run", "--spec", folder / "experiment.json", "--output", folder / "run")
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert json.loads((folder / "run" / "run_summary.json").read_text())["status"] == "passed"


def test_cuda_scaffold_defers_cleanly_without_optional_gpu_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    folder = tmp_path / "cuda-kernel"
    made = _cli("scaffold", "kernel", "--runtime", "cuda", "--output", folder)
    assert made.returncode == 0, made.stdout + made.stderr
    ran = _cli("run", "--spec", folder / "experiment.json", "--output", folder / "run")
    assert ran.returncode == 3, ran.stdout + ran.stderr
    summary = json.loads((folder / "run" / "run_summary.json").read_text())
    assert summary["status"] == "deferred"
    preflight = json.loads((folder / "run" / "preflight.json").read_text())
    assert "CUDA" in json.dumps(preflight)


def test_persisted_counterexample_revokes_only_claims_whose_scope_covers_it():
    operator = _operator()
    broad = declare_correctness(
        _kernel(operator), operator, id="broad", source="test", evidence=["declared"],
        scope={"runtimes": ["python"], "dtypes": ["float32"], "architectures": ["cpu"],
               "shapes": "*", "configs": "*", "parameters": "*"})
    narrow = declare_correctness(
        _kernel(operator), operator, id="narrow", source="test", evidence=["declared"],
        scope={"runtimes": ["python"], "dtypes": ["float32"], "architectures": ["cpu"],
               "shapes": {"M": 1, "N": 1}, "configs": "*", "parameters": "*"})
    kernel = _kernel(operator, claims=(broad, narrow))
    hardware = HardwareSpec("CPU", architecture="cpu", warp_size=1, max_threads_per_block=1)
    op_spec = OperatorSpec(name=operator.id, kind=operator.id, implementation=kernel.id, dtype="float32")
    failing_shape = ShapeSpec({"M": 2, "N": 1})
    record = MeasurementRecord(
        hardware, op_spec, failing_shape, KernelConfig({"tile": 1}), RecordStatus.CORRECTNESS_ERROR,
        provenance="measured", failure_reason="platform comparison found incorrect output",
        environment={"kernel_fingerprint": kernel.fingerprint, "operator_fingerprint": operator.fingerprint},
        metadata={"correctness": {"correctness_origin": "platform_reference"}},
    )

    first_adapter = AssetKernelAdapter(kernel, operator)
    assert first_adapter.remember_counterexample(record) == ["broad"]
    second_adapter = AssetKernelAdapter(kernel, operator)
    assert second_adapter.remember_counterexample(record) == ["broad"]
    prepared = second_adapter.prepare(hardware, op_spec, ShapeSpec({"M": 1, "N": 1}), 0)
    compiled = second_adapter.compile(prepared, KernelConfig({"tile": 1}))
    ok, evidence = second_adapter.check_correctness(compiled)
    assert ok
    assert evidence["correctness_origin"] == "submitter_claim"
    assert [item["id"] for item in evidence["claims_evaluated"] if item.get("accepted")] == ["narrow"]


def test_fixed_kernel_with_empty_space_runs_without_a_fake_knob(tmp_path):
    folder = tmp_path / "fixed-kernel"
    made = _cli("scaffold", "kernel", "--runtime", "python", "--output", folder)
    assert made.returncode == 0, made.stdout + made.stderr
    source = (folder / "assets.py").read_text()
    source = source.replace("from kernel_tuning_lab.sdk import get_catalog",
                            "from kernel_tuning_lab.sdk import SpaceDefinition, get_catalog")
    source = source.replace("build=build, claims=(),", "build=build, claims=(), space=SpaceDefinition({}),")
    (folder / "assets.py").write_text(source)
    checked = _cli("check", "--spec", folder / "experiment.json", "--output", folder / "check")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    ran = _cli("run", "--spec", folder / "experiment.json", "--output", folder / "run")
    assert ran.returncode == 0, ran.stdout + ran.stderr
    trials = json.loads((folder / "run" / "trials.json").read_text())
    assert trials and all(trial["best_config"] == {} for trial in trials)
    assert all(trial["space_cardinality"] == 1 for trial in trials)


def test_million_point_asset_space_random_grid_and_preflight_are_bounded(monkeypatch):
    from kernel_tuning_lab.acceptance import probe_legal_configs, request_candidates
    from kernel_tuning_lab.asset_space import build_effective_space
    from kernel_tuning_lab.catalog import get_catalog
    from kernel_tuning_lab.models import HardwareSpec, OperatorSpec, SearchBudget, ShapeSpec
    from kernel_tuning_lab.search import SearchHistory

    catalog = get_catalog()
    operator_definition = catalog.resolve("operator", "gemm@1.0.0")
    base = catalog.resolve("kernel", "python/gemm@1.0.0")
    kernel = replace(base, id="test/million-space", version="1.0.0",
                     space=__import__("kernel_tuning_lab.sdk", fromlist=["SpaceDefinition"]).SpaceDefinition(
                         {key: __import__("kernel_tuning_lab.sdk", fromlist=["ParameterSpec"]).ParameterSpec(tuple(range(100)))
                          for key in ("x", "y", "z")}))
    hardware = HardwareSpec("CPU", architecture="cpu", warp_size=1, max_threads_per_block=1)
    shape = ShapeSpec({"M": 4, "N": 4, "K": 4})
    operator = OperatorSpec("gemm", kind="gemm", implementation=kernel.ref, dtype="float32", version="1.0.0")
    space = build_effective_space(kernel, hardware, operator, shape)
    assert space.cardinality == 1_000_000
    def forbidden(*args, **kwargs):
        raise AssertionError("million-point search must not enumerate iter_configs")
    monkeypatch.setattr(space, "iter_configs", forbidden)
    probe = probe_legal_configs(space, hardware, operator, shape, seed=3, count=2)
    assert probe["inspected"] <= 512
    assert probe["selected"] == 2
    for algorithm_name in ("random", "grid"):
        definition = catalog.resolve("algorithm", f"{algorithm_name}@1.0.0")
        searcher = definition.factory(seed=7)
        candidates = request_candidates(searcher, hardware, operator, shape, space,
                                       SearchHistory(), SearchBudget(3), batch_size=2)
        assert len(candidates) == 2
        assert all(space.validate(candidate, hardware, operator, shape).valid for candidate in candidates)

