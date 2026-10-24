"""CPU contract fixtures for the local Blackwell machine; no physical GPU claims."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kernel_tuning_lab.backends import BackendUnavailableError, MockAdapter
from kernel_tuning_lab.backends.compatibility import validate_triton_runtime
from kernel_tuning_lab.backends.triton_gemm import TritonGemmAdapter
from kernel_tuning_lab.cli import main
from kernel_tuning_lab.experiments import example_spec
from kernel_tuning_lab.gpu_validation import SMOKE_CASES, summarize_gpu_smoke
from kernel_tuning_lab.hardware import get_hardware_profile, validate_hardware_profile
from kernel_tuning_lab.models import KernelConfig, MeasurementRecord, RecordStatus
from kernel_tuning_lab.operators import OPERATOR_NAMES, operator_spec
from kernel_tuning_lab.runner import BenchmarkRunner
from kernel_tuning_lab.space import gemm_space


@pytest.mark.parametrize("alias", ("5060ti", "5060ti16g", "5060 Ti 16GB", "RTX 5060 Ti",
                                  "NVIDIA GeForce RTX 5060 Ti 16GB", "rtx5060ti-16gb"))
def test_5060ti_profile_is_consumer_blackwell_16gb(alias):
    profile = get_hardware_profile(alias)
    assert profile.architecture == "sm_120" and profile.sm_count == 36
    assert profile.memory_bytes == 16 * 10**9 and profile.bandwidth_gbps == 448
    assert profile.smem_per_block == 99 * 1024
    assert profile.compute_tflops is None  # Do not relabel sparse AI TOPS.
    assert profile.metadata["specification_only"] is True


def physical_fixture():
    return replace(get_hardware_profile("5060ti"), name="NVIDIA GeForce RTX 5060 Ti",
                   memory_bytes=16 * 1024**3, metadata={"source": "CPU unit-test probe fixture"})


def test_5060ti_rejects_8gb_non_ti_and_unknown_capacity():
    profile = get_hardware_profile("5060ti16g")
    actual = physical_fixture()
    assert validate_hardware_profile(profile, actual).valid
    for wrong in (replace(actual, memory_bytes=8 * 1024**3), replace(actual, memory_bytes=None),
                  replace(actual, name="NVIDIA GeForce RTX 5060"), replace(actual, architecture="sm_90"), profile):
        assert not validate_hardware_profile(profile, wrong).valid
    with pytest.raises(ValueError):
        get_hardware_profile("RTX 5060 Ti 8GB")


def test_5060ti_uses_its_own_smem_limit_instead_of_hopper_limits():
    spec = example_spec("triton")
    config = KernelConfig({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 4})
    space = gemm_space()
    assert space.validate(config, get_hardware_profile("H200"), spec.operator, spec.shapes[0]).valid
    assert not space.validate(config, get_hardware_profile("5060ti"), spec.operator, spec.shapes[0]).valid


@pytest.mark.parametrize("operator", OPERATOR_NAMES)
def test_all_four_mock_smokes_cover_both_shapes_on_5060ti(operator):
    spec = example_spec("mock", operator)
    runner = BenchmarkRunner(MockAdapter(0, 0, 0))
    rows = runner.run(get_hardware_profile("5060ti16g"), spec.operator, spec.shapes, spec.config_space(), spec.budget)
    assert len(rows) == 8 and all(row.successful for row in rows)
    assert {row.shape.id for row in rows} == {shape.id for shape in spec.shapes}


@pytest.mark.parametrize("torch,cuda,triton,valid", (("2.7.1+cu128", "12.8", "3.3.0", True),
    ("2.12.0+cu130", "13.0", "3.6.0", True), ("2.6.0", "12.8", "3.3.0", False),
    ("2.7.1", "12.6", "3.3.0", False), ("2.7.1", "12.8", "3.2.0", False),
    ("2.7.1", None, "3.3.0", False)))
def test_native_blackwell_runtime_baseline(torch, cuda, triton, valid):
    result = validate_triton_runtime("sm_120", torch_version=torch, cuda_version=cuda, triton_version=triton)
    assert result.valid is valid
    assert validate_triton_runtime("sm_90", torch_version=torch, cuda_version=cuda, triton_version=triton).valid


def test_backend_rejects_old_blackwell_runtime_before_setting_device(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_capability=lambda device: (12, 0))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.6.0", cuda=cuda,
                                                             version=SimpleNamespace(hip=None, cuda="12.6")))
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="3.2.0"))
    with pytest.raises(BackendUnavailableError, match="sm120 requires"):
        TritonGemmAdapter()


def stage_fixtures(root):
    """Artificial measured-record fixtures test the gate, never real experiment data."""
    hardware = physical_fixture()
    environment = {"kernel_source_sha256": "f" * 64, "adapter_source_sha256": "f" * 64,
                   "python": "CPU test fixture", "torch": "2.7.1", "triton": "3.3.0", "cuda_runtime": "12.8"}
    (root / "preflight.json").write_text(json.dumps({"hardware": asdict(hardware), "requested_profile": "rtx5060ti-16gb",
                                                    "environment": environment, "runtime_requirements_met": True}))
    for operator, dtype in SMOKE_CASES:
        spec = example_spec("triton", operator, dtype)
        directory = root / f"{operator}-{dtype}"
        directory.mkdir()
        record = MeasurementRecord(hardware, operator_spec(operator, "triton", dtype), spec.shapes[0],
                                    next(spec.config_space().iter_configs()), RecordStatus.SUCCESS,
                                    runtime_samples_ms=(0.01,), provenance="measured",
                                    environment={**environment, "base_adapter_source_sha256": "f" * 64},
                                    metadata={"correctness": {"verified_on_gpu": True}, "test_fixture_only": True})
        (directory / "run_records.jsonl").write_text(json.dumps(record.to_dict()) + "\n")
        (directory / "run_summary.json").write_text(json.dumps({"provenance": "measured", "all_successful": True,
            "requested_shapes": 1, "requested_shape_ids": [spec.shapes[0].id], "successful_shapes": 1,
            "returned_records": 1, "stats": {"cache_hits": 0}}))


def test_smoke_gate_requires_all_ten_physical_stages(tmp_path):
    stage_fixtures(tmp_path)
    result = summarize_gpu_smoke(tmp_path)
    assert result["physical_gpu_validated"] and result["total_records"] == 10
    assert result["target_hopper_validation_required"]
    path = tmp_path / "layernorm-float32/run_records.jsonl"
    row = json.loads(path.read_text())
    row["provenance"] = "mock"
    path.write_text(json.dumps(row) + "\n")
    failed = summarize_gpu_smoke(tmp_path)
    assert not failed["physical_gpu_validated"] and failed["status"] == "failed"


@pytest.mark.parametrize("operator", ("batched_gemm", "layernorm"))
def test_smoke_gate_detects_shared_kernel_changes_between_operators(tmp_path, operator):
    stage_fixtures(tmp_path)
    for kind, dtype in SMOKE_CASES:
        if kind == operator:
            path = tmp_path / f"{kind}-{dtype}/run_records.jsonl"
            row = json.loads(path.read_text())
            row["environment"]["kernel_source_sha256"] = "e" * 64
            path.write_text(json.dumps(row) + "\n")
    result = summarize_gpu_smoke(tmp_path)
    assert not result["physical_gpu_validated"]
    assert any("source changed" in error for error in result["errors"])


@pytest.mark.parametrize("mismatch", ("runtime", "memory", "shape", "adapter"))
def test_smoke_gate_rejects_mixed_or_incomplete_stage_artifacts(tmp_path, mismatch):
    stage_fixtures(tmp_path)
    directory = tmp_path / "gemm-float16"
    if mismatch == "shape":
        path = directory / "run_summary.json"
        row = json.loads(path.read_text())
        row["requested_shape_ids"] = ["different shape"]
    else:
        path = directory / "run_records.jsonl"
        row = json.loads(path.read_text())
        if mismatch == "runtime":
            row["environment"]["cuda_runtime"] = "13.0"
        elif mismatch == "adapter":
            row["environment"]["adapter_source_sha256"] = "e" * 64
        else:
            row["hardware"]["memory_bytes"] = 8 * 1024**3
    path.write_text(json.dumps(row) + "\n")
    assert not summarize_gpu_smoke(tmp_path)["physical_gpu_validated"]


def test_smoke_gate_writes_failed_summary_for_corrupt_preflight(tmp_path):
    stage_fixtures(tmp_path)
    (tmp_path / "preflight.json").write_text("[]\n")
    result = summarize_gpu_smoke(tmp_path)
    assert result["status"] == "failed" and result["errors"]
    assert json.loads((tmp_path / "validation_summary.json").read_text())["status"] == "failed"


def test_smoke_retry_clears_old_completion_markers_before_failed_preflight(tmp_path):
    stage_fixtures(tmp_path)
    (tmp_path / "validation_summary.json").write_text('{"status": "passed"}\n')
    journal = tmp_path / "gemm-float16/measurements.sqlite"
    journal.write_bytes(b"existing measurement cache")
    interpreter = tmp_path / "unavailable-python.sh"
    interpreter.write_text("#!/bin/sh\nexit 2\n")
    interpreter.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts/run_gpu_smoke.sh"
    result = subprocess.run(["bash", str(script), "5060ti16g", str(tmp_path)],
                            env={**os.environ, "KT_PYTHON_BIN": str(interpreter)}, capture_output=True, check=False)
    assert result.returncode == 2
    assert not (tmp_path / "validation_summary.json").exists()
    for operator, dtype in SMOKE_CASES:
        assert not (tmp_path / f"{operator}-{dtype}/run_summary.json").exists()
        assert not (tmp_path / f"{operator}-{dtype}/run_records.jsonl").exists()
    assert journal.read_bytes() == b"existing measurement cache"


def test_strict_cli_fails_even_when_another_config_succeeds(tmp_path, monkeypatch, capsys):
    class OneBadConfig(MockAdapter):
        def check_correctness(self, compiled):
            return compiled.config.get("num_warps") == 4, {"correctness_mode": "synthetic", "verified_on_gpu": False}
    monkeypatch.setattr("kernel_tuning_lab.cli.get_adapter", lambda name: OneBadConfig(0, 0, 0))
    result = main(["benchmark", "--backend", "mock", "--hardware", "5060ti16g", "--require-all-success",
                   "--output", str(tmp_path / "run")])
    assert result == 2
    summary = json.loads((tmp_path / "run/run_summary.json").read_text())
    assert summary["successful_shapes"] == 2 and not summary["all_successful"]
    assert "strict GPU validation failed" in capsys.readouterr().err
