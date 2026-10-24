"""Planning profiles do not require GPU packages and cannot certify measurements."""

from dataclasses import replace

import pytest

from kernel_tuning_lab.hardware import get_hardware_profile, list_hardware_profiles, validate_hardware_profile
from kernel_tuning_lab.models import HardwareSpec, OperatorSpec, ShapeSpec
from kernel_tuning_lab.space import gemm_space


def test_profiles_have_explicit_variants_units_and_primary_sources():
    assert list_hardware_profiles() == ("h200-sxm", "h800-sxm", "rtx4090", "rtx5060ti-16gb")
    for name in list_hardware_profiles():
        profile = get_hardware_profile(name)
        assert profile.metadata["specification_only"] is True
        assert profile.metadata["probe_required_before_measurement"] is True
        assert profile.metadata["structured_sparsity"] is False
        assert profile.metadata["sources"]
        assert profile.metadata["units"]["bandwidth_gbps"].startswith("10^9 bytes/s")
        assert any("nvidia.com" in url for url in profile.metadata["sources"])
    with pytest.raises(ValueError):
        get_hardware_profile("H800 PCIe")
    with pytest.raises(ValueError):
        get_hardware_profile("RTX 4090 D")


def test_h200_and_4090_keep_dense_fp16_precision_conventions():
    hopper = get_hardware_profile("H200")
    ada = get_hardware_profile("NVIDIA GeForce RTX 4090")
    assert hopper.architecture == "sm_90" and hopper.sm_count == 132
    assert hopper.compute_tflops == 1979.0 / 2
    assert hopper.bandwidth_gbps == 4800.0
    assert ada.architecture == "sm_89" and ada.sm_count == 128
    assert ada.compute_tflops == 165.2
    assert ada.metadata["accumulator_precision"] == "float32"
    assert ada.metadata["fp16_accumulate_fp16_dense_tflops"] == 330.3
    assert ada.smem_per_block == 99 * 1024


def test_h800_unknown_sxm_metrics_are_not_copied_from_other_boards():
    profile = get_hardware_profile("H800 SXM5")
    assert profile.architecture == "sm_90"
    assert profile.sm_count == 0
    assert profile.compute_tflops is None
    assert profile.bandwidth_gbps is None
    assert profile.l2_bytes is None
    assert profile.metadata["vendor_nominal_memory_gb"] == 80
    assert "inferred" in profile.metadata["architecture_basis"]
    actual = HardwareSpec("NVIDIA H800 SXM5 80GB", "sm_90", sm_count=132,
                          smem_per_block=227 * 1024, registers_per_sm=65536,
                          memory_bytes=80 * 1024**3, metadata={"source": "test probe fixture"})
    assert validate_hardware_profile(profile, actual).valid


def test_profile_validation_rejects_wrong_model_architecture_variant_or_fake_probe():
    profile = get_hardware_profile("h200-sxm")
    actual = replace(profile, name="NVIDIA H200", memory_bytes=141 * 1024**3,
                     metadata={"source": "test probe fixture"})
    assert validate_hardware_profile(profile, actual).valid
    for wrong in (replace(actual, architecture="sm_89"),
                  replace(actual, name="NVIDIA H800 SXM5 80GB"),
                  replace(actual, name="NVIDIA H200 NVL"),
                  replace(actual, metadata={"form_factor": "PCIe"}),
                  replace(actual, smem_per_block=48 * 1024),
                  replace(actual, memory_bytes=24 * 1024**3), profile):
        result = validate_hardware_profile(profile, wrong)
        assert not result.valid and result.reasons


def test_profiles_allow_portable_gemm_legality_without_exposing_hopper_only_knobs():
    shape = ShapeSpec({"M": 127, "N": 129, "K": 65})
    operator = OperatorSpec("gemm", implementation="triton_gemm")
    space = gemm_space(small=True)
    for name in list_hardware_profiles():
        profile = get_hardware_profile(name)
        assert space.legal_configs(profile, operator, shape)
    assert set(space.domains) == {"BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"}
    profile = get_hardware_profile("H200")
    profile.metadata["sources"].clear()
    assert get_hardware_profile("H200").metadata["sources"]
