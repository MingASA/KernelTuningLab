"""Cited planning profiles; actual device probes remain authoritative for execution."""

from __future__ import annotations

import re
from typing import Any

from .models import HardwareSpec
from .space import ValidationResult


_H200_PRODUCT = "https://www.nvidia.com/en-us/data-center/h200/"
_H200_DEVTECH = "https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/CUDA-Programming-and-Optimization.pdf"
_H800_SUPPORT = "https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h800.html"
_ADA_WHITEPAPER = "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf"
_HOPPER_GUIDE = "https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html"
_ADA_GUIDE = "https://docs.nvidia.com/cuda/ada-tuning-guide/index.html"
_CUDA_CAPABILITIES = "https://developer.nvidia.com/cuda/gpus"
_CUDA_LIMITS = "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html"
_BLACKWELL_GUIDE = "https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html"
_RTX_BLACKWELL_WHITEPAPER = "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf"
_GEFORCE_COMPARE = "https://www.nvidia.com/en-us/geforce/graphics-cards/compare/"
_RTX5060_PRODUCT = "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/"


def list_hardware_profiles() -> tuple[str, ...]:
    """Canonical profile names; SXM and PCIe/NVL products are distinct SKUs."""
    return "h200-sxm", "h800-sxm", "rtx4090", "rtx5060ti-16gb"


def _canonical_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("hardware profile name must be a nonempty string")
    key = re.sub(r"[^a-z0-9]", "", name.casefold())
    for prefix in ("nvidia", "geforce"):
        if key.startswith(prefix):
            key = key[len(prefix):]
    aliases = {
        "h200": "h200-sxm", "h200sxm": "h200-sxm", "h200sxm5": "h200-sxm",
        "h200sxm141gb": "h200-sxm", "h200sxm5141gb": "h200-sxm",
        "h800": "h800-sxm", "h800sxm": "h800-sxm", "h800sxm5": "h800-sxm",
        "h800sxm80gb": "h800-sxm", "h800sxm580gb": "h800-sxm",
        "4090": "rtx4090", "rtx4090": "rtx4090",
        "5060ti": "rtx5060ti-16gb", "rtx5060ti": "rtx5060ti-16gb",
        "5060ti16g": "rtx5060ti-16gb", "5060ti16gb": "rtx5060ti-16gb",
        "rtx5060ti16g": "rtx5060ti-16gb", "rtx5060ti16gb": "rtx5060ti-16gb",
        "rtx5060ti16gbgddr7": "rtx5060ti-16gb",
    }
    try:
        return aliases[key]
    except KeyError:
        raise ValueError(f"unknown hardware profile {name!r}; choose from {', '.join(list_hardware_profiles())}") from None


def _metadata(key: str, *, variant: str, memory_gb: int, sources: list[str],
              unknown: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "profile_name": key, "variant": variant,
        "specification_only": True, "source_kind": "vendor_documentation",
        "source": "NVIDIA specifications; not a physical device probe",
        "sources": sources, "verified_date": "2026-10-22",
        "vendor_nominal_memory_gb": memory_gb,
        "unknown_fields": unknown, "probe_required_before_measurement": True,
        "compute_precision": "FP16 Tensor Core dense",
        "compute_kind": "theoretical_peak_not_measured", "structured_sparsity": False,
        "units": {
            "compute_tflops": "10^12 floating point operations/s; FMA counts as two",
            "bandwidth_gbps": "10^9 bytes/s (GB/s), not gigabits/s",
            "memory_bytes": "nominal vendor GB converted with 10^9 bytes/GB; not CUDA usable bytes",
            "smem_per_block": "bytes; architectural opt-in maximum, not default static allocation",
            "l2_bytes": "bytes; architectural cache KB/MB normalized as KiB/MiB",
            "registers_per_sm": "count of 32-bit registers",
        },
        "runtime_note": "Use the physical probe for hardware identity, capacities, and experiment records.",
    }
    metadata.update(extra or {})
    return metadata


def get_hardware_profile(name: str) -> HardwareSpec:
    """Return a fresh, cited profile without importing CUDA, torch, or Triton.

    Bare H200/H800 names select their explicitly documented SXM profiles. PCIe,
    NVL, RTX 4090 D, and laptop variants are not aliases of these profiles.
    Unknown H800 throughput/bandwidth/SM/cache values intentionally remain unset.
    Bare 5060ti aliases explicitly select the 16GB desktop profile; physical
    validation checks memory capacity so an 8GB board cannot pass as 16GB.
    """
    key = _canonical_name(name)
    if key == "h200-sxm":
        return HardwareSpec(
            "NVIDIA H200 SXM", "sm_90", sm_count=132,
            smem_per_block=227 * 1024, registers_per_sm=64 * 1024,
            compute_tflops=1979.0 / 2, bandwidth_gbps=4800.0,
            l2_bytes=50 * 1024**2, memory_bytes=141 * 10**9,
            features=("tensor_cores", "fp16", "bfloat16", "fp8", "tma", "thread_block_clusters"),
            metadata=_metadata(key, variant="SXM", memory_gb=141,
                               sources=[_H200_PRODUCT, _H200_DEVTECH, _HOPPER_GUIDE, _CUDA_CAPABILITIES, _CUDA_LIMITS],
                               unknown=[], extra={
                "compute_derivation": "1979 vendor FP16 sparse TFLOPS / 2 = 989.5 dense TFLOPS; rounded source values",
                "accumulator_precision": "vendor product table does not separately specify accumulation mode",
                "field_sources": {"memory_bytes": _H200_PRODUCT, "bandwidth_gbps": _H200_PRODUCT,
                                  "compute_tflops": _H200_PRODUCT, "sm_count": _H200_DEVTECH,
                                  "l2_bytes": _H200_DEVTECH, "smem_per_block": _HOPPER_GUIDE,
                                  "registers_per_sm": _HOPPER_GUIDE, "architecture": _CUDA_CAPABILITIES},
            }),
        )
    if key == "h800-sxm":
        return HardwareSpec(
            "NVIDIA H800 SXM5 80GB", "sm_90", sm_count=0,
            smem_per_block=227 * 1024, registers_per_sm=64 * 1024,
            compute_tflops=None, bandwidth_gbps=None, l2_bytes=None,
            memory_bytes=80 * 10**9,
            features=("tensor_cores", "fp16", "bfloat16", "fp8", "tma", "thread_block_clusters"),
            metadata=_metadata(key, variant="SXM5", memory_gb=80,
                               sources=[_H800_SUPPORT, _HOPPER_GUIDE, _CUDA_LIMITS],
                               unknown=["sm_count", "compute_tflops", "bandwidth_gbps", "l2_bytes"], extra={
                "architecture_basis": "inferred sm_90 from NVIDIA's explicit Hopper H800 classification; verify using the device probe",
                "resource_limits_basis": "Hopper architectural bounds; verify opt-in device limits before execution",
                "missing_specification_reason": "No primary source confirming these SXM-specific values was retrieved; no H100/PCIe values substituted.",
                "field_sources": {"memory_bytes": _H800_SUPPORT, "variant": _H800_SUPPORT,
                                  "architecture": [_H800_SUPPORT, _HOPPER_GUIDE],
                                  "smem_per_block": _HOPPER_GUIDE, "registers_per_sm": _HOPPER_GUIDE},
            }),
        )
    if key == "rtx5060ti-16gb":
        return HardwareSpec(
            "NVIDIA GeForce RTX 5060 Ti 16GB", "sm_120", sm_count=36,
            smem_per_block=99 * 1024, registers_per_sm=64 * 1024,
            compute_tflops=None, bandwidth_gbps=448.0, l2_bytes=None,
            memory_bytes=16 * 10**9,
            features=("tensor_cores", "fp16", "bfloat16", "fp8"),
            metadata=_metadata(key, variant="desktop RTX 5060 Ti 16GB", memory_gb=16,
                               sources=[_RTX5060_PRODUCT, _GEFORCE_COMPARE, _CUDA_CAPABILITIES,
                                        _BLACKWELL_GUIDE, _RTX_BLACKWELL_WHITEPAPER],
                               unknown=["compute_tflops", "l2_bytes"], extra={
                "role": "single-card correctness validation before Hopper deployment",
                "memory_type": "GDDR7", "memory_bus_bits": 128,
                "cuda_cores": 4608,
                "sm_count_basis": "derived: 4608 vendor CUDA cores / 128 cores per RTX Blackwell SM = 36; physical probe remains authoritative",
                "compute_precision_note": "AI TOPS headline is not dense FP16 with FP32 accumulation; no conversion is assumed",
                "accumulator_precision": "unknown theoretical FP16/FP32-accumulation peak",
                "runtime_baseline": {"torch": ">=2.7", "torch_cuda_runtime": ">=12.8", "triton": ">=3.3"},
                "field_sources": {"memory_bytes": _RTX5060_PRODUCT, "architecture": _CUDA_CAPABILITIES,
                                  "bandwidth_gbps": _GEFORCE_COMPARE, "sm_count": [_GEFORCE_COMPARE, _RTX_BLACKWELL_WHITEPAPER],
                                  "smem_per_block": _BLACKWELL_GUIDE, "registers_per_sm": _BLACKWELL_GUIDE},
            }),
        )
    return HardwareSpec(
        "NVIDIA GeForce RTX 4090", "sm_89", sm_count=128,
        smem_per_block=99 * 1024, registers_per_sm=64 * 1024,
        compute_tflops=165.2, bandwidth_gbps=1008.0,
        l2_bytes=73728 * 1024, memory_bytes=24 * 10**9,
        features=("tensor_cores", "fp16", "bfloat16", "fp8"),
        metadata=_metadata(key, variant="desktop RTX 4090", memory_gb=24,
                           sources=[_ADA_WHITEPAPER, _ADA_GUIDE, _CUDA_CAPABILITIES, _CUDA_LIMITS],
                           unknown=[], extra={
            "accumulator_precision": "float32",
            "fp16_accumulate_fp16_dense_tflops": 330.3,
            "compute_derivation": "Ada whitepaper v2.02 Table 2: FP16 Tensor with FP32 accumulate, first (non-sparse) value 165.2",
            "field_sources": {"sm_count": _ADA_WHITEPAPER, "memory_bytes": _ADA_WHITEPAPER,
                              "compute_tflops": _ADA_WHITEPAPER, "bandwidth_gbps": _ADA_WHITEPAPER,
                              "l2_bytes": _ADA_WHITEPAPER, "smem_per_block": _ADA_GUIDE,
                              "registers_per_sm": _ADA_GUIDE, "architecture": _CUDA_CAPABILITIES},
        }),
    )


def _actual_family(name: str) -> str | None:
    compact = re.sub(r"[^a-z0-9]", "", name.casefold())
    if "h200" in compact:
        return "h200-sxm"
    if "h800" in compact:
        return "h800-sxm"
    if "4090" in compact and not any(token in compact for token in ("4090d", "laptop", "mobile")):
        return "rtx4090"
    if "5060ti" in compact and not any(token in compact for token in ("laptop", "mobile")):
        return "rtx5060ti-16gb"
    return None


def _architecture_key(architecture: str) -> str:
    return re.sub(r"[^a-z0-9]", "", architecture.casefold())


def validate_hardware_profile(profile: HardwareSpec, actual: HardwareSpec) -> ValidationResult:
    """Check profile compatibility with a separately obtained physical probe.

    Missing probe capacities do not become confirmed values. An explicit PCIe/NVL
    or MIG identity is rejected for SXM profiles. A generic H200/H800 device name
    cannot attest board form factor; this check establishes family compatibility,
    while experiment records must retain the actual probe identity.
    """
    reasons = []
    if actual.metadata.get("specification_only") is True:
        reasons.append("actual hardware is another specification profile, not a physical device probe")
    key = profile.metadata.get("profile_name") or _actual_family(profile.name)
    if key not in list_hardware_profiles():
        reasons.append("requested profile is not a supported NVIDIA profile")
    if key != _actual_family(actual.name):
        reasons.append(f"requested model {profile.name!r} does not match probed model {actual.name!r}")
    if _architecture_key(profile.architecture) != _architecture_key(actual.architecture):
        reasons.append(f"architecture mismatch: requested {profile.architecture}, probed {actual.architecture}")
    actual_identity = re.sub(r"[^a-z0-9]", "", actual.name.casefold())
    actual_variant = str(actual.metadata.get("form_factor", actual.metadata.get("variant", ""))).casefold()
    if "mig" in actual_identity or actual.metadata.get("mig_enabled") is True:
        reasons.append("a full-GPU profile cannot be used as a MIG instance specification")
    if key in {"h200-sxm", "h800-sxm"} and (
        any(word in actual_identity for word in ("pcie", "nvl"))
        or any(word in actual_variant for word in ("pcie", "pci express", "nvl"))
    ):
        reasons.append("SXM profile is incompatible with the probed PCIe/NVL variant")
    for attribute in ("sm_count", "registers_per_sm"):
        expected, observed = getattr(profile, attribute), getattr(actual, attribute)
        if expected and observed and expected != observed:
            reasons.append(f"{attribute} mismatch: profile {expected}, probe {observed}")
    for attribute in ("smem_per_block", "max_threads_per_block"):
        expected, observed = getattr(profile, attribute), getattr(actual, attribute)
        if expected and observed and observed < expected:
            reasons.append(f"probed {attribute} {observed} is below profile requirement {expected}")
    if profile.memory_bytes and actual.memory_bytes:
        # Vendor GB labels are nominal, and CUDA reports usable bytes after
        # reservations/ECC. Allow both SI/GiB label conventions without pretending
        # that a nominal capacity is an exact physical byte measurement.
        ratio = actual.memory_bytes / profile.memory_bytes
        if not 0.90 <= ratio <= 1.15:
            reasons.append("probed memory capacity is inconsistent with the profile's nominal capacity")
    if key == "rtx5060ti-16gb" and not actual.memory_bytes:
        reasons.append("RTX 5060 Ti 16GB validation requires probed memory capacity to distinguish the 8GB model")
    return ValidationResult(not reasons, tuple(reasons))
