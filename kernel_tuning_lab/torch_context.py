"""CPU-safe Torch reference isolation and explicit execution-state identity."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path


@contextmanager
def strict_fp32_reference(torch):
    """Disable CUDA autocast/TF32 locally and restore the caller's exact mode.

    New Torch precision settings must not be mixed with allow_tf32 setters.
    Older supported Torch releases retain the full highest/high/medium mode.
    No Torch import occurs at asset discovery or on CPU-only installations.
    """
    matmul = torch.backends.cuda.matmul
    modern = hasattr(matmul, "fp32_precision")
    previous = matmul.fp32_precision if modern else torch.get_float32_matmul_precision()
    try:
        if modern:
            matmul.fp32_precision = "ieee"
        else:
            torch.set_float32_matmul_precision("highest")
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        if modern:
            matmul.fp32_precision = previous
        else:
            torch.set_float32_matmul_precision(previous)


def _call(obj, name):
    method = getattr(obj, name, None)
    return method() if callable(method) else None


def torch_execution_context(torch):
    """Record known math/dispatch/compiler controls; never infer an unknown backend.

    Compiler cache directory is deliberately excluded: its location is not a
    math protocol. Compilation cache warmth and acquisition wall costs remain
    separate from latency/cache compatibility. Unknown external plugin controls
    still require explicit asset metadata/source_files.
    """
    matmul = torch.backends.cuda.matmul
    precision = {name: getattr(matmul, name, None) for name in (
        "fp32_precision", "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction", "allow_fp16_accumulation")}
    if not hasattr(matmul, "fp32_precision"):
        precision["allow_tf32"] = matmul.allow_tf32
    try:
        precision["legacy_float32_mode"] = _call(torch, "get_float32_matmul_precision")
    except RuntimeError:
        # New per-backend settings can make the legacy aggregate getter undefined.
        precision["legacy_float32_mode"] = "undefined_for_per_backend_settings"
    try:
        autocast_enabled = torch.is_autocast_enabled("cuda")
    except TypeError:
        autocast_enabled = torch.is_autocast_enabled()
    dtype_getter = getattr(torch, "get_autocast_dtype", None)
    autocast_dtype = dtype_getter("cuda") if dtype_getter else _call(torch, "get_autocast_gpu_dtype")
    overrides = ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                 "TRITON_INTERPRET", "TRITON_F32_DEFAULT", "TRITON_ENABLE_FP_FUSION", "TRITON_ALWAYS_COMPILE",
                 "TRITON_KERNEL_OVERRIDE", "TRITON_OVERRIDE_DIR", "TRITON_PTXAS_PATH",
                 "TRITON_PTXAS_BLACKWELL_PATH", "TRITON_CUDA_LIBDEVICE_PATH")
    return {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "matmul": precision, "global_fp32_precision": getattr(torch.backends, "fp32_precision", None),
            "cuda_autocast": {"enabled": bool(autocast_enabled), "dtype": str(autocast_dtype)},
            "deterministic_algorithms": _call(torch, "are_deterministic_algorithms_enabled"),
            "default_dtype": str(_call(torch, "get_default_dtype")),
            "preferred_blas_library": str(_call(torch.backends.cuda, "preferred_blas_library")),
            "sdpa": {name: _call(torch.backends.cuda, name) for name in (
                "flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled")},
            "compiler_overrides": {name: os.environ[name] for name in overrides if name in os.environ}}
