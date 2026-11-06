"""Optional NVIDIA Triton GEMM backend with separate preparation/JIT/device timing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable

from ..torch_context import strict_fp32_reference, torch_execution_context
from .base import BackendUnavailableError
from .compatibility import validate_triton_runtime
from ..models import HardwareSpec, KernelConfig, OperatorSpec, ShapeSpec
from ..space import ValidationResult


@dataclass
class PreparedGemm:
    a: Any
    b: Any
    c: Any
    reference: Any
    shape: ShapeSpec


@dataclass
class CompiledGemm:
    prepared: PreparedGemm
    launch: Callable[[], Any]


class TritonGemmAdapter:
    name = "triton"
    provenance = "measured"
    measurement_mode = "cuda_events_hot_single_kernel"

    def __init__(self, device: int = 0, atol: float = 0.02, rtol: float = 0.02, batched: bool = False):
        try:
            import torch
            import triton
        except ImportError as exc:
            raise BackendUnavailableError("Triton GPU backend requires optional dependencies: pip install '.[gpu]'") from exc
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise BackendUnavailableError("v0.1 Triton GEMM backend requires an available NVIDIA CUDA GPU")
        if not 0 <= device < torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device} is unavailable")
        major, minor = torch.cuda.get_device_capability(device)
        compatibility = validate_triton_runtime(f"sm_{major}{minor}", torch_version=torch.__version__,
                                                cuda_version=torch.version.cuda, triton_version=triton.__version__)
        if not compatibility.valid:
            raise BackendUnavailableError("; ".join(compatibility.reasons) + "; install a Blackwell-capable CUDA PyTorch build and matching Triton; see docs/RTX5060TI.md")
        if atol < 0 or rtol < 0:
            raise ValueError("correctness tolerances must be nonnegative")
        self.torch = torch
        self.triton = triton
        self.device = device
        self.atol, self.rtol = atol, rtol
        self.batched = batched
        if batched:
            self.name = "triton_batched_gemm"
        torch.cuda.set_device(device)
        self._environment: dict[str, Any] | None = None

    def probe_hardware(self) -> HardwareSpec:
        props = self.torch.cuda.get_device_properties(self.device)
        major, minor = self.torch.cuda.get_device_capability(self.device)
        return HardwareSpec(name=props.name, architecture=f"sm_{major}{minor}",
                            sm_count=props.multi_processor_count,
                            smem_per_block=getattr(props, "shared_memory_per_block_optin", getattr(props, "shared_memory_per_block", 0)),
                            registers_per_sm=getattr(props, "regs_per_multiprocessor", 0),
                            memory_bytes=props.total_memory,
                            l2_bytes=getattr(props, "L2_cache_size", None) or None,
                            warp_size=getattr(props, "warp_size", 32),
                            max_threads_per_block=getattr(props, "max_threads_per_block", 1024),
                            features=("tensor_cores", "cuda_events"),
                            metadata={"source": "torch.cuda.get_device_properties", "device_index": self.device,
                                      "device_uuid": str(props.uuid) if getattr(props, "uuid", None) is not None else None,
                                      "unknown_fields": ["compute_tflops", "bandwidth_gbps"]})

    def environment(self) -> dict[str, Any]:
        if self._environment is None:
            driver = None
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                        capture_output=True, text=True, timeout=5, check=False)
                if result.returncode == 0:
                    driver = result.stdout.strip().splitlines()[self.device]
            except (OSError, subprocess.TimeoutExpired, IndexError):
                pass
            self._environment = {"python": platform.python_version(), "platform": platform.platform(),
                                 "torch": self.torch.__version__, "triton": self.triton.__version__,
                                 "cuda_runtime": self.torch.version.cuda, "nvidia_driver": driver,
                                 "torch_compiled_arches": list(self.torch.cuda.get_arch_list()),
                                 "gpu": self.torch.cuda.get_device_name(self.device), "device_index": self.device,
                                 "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "kernel_source_sha256": hashlib.sha256(Path(__file__).with_name("_triton_kernel.py").read_bytes()).hexdigest()}
        return {**self._environment, "torch_execution_context": torch_execution_context(self.torch)}

    def protocol_metadata(self) -> dict[str, Any]:
        return {"atol": self.atol, "rtol": self.rtol, "reference": "torch.float32_matmul_cast_to_output",
                "compile_timing": "JIT warmup compilation only; includes lookup when Triton disk cache is warm",
                "runtime_timing": "CUDA events around one launch; hot inputs, host launch gaps may be included",
                "cache_state": "uncontrolled_hot", "device": self.device}

    def validate(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec, config: KernelConfig) -> ValidationResult:
        reasons = []
        actual = self.probe_hardware()
        if (hardware.name, hardware.architecture) != (actual.name, actual.architecture):
            reasons.append("requested hardware does not match the physical CUDA device")
        expected = "batched_gemm" if self.batched else "gemm"
        if operator.kind != expected or operator.implementation != f"triton_{expected}":
            reasons.append(f"TritonGemmAdapter requires kind={expected} and implementation=triton_{expected}")
        if operator.dtype not in {"float16", "bfloat16"}:
            reasons.append("only float16 and bfloat16 are implemented")
        if not {"M", "N", "K"} <= shape.dims.keys():
            reasons.append("GEMM requires M/N/K")
        if self.batched and "Batch" not in shape.dims:
            reasons.append("batched GEMM requires Batch")
        if (not self.batched and (shape.get("Batch") != 1 or shape.get("batch") != 1)) or shape.dims.keys() - {"M", "N", "K", "Batch"}:
            reasons.append("GEMM supports contiguous matrices and an optional explicit Batch dimension")
        allowed = {"BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"}
        if config.params.keys() != allowed:
            reasons.append("Triton GEMM supports BLOCK_M/N/K, num_warps, num_stages only")
        return ValidationResult(not reasons, tuple(reasons))

    def prepare(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec, seed: int) -> PreparedGemm:
        self.torch.cuda.set_device(self.device)
        torch = self.torch
        dtype = getattr(torch, operator.dtype)
        generator = torch.Generator(device=f"cuda:{self.device}")
        generator.manual_seed(seed)
        m, n, k = (shape.get(d) for d in ("M", "N", "K"))
        batch_prefix = (shape.get("Batch"),) if self.batched else ()
        a = torch.randn((*batch_prefix, m, k), device=f"cuda:{self.device}", dtype=dtype, generator=generator) * 0.25
        b = torch.randn((*batch_prefix, k, n), device=f"cuda:{self.device}", dtype=dtype, generator=generator) * 0.25
        c = torch.empty((*batch_prefix, m, n), device=f"cuda:{self.device}", dtype=dtype)
        # Disable TF32 explicitly for the reference and restore caller state.
        with strict_fp32_reference(torch):
            reference = (a.float() @ b.float()).to(dtype)
        torch.cuda.synchronize(self.device)
        return PreparedGemm(a, b, c, reference, shape)

    def compile(self, prepared: PreparedGemm, config: KernelConfig) -> CompiledGemm:
        self.torch.cuda.set_device(self.device)
        from ._triton_kernel import gemm_kernel
        m, n, k = (prepared.shape.get(d) for d in ("M", "N", "K"))
        params = dict(config.params)
        grid = (self.triton.cdiv(m, params["BLOCK_M"]), self.triton.cdiv(n, params["BLOCK_N"]), prepared.shape.get("Batch"))
        gemm_kernel.warmup(prepared.a, prepared.b, prepared.c, m, n, k, grid=grid, **params)

        def launch():
            return gemm_kernel[grid](prepared.a, prepared.b, prepared.c, m, n, k, **params)

        return CompiledGemm(prepared, launch)

    def check_correctness(self, compiled: CompiledGemm) -> tuple[bool, dict[str, Any]]:
        self.torch.cuda.set_device(self.device)
        compiled.launch()
        self.torch.cuda.synchronize(self.device)
        p = compiled.prepared
        correct = bool(self.torch.allclose(p.c, p.reference, atol=self.atol, rtol=self.rtol))
        max_error = float((p.c.float() - p.reference.float()).abs().max().item())
        return correct, {"correctness_mode": "torch_reference", "verified_on_gpu": True,
                         "max_absolute_error": max_error if math.isfinite(max_error) else None,
                         "nonfinite_output": not math.isfinite(max_error), "atol": self.atol, "rtol": self.rtol}

    def warmup(self, compiled: CompiledGemm, iterations: int) -> None:
        self.torch.cuda.set_device(self.device)
        for _ in range(iterations):
            compiled.launch()
        self.torch.cuda.synchronize(self.device)

    def measure(self, compiled: CompiledGemm, repetitions: int) -> tuple[float, ...]:
        self.torch.cuda.set_device(self.device)
        samples = []
        for _ in range(repetitions):
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            compiled.launch()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        return tuple(samples)
