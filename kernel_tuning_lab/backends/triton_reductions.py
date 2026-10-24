"""Optional real softmax and non-affine LayerNorm on contiguous 2-D CUDA tensors.

Importing this module does not import torch or Triton. Adapter construction uses
the lazy GPU-runtime check inherited from TritonGemmAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable

from .triton_gemm import TritonGemmAdapter
from ..models import HardwareSpec, KernelConfig, OperatorSpec, ShapeSpec
from ..space import ValidationResult, reduction_constraint


@dataclass
class PreparedReduction:
    x: Any
    c: Any
    reference: Any
    shape: ShapeSpec
    hardware: HardwareSpec
    dtype: str
    eps: float = 1e-5


@dataclass
class CompiledReduction:
    prepared: PreparedReduction
    launch: Callable[[], Any]
    resources: dict[str, Any]


class _TritonReductionAdapter(TritonGemmAdapter):
    kind = ""
    _default_tolerances = {"float16": (1e-3, 1e-3), "bfloat16": (1e-2, 1e-2), "float32": (1e-5, 1e-5)}

    def __init__(self, device: int = 0, atol: float | None = None, rtol: float | None = None):
        for value in (atol, rtol):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("correctness tolerances must be finite and nonnegative")
        super().__init__(device=device, atol=0.0 if atol is None else atol, rtol=0.0 if rtol is None else rtol)
        self._atol_override, self._rtol_override = atol, rtol

    def environment(self) -> dict[str, Any]:
        result = super().environment()
        result["base_adapter_source_sha256"] = result["adapter_source_sha256"]
        result["adapter_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["kernel_source_sha256"] = hashlib.sha256(Path(__file__).with_name("_triton_reductions.py").read_bytes()).hexdigest()
        result["adapter"] = self.name
        return result

    def protocol_metadata(self) -> dict[str, Any]:
        result = super().protocol_metadata()
        result.update({"reference": f"torch.float32_{self.kind}_cast_to_output", "axis": -1,
                       "affine": False, "accumulation_dtype": "float32",
                       "atol": self._atol_override, "rtol": self._rtol_override,
                       "default_tolerances_by_dtype": self._default_tolerances,
                       "layout": "contiguous_row_major_2d", "backward": False,
                       "resource_validation": "compiled shared-memory usage; no vector-size SMEM assumption"})
        if self.kind == "layernorm":
            result.update({"eps_default": 1e-5,
                           "eps_source": "operator.parameters.eps; actual value is included in operator.id and correctness metadata"})
        return result

    def validate(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec,
                 config: KernelConfig) -> ValidationResult:
        reasons = []
        actual = self.probe_hardware()
        if (hardware.name, hardware.architecture) != (actual.name, actual.architecture):
            reasons.append("requested hardware does not match the physical CUDA device")
        if operator.kind != self.kind or operator.implementation != self.name:
            reasons.append(f"{type(self).__name__} requires kind={self.kind} and implementation={self.name}")
        reason = reduction_constraint(config, hardware, operator, shape)
        if reason:
            reasons.append(reason)
        if set(config.params) != {"BLOCK_SIZE", "num_warps", "num_stages"}:
            reasons.append("row reductions support BLOCK_SIZE, num_warps, num_stages only")
        if any(isinstance(config.get(key), bool) or not isinstance(config.get(key), int)
               for key in ("BLOCK_SIZE", "num_warps", "num_stages")):
            reasons.append("row reduction configuration values must be integers")
        allowed = {"axis"} if self.kind == "softmax" else {"axis", "eps", "affine", "elementwise_affine"}
        if operator.parameters.keys() - allowed:
            reasons.append("unsupported operator parameters: " + ", ".join(sorted(operator.parameters.keys() - allowed)))
        if self.kind == "layernorm" and operator.parameters.get("elementwise_affine", False):
            reasons.append("v0.1 layernorm implements non-affine forward only")
        if isinstance(operator.parameters.get("axis", -1), bool):
            reasons.append("axis must be the integer -1 or 1")
        if self.kind == "layernorm" and isinstance(operator.parameters.get("eps", 1e-5), bool):
            reasons.append("layernorm eps must be numeric, not boolean")
        if operator.dtype == "bfloat16" and actual.architecture.startswith("sm_"):
            capability = actual.architecture[3:]
            if capability.isdigit() and int(capability) < 80:
                reasons.append("bfloat16 requires NVIDIA compute capability >= 8.0")
        return ValidationResult(not reasons, tuple(reasons))

    def prepare(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec,
                seed: int) -> PreparedReduction:
        self.torch.cuda.set_device(self.device)
        torch = self.torch
        dtype = getattr(torch, operator.dtype)
        generator = torch.Generator(device=f"cuda:{self.device}")
        generator.manual_seed(seed)
        m, n = shape.get("M"), shape.get("N")
        x = torch.randn((m, n), device=f"cuda:{self.device}", dtype=dtype, generator=generator) * 0.5
        # Constant and large-offset rows exercise zero variance and stable exponentiation.
        if m > 1:
            x[0].fill_(0.25)
        if m > 2:
            x[1].mul_(20).add_(50)
        c = torch.empty_like(x)
        eps = float(operator.parameters.get("eps", 1e-5))
        if self.kind == "softmax":
            reference = torch.softmax(x.float(), dim=-1).to(dtype)
        else:
            reference = torch.nn.functional.layer_norm(x.float(), (n,), weight=None, bias=None, eps=eps).to(dtype)
        torch.cuda.synchronize(self.device)
        return PreparedReduction(x, c, reference, shape, hardware, operator.dtype, eps)

    def compile(self, prepared: PreparedReduction, config: KernelConfig) -> CompiledReduction:
        self.torch.cuda.set_device(self.device)
        from ._triton_reductions import layernorm_kernel, softmax_kernel
        kernel = softmax_kernel if self.kind == "softmax" else layernorm_kernel
        grid = (prepared.shape.get("M"),)
        arguments = (prepared.x, prepared.c, prepared.shape.get("N"))
        parameters = dict(config.params)
        if self.kind == "layernorm":
            parameters["EPS"] = prepared.eps
        compiled = kernel.warmup(*arguments, grid=grid, **parameters)
        metadata = getattr(compiled, "metadata", None)
        shared = getattr(metadata, "shared", None)
        if shared is not None and prepared.hardware.smem_per_block and shared > prepared.hardware.smem_per_block:
            raise RuntimeError(f"compiled shared-memory usage {shared} exceeds {prepared.hardware.smem_per_block} bytes")
        resources = {"shared_memory_bytes": shared, "registers_per_thread": getattr(compiled, "n_regs", None)}

        def launch():
            return kernel[grid](*arguments, **parameters)

        return CompiledReduction(prepared, launch, resources)

    def check_correctness(self, compiled: CompiledReduction) -> tuple[bool, dict[str, Any]]:
        self.torch.cuda.set_device(self.device)
        compiled.launch()
        self.torch.cuda.synchronize(self.device)
        prepared = compiled.prepared
        default_atol, default_rtol = self._default_tolerances[prepared.dtype]
        atol = default_atol if self._atol_override is None else self._atol_override
        rtol = default_rtol if self._rtol_override is None else self._rtol_override
        correct = bool(self.torch.allclose(prepared.c, prepared.reference, atol=atol, rtol=rtol))
        maximum = float((prepared.c.float() - prepared.reference.float()).abs().max().item())
        return correct, {"correctness_mode": "torch_float32_reference_cast_to_output", "verified_on_gpu": True,
                         "axis": -1, "affine": False, "eps": prepared.eps if self.kind == "layernorm" else None,
                         "atol": atol, "rtol": rtol, "dtype": prepared.dtype,
                         "max_absolute_error": maximum if math.isfinite(maximum) else None,
                         "nonfinite_output": not math.isfinite(maximum), "kernel_resources": compiled.resources,
                         "input_patterns": (["seeded_random"]
                                            + (["constant_first_row"] if prepared.shape.get("M") > 1 else [])
                                            + (["large_offset_second_row"] if prepared.shape.get("M") > 2 else []))}


class TritonSoftmaxAdapter(_TritonReductionAdapter):
    name = "triton_softmax"
    kind = "softmax"


class TritonLayerNormAdapter(_TritonReductionAdapter):
    name = "triton_layernorm"
    kind = "layernorm"


__all__ = ["TritonSoftmaxAdapter", "TritonLayerNormAdapter"]
