"""Reusable semantics, inputs and references for the four built-in operators.

Python implementations are small real CPU examples.  They are not GPU baselines.
CUDA SDKs are imported only while preparing or building a CUDA implementation.
"""

from __future__ import annotations

import math
import random
import struct
from typing import Any

from .torch_context import strict_fp32_reference
from .kernel_runtime import ExecutionContext
from .models import ShapeSpec
from .space import ValidationResult, gemm_constraint, reduction_constraint


def _cast(value: float, dtype: str) -> float:
    return struct.unpack("f", struct.pack("f", value))[0] if dtype == "float32" else float(value)


def _matrix(rows: int, columns: int, rng: random.Random, dtype: str, *, offset: float = 0.0, constant: bool = False):
    return [[_cast(offset + (0.125 if constant else rng.uniform(-0.25, 0.25)), dtype) for _ in range(columns)] for _ in range(rows)]


def _input_variant(ctx: ExecutionContext) -> str:
    """Reserved validation seeds exercise constant and offset distributions."""
    if ctx.seed == 104729:
        return "constant"
    if ctx.seed == 130363:
        return "offset"
    return "random"


def _generate_gemm_inputs(ctx: ExecutionContext, seed: int):
    m, n, k = (ctx.shape.get(name) for name in ("M", "N", "K"))
    batch = ctx.shape.get("Batch") if ctx.operator.kind == "batched_gemm" else 1
    variant = _input_variant(ctx)
    if ctx.runtime == "python":
        if ctx.operator.dtype not in {"float32", "float64"}:
            raise ValueError("the stdlib Python runtime supports float32/float64 inputs")
        if batch * (m * k + k * n + m * n) > 2_000_000:
            raise ValueError("Python built-in input generation is limited to 2,000,000 elements; use a CUDA Kernel for larger inputs")
        rng = random.Random(seed)
        offset = 8.0 if variant == "offset" else 0.0
        a = [_matrix(m, k, rng, ctx.operator.dtype, offset=offset, constant=variant == "constant") for _ in range(batch)]
        b = [_matrix(k, n, rng, ctx.operator.dtype, constant=variant == "constant") for _ in range(batch)]
        if ctx.operator.kind != "batched_gemm":
            a, b = a[0], b[0]
        ctx.allocate_output("out", (batch, m, n) if ctx.operator.kind == "batched_gemm" else (m, n))
    else:
        torch = ctx.torch
        dtype = getattr(torch, ctx.operator.dtype)
        device = f"cuda:{ctx.device}"
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        prefix = (batch,) if ctx.operator.kind == "batched_gemm" else ()
        if variant == "constant":
            a = torch.full((*prefix, m, k), 0.125, device=device, dtype=dtype)
            b = torch.full((*prefix, k, n), 0.125, device=device, dtype=dtype)
        else:
            a = torch.randn((*prefix, m, k), device=device, dtype=dtype, generator=generator) * 0.25
            b = torch.randn((*prefix, k, n), device=device, dtype=dtype, generator=generator) * 0.25
            if variant == "offset":
                a.add_(8.0)
        ctx.allocate_output("out", (*prefix, m, n))
    ctx.metadata["input_variant"] = variant
    return {"a": a, "b": b}


def _reference_gemm(ctx: ExecutionContext):
    a, b = ctx.inputs["a"], ctx.inputs["b"]
    if ctx.runtime == "cuda":
        torch = ctx.torch
        with strict_fp32_reference(torch):
            accumulation = torch.float64 if ctx.operator.dtype == "float64" else torch.float32
            return {"out": (a.to(accumulation) @ b.to(accumulation)).to(getattr(torch, ctx.operator.dtype))}
    aa, bb = (a, b) if ctx.operator.kind == "batched_gemm" else ([a], [b])
    result = []
    for lhs, rhs in zip(aa, bb):
        result.append([[_cast(math.fsum(x * rhs[index][column] for index, x in enumerate(row)), ctx.operator.dtype)
                        for column in range(len(rhs[0]))] for row in lhs])
    return {"out": result if ctx.operator.kind == "batched_gemm" else result[0]}


def _generate_reduction_inputs(ctx: ExecutionContext, seed: int):
    m, n = ctx.shape.get("M"), ctx.shape.get("N")
    variant = _input_variant(ctx)
    if ctx.runtime == "python":
        if ctx.operator.dtype not in {"float32", "float64"}:
            raise ValueError("the stdlib Python runtime supports float32/float64 inputs")
        if m * n > 2_000_000:
            raise ValueError("Python built-in input generation is limited to 2,000,000 elements")
        x = _matrix(m, n, random.Random(seed), ctx.operator.dtype,
                    offset=64.0 if variant == "offset" else 0.0, constant=variant == "constant")
    else:
        torch = ctx.torch
        dtype = getattr(torch, ctx.operator.dtype)
        generator = torch.Generator(device=f"cuda:{ctx.device}")
        generator.manual_seed(seed)
        if variant == "constant":
            x = torch.full((m, n), 0.125, device=f"cuda:{ctx.device}", dtype=dtype)
        else:
            x = torch.randn((m, n), device=f"cuda:{ctx.device}", dtype=dtype, generator=generator) * 0.25
            if variant == "offset":
                x.add_(64.0)
    ctx.allocate_output("out", (m, n))
    ctx.metadata["input_variant"] = variant
    return {"x": x}


def _reference_reduction(ctx: ExecutionContext):
    x = ctx.inputs["x"]
    if ctx.runtime == "cuda":
        torch = ctx.torch
        accumulation = torch.float64 if ctx.operator.dtype == "float64" else torch.float32
        xx = x.to(accumulation)
        if ctx.operator.kind == "softmax":
            result = torch.softmax(xx, dim=-1)
        else:
            import torch.nn.functional as functional
            result = functional.layer_norm(xx, (ctx.shape.get("N"),), weight=None, bias=None,
                                           eps=ctx.operator.parameters.get("eps", 1e-5))
        return {"out": result.to(getattr(torch, ctx.operator.dtype))}
    result = []
    for row in x:
        if ctx.operator.kind == "softmax":
            maximum = max(row)
            exponents = [math.exp(value - maximum) for value in row]
            total = math.fsum(exponents)
            result.append([_cast(value / total, ctx.operator.dtype) for value in exponents])
        else:
            mean = math.fsum(row) / len(row)
            variance = math.fsum((value - mean)**2 for value in row) / len(row)
            scale = math.sqrt(variance + ctx.operator.parameters.get("eps", 1e-5))
            result.append([_cast((value - mean) / scale, ctx.operator.dtype) for value in row])
    return {"out": result}


def _validate_gemm(ctx: ExecutionContext):
    required = {"M", "N", "K", "Batch"} if ctx.operator.kind == "batched_gemm" else {"M", "N", "K"}
    if ctx.shape.dims.keys() != required:
        return "GEMM requires exactly M/N/K, plus Batch for batched GEMM"
    if ctx.operator.parameters:
        return "the built-in GEMM semantics do not include transpose, broadcasting or fusion"
    return None


def _validate_reduction(ctx: ExecutionContext):
    if set(ctx.shape.dims) != {"M", "N"}:
        return "row reduction requires exactly M/N dimensions"
    axis = ctx.operator.parameters.get("axis", -1)
    if isinstance(axis, bool) or not isinstance(axis, int) or axis not in {-1, 1}:
        return "the built-in reduction semantics require the last axis"
    if ctx.operator.kind == "layernorm":
        if ctx.operator.parameters.get("affine", False):
            return "the built-in LayerNorm semantics are non-affine"
        eps = ctx.operator.parameters.get("eps", 1e-5)
        if isinstance(eps, bool) or not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
            return "LayerNorm eps must be finite and positive"
    return None


def _build_python_gemm(ctx: ExecutionContext, tile_size: int = 1):
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    def launch():
        a, b = ctx.inputs["a"], ctx.inputs["b"]
        aa, bb = (a, b) if ctx.operator.kind == "batched_gemm" else ([a], [b])
        result = []
        for lhs, rhs in zip(aa, bb):
            output = []
            for row in lhs:
                values = []
                for column in range(len(rhs[0])):
                    total = 0.0
                    for offset in range(0, len(row), tile_size):
                        for index in range(offset, min(offset + tile_size, len(row))):
                            total += row[index] * rhs[index][column]
                    values.append(_cast(total, ctx.operator.dtype))
                output.append(values)
            result.append(output)
        ctx.outputs["out"] = result if ctx.operator.kind == "batched_gemm" else result[0]
    return launch


def _build_python_reduction(ctx: ExecutionContext, tile_size: int = 1):
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    def launch():
        output = []
        for row in ctx.inputs["x"]:
            # Ordinary accumulation differs from the fsum-based reference.
            if ctx.operator.kind == "softmax":
                maximum = max(row)
                values = [math.exp(value - maximum) for value in row]
                total = 0.0
                for offset in range(0, len(values), tile_size):
                    total += sum(values[offset:offset + tile_size])
                output.append([_cast(value / total, ctx.operator.dtype) for value in values])
            else:
                mean = sum(sum(row[offset:offset + tile_size]) for offset in range(0, len(row), tile_size)) / len(row)
                variance = sum((value - mean)**2 for value in row) / len(row)
                scale = math.sqrt(variance + ctx.operator.parameters.get("eps", 1e-5))
                output.append([_cast((value - mean) / scale, ctx.operator.dtype) for value in row])
        ctx.outputs["out"] = output
    return launch


def _build_triton_gemm(ctx: ExecutionContext, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int,
                       num_warps: int, num_stages: int):
    import triton
    from .backends._triton_kernel import gemm_kernel
    m, n, k = (ctx.shape.get(name) for name in ("M", "N", "K"))
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N), ctx.shape.get("Batch"))
    args = (ctx.inputs["a"], ctx.inputs["b"], ctx.outputs["out"], m, n, k)
    parameters = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
                  "num_warps": num_warps, "num_stages": num_stages}
    gemm_kernel.warmup(*args, grid=grid, **parameters)
    def launch():
        gemm_kernel[grid](*args, **parameters)
    return launch


def _build_triton_reduction(ctx: ExecutionContext, BLOCK_SIZE: int, num_warps: int, num_stages: int):
    from .backends._triton_reductions import layernorm_kernel, softmax_kernel
    kernel = softmax_kernel if ctx.operator.kind == "softmax" else layernorm_kernel
    grid = (ctx.shape.get("M"),)
    args = (ctx.inputs["x"], ctx.outputs["out"], ctx.shape.get("N"))
    if ctx.operator.kind == "layernorm":
        args = (*args, ctx.operator.parameters.get("eps", 1e-5))
    parameters = {"BLOCK_SIZE": BLOCK_SIZE, "num_warps": num_warps, "num_stages": num_stages}
    kernel.warmup(*args, grid=grid, **parameters)
    def launch():
        kernel[grid](*args, **parameters)
    return launch


def register_builtin_assets(catalog) -> None:
    """Register independent semantic assets and eight reusable implementations."""
    from .assets import KernelDefinition, OperatorDefinition, ParameterSpec, SpaceDefinition

    version = "1.0.0"
    tolerances = {"float16": (0.02, 0.02), "bfloat16": (0.03, 0.03), "float32": (1e-5, 1e-4), "float64": (1e-10, 1e-9)}
    gemm_space = SpaceDefinition({"BLOCK_M": ParameterSpec((16, 32, 64, 128)),
                                 "BLOCK_N": ParameterSpec((32, 64, 128)), "BLOCK_K": ParameterSpec((32, 64)),
                                 "num_warps": ParameterSpec((4, 8)), "num_stages": ParameterSpec((2, 3, 4))},
                                constraints=(gemm_constraint,), name="triton_gemm_v1")
    reduction_space = SpaceDefinition({"BLOCK_SIZE": ParameterSpec((64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)),
                                      "num_warps": ParameterSpec((4, 8)), "num_stages": ParameterSpec((1, 2))},
                                     constraints=(reduction_constraint,), name="triton_row_reduction_v1")
    python_space = SpaceDefinition({"tile_size": ParameterSpec((1, 4, 16))}, name="python_tiles_v1")
    for name in ("gemm", "batched_gemm", "softmax", "layernorm"):
        gemm = name in {"gemm", "batched_gemm"}
        parameters = {} if gemm else {"axis": -1}
        if name == "layernorm":
            parameters.update({"eps": 1e-5, "affine": False})
        dims = ("M", "N", "K", "Batch") if name == "batched_gemm" else ("M", "N", "K") if gemm else ("M", "N")
        validation_shapes = ((ShapeSpec({"M": 3, "N": 5, "K": 7, "Batch": 2}), ShapeSpec({"M": 1, "N": 1, "K": 1, "Batch": 1})) if name == "batched_gemm" else
                             (ShapeSpec({"M": 3, "N": 5, "K": 7}), ShapeSpec({"M": 1, "N": 1, "K": 1})) if gemm else
                             (ShapeSpec({"M": 3, "N": 13}), ShapeSpec({"M": 1, "N": 1})))
        operator = OperatorDefinition(id=name, version=version, shape_dims=dims,
                                      dtypes=("float16", "bfloat16", "float32", "float64"),
                                      generate_inputs=_generate_gemm_inputs if gemm else _generate_reduction_inputs,
                                      reference=_reference_gemm if gemm else _reference_reduction,
                                      tolerances=tolerances, validation_shapes=validation_shapes,
                                      source_files=("torch_context.py",) if gemm else (),
                                      parameters=parameters, validate=_validate_gemm if gemm else _validate_reduction,
                                      metadata={"description": "Contiguous independent matrices" if gemm else "Non-affine row reduction",
                                                "validation_seeds": [0, 104729, 130363],
                                                "input_names": ["a", "b"] if gemm else ["x"], "output_names": ["out"]})
        catalog.register(operator)
        catalog.register(KernelDefinition(id=f"python/{name}", version=version, operator=operator.ref,
                                          build=_build_python_gemm if gemm else _build_python_reduction,
                                          space=python_space, runtime="python", dtypes=("float32", "float64"), architectures=("cpu",),
                                          metadata={"description": "Small real CPU implementation for interface acceptance; not a GPU performance baseline"}))
        catalog.register(KernelDefinition(id=f"triton/{name}", version=version, operator=operator.ref,
                                          build=_build_triton_gemm if gemm else _build_triton_reduction,
                                          space=gemm_space if gemm else reduction_space, runtime="cuda",
                                          dtypes=("float16", "bfloat16") if gemm else ("float16", "bfloat16", "float32"),
                                          architectures=("sm_*",),
                                          source_files=("backends/_triton_kernel.py",) if gemm else ("backends/_triton_reductions.py",),
                                          metadata={"requires_triton": True,
                                                                                         "description": "Portable complete-flow Triton implementation"}))


    from .builtin_assets import register_builtin_extensions
    register_builtin_extensions(catalog)


__all__ = ["register_builtin_assets"]
