"""Portable operator registry: independent implementation contracts and spaces."""

from __future__ import annotations

from typing import Iterable

from .models import OperatorSpec, ShapeSpec
from .space import ConfigSpace, gemm_space, reduction_constraint


OPERATOR_NAMES = ("gemm", "batched_gemm", "softmax", "layernorm")


def operator_spec(name: str, backend: str = "mock", dtype: str = "float16") -> OperatorSpec:
    if name not in OPERATOR_NAMES:
        raise ValueError(f"unknown operator {name!r}; choose from {OPERATOR_NAMES}")
    prefix = "mock" if backend == "mock" else "triton"
    parameters = {"axis": -1} if name in {"softmax", "layernorm"} else {}
    if name == "layernorm":
        parameters.update({"eps": 1e-5, "affine": False})
    baselines = {"gemm": "torch.matmul", "batched_gemm": "torch.bmm",
                 "softmax": "torch.softmax(float32)", "layernorm": "torch.nn.functional.layer_norm(float32, affine=False)"}
    return OperatorSpec(name=name, kind=name, implementation=f"{prefix}_{name}", dtype=dtype,
                        parameters=parameters, correctness_baseline="synthetic_correctness_flag" if prefix == "mock" else baselines[name])


def operator_space(name: str, small: bool = False, shapes: Iterable[ShapeSpec] = ()) -> ConfigSpace:
    if name in {"gemm", "batched_gemm"}:
        space = gemm_space(small)
        space.name = f"triton_{name}_v1"
        return space
    if name not in {"softmax", "layernorm"}:
        raise ValueError(f"unknown operator: {name}")
    columns = tuple(shape.get("N") for shape in shapes)
    if columns:
        lo, hi = min(columns), max(columns)
        minimum, maximum = 1 << (lo - 1).bit_length(), 1 << (hi - 1).bit_length()
        blocks = tuple(1 << exponent for exponent in range(minimum.bit_length() - 1, maximum.bit_length()))
    else:
        blocks = (128, 256, 512, 1024, 2048) if small else (64, 128, 256, 512, 1024, 2048, 4096, 8192)
    return ConfigSpace({"BLOCK_SIZE": blocks, "num_warps": (4, 8), "num_stages": (1, 2)},
                       constraints=(reduction_constraint,), name=f"triton_{name}_v1")


def operator_demo_shapes(name: str) -> tuple[ShapeSpec, ...]:
    from .experiments import demo_shapes
    if name == "gemm":
        return demo_shapes()
    if name == "batched_gemm":
        return tuple(ShapeSpec({**shape.dims, "Batch": (2, 4, 8, 2, 4, 8, 2, 4)[i]}) for i, shape in enumerate(demo_shapes()))
    if name in {"softmax", "layernorm"}:
        return tuple(ShapeSpec({"M": m, "N": n}) for m, n in ((1, 64), (7, 193), (32, 128), (64, 256),
                                                             (128, 512), (4, 1024), (33, 1025), (128, 1024)))
    raise ValueError(f"unknown operator: {name}")
