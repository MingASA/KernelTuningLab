"""Independent Kernels and a custom operator using only the public SDK.

Load this file as an asset plugin; no adapter or platform registration changes
are required.  These implementations execute on the CPU and report CPU timing.
"""

import math
import random
import struct

from kernel_tuning_lab.sdk import (
    ExecutionPlan, ExecutionStep, KernelDefinition, OperatorDefinition,
    ParameterSpec, ShapeSpec, SpaceDefinition,
)


def cast_output(value, dtype):
    return struct.unpack("f", struct.pack("f", value))[0] if dtype == "float32" else float(value)


def build_gemm(ctx, tile_size):
    """Reuse the platform's GEMM inputs, reference and default tolerances."""
    def launch():
        a, b = ctx.inputs["a"], ctx.inputs["b"]
        output = ctx.outputs["out"]
        for row_index, row in enumerate(a):
            for column in range(ctx.shape.get("N")):
                total = 0.0
                for offset in range(0, ctx.shape.get("K"), tile_size):
                    for index in range(offset, min(offset + tile_size, ctx.shape.get("K"))):
                        total += row[index] * b[index][column]
                output[row_index][column] = cast_output(total, ctx.operator.dtype)
    return launch


def build_gemm_pipeline(ctx, tile_size):
    """Split-K partial products followed by a separate reduction stage."""
    m, n, k = (ctx.shape.get(name) for name in ("M", "N", "K"))
    partials = ctx.allocate_workspace("split_k_partials", (2, m, n), dtype="float64")

    def setup():
        # Optional one-time work; allocation and setup stay outside timing.
        ctx.metadata["pipeline_setup_calls"] = ctx.metadata.get("pipeline_setup_calls", 0) + 1

    def initialize():
        # This reset runs inside correctness, warmup and every timed invocation.
        for partition in partials:
            for row in partition:
                for column in range(n):
                    row[column] = 0.0
        ctx.metadata["pipeline_initializations"] = ctx.metadata.get("pipeline_initializations", 0) + 1

    def compute_partials():
        a, b = ctx.inputs["a"], ctx.inputs["b"]
        midpoint = (k + 1) // 2
        for split, (start, stop) in enumerate(((0, midpoint), (midpoint, k))):
            for row in range(m):
                for column in range(n):
                    for offset in range(start, stop, tile_size):
                        for index in range(offset, min(offset + tile_size, stop)):
                            partials[split][row][column] += a[row][index] * b[index][column]

    def reduce_partials():
        output = ctx.outputs["out"]
        for row in range(m):
            for column in range(n):
                output[row][column] = cast_output(partials[0][row][column] + partials[1][row][column], ctx.operator.dtype)

    return ExecutionPlan(steps=(ExecutionStep("split_k_products", compute_partials),
                                ExecutionStep("reduce_partials", reduce_partials)),
                         initialize=initialize, setup=setup)


def validate_axpy(ctx):
    if ctx.runtime != "python":
        return "this example operator's input service supports the Python runtime"
    if set(ctx.shape.dims) != {"N"}:
        return "AXPY requires exactly the N dimension"
    alpha = ctx.operator.parameters.get("alpha", 1.5)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        return "AXPY alpha must be finite"
    if ctx.shape.get("N") > 1_000_000:
        return "this CPU example limits N to 1,000,000"
    return None


def generate_axpy_inputs(ctx, seed):
    rng = random.Random(seed)
    n = ctx.shape.get("N")
    ctx.allocate_output("out", (n,))
    return {"x": [cast_output(rng.uniform(-1.0, 1.0), ctx.operator.dtype) for _ in range(n)],
            "y": [cast_output(rng.uniform(-1.0, 1.0), ctx.operator.dtype) for _ in range(n)]}


def reference_axpy(ctx):
    alpha = ctx.operator.parameters.get("alpha", 1.5)
    return {"out": [cast_output(math.fsum((alpha * x, y)), ctx.operator.dtype)
                    for x, y in zip(ctx.inputs["x"], ctx.inputs["y"])]}


def build_axpy(ctx, tile_size):
    alpha = ctx.operator.parameters.get("alpha", 1.5)
    def launch():
        output = ctx.outputs["out"]
        for offset in range(0, ctx.shape.get("N"), tile_size):
            for index in range(offset, min(offset + tile_size, ctx.shape.get("N"))):
                output[index] = cast_output(alpha * ctx.inputs["x"][index] + ctx.inputs["y"][index], ctx.operator.dtype)
    return launch


TILES = SpaceDefinition({"tile_size": ParameterSpec((1, 2, 4), kind="integer")}, name="demo_cpu_tiles")

AXPY = OperatorDefinition(
    id="demo/axpy", version="1.0.0", shape_dims=("N",), dtypes=("float32", "float64"),
    generate_inputs=generate_axpy_inputs, reference=reference_axpy,
    tolerances={"float32": (1e-6, 1e-5), "float64": (1e-12, 1e-12)},
    parameters={"alpha": 1.5}, validate=validate_axpy,
    validation_shapes=(ShapeSpec({"N": 1}), ShapeSpec({"N": 7})),
    metadata={"input_names": ["x", "y"], "output_names": ["out"], "description": "out = alpha*x + y"},
)

ASSETS = (
    KernelDefinition(id="demo/gemm", version="1.0.0", operator="gemm@1.0.0", build=build_gemm,
                     space=TILES, runtime="python", dtypes=("float32", "float64"), architectures=("cpu",)),
    KernelDefinition(id="demo/gemm_pipeline", version="1.0.0", operator="gemm@1.0.0", build=build_gemm_pipeline,
                     space=TILES, runtime="python", dtypes=("float32", "float64"), architectures=("cpu",),
                     metadata={"description": "Two ordered stages with one-time workspace and per-invocation reset"}),
    AXPY,
    KernelDefinition(id="demo/axpy_kernel", version="1.0.0", operator=AXPY.ref, build=build_axpy,
                     space=TILES, runtime="python", dtypes=("float32", "float64"), architectures=("cpu",)),
)
