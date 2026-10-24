"""A mixed, conditional finite space; inactive knobs retain their declared default."""

import math
import struct

from kernel_tuning_lab.sdk import KernelDefinition, ParameterSpec, SpaceDefinition


def build(ctx, mode, tile_size, reverse_chunk):
    def launch():
        a, b, output = ctx.inputs["a"], ctx.inputs["b"], ctx.outputs["out"]
        k = ctx.shape.get("K")
        indices = list(range(k))
        if mode == "reverse":
            indices = [index for start in reversed(range(0, k, reverse_chunk))
                       for index in range(start, min(start + reverse_chunk, k))]
        for start in range(0, ctx.shape.get("M"), tile_size):
            for row in range(start, min(start + tile_size, ctx.shape.get("M"))):
                for column in range(ctx.shape.get("N")):
                    result = math.fsum(a[row][index] * b[index][column] for index in indices)
                    output[row][column] = (struct.unpack("f", struct.pack("f", result))[0]
                                           if ctx.operator.dtype == "float32" else result)
    return launch


ASSETS = (KernelDefinition(
    "demo/conditional_gemm", "1.0.0", "gemm@1.0.0", build,
    SpaceDefinition({"mode": ParameterSpec(("forward", "reverse")), "tile_size": ParameterSpec((1, 2, 4)),
                     "reverse_chunk": ParameterSpec((1, 2), active_when={"mode": ("reverse",)}, default=1)}),
    runtime="python", dtypes=("float32", "float64"), architectures=("cpu",)),)
