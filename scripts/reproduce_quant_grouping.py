"""Compare legality rejection with a direct physical launch of that exact tile.

Diagnostic only: no bypassed measurement is inserted in research matrices.
"""

import argparse
import json
from pathlib import Path

from kernel_tuning_lab.assets import SpaceContext
from kernel_tuning_lab.builtin_assets.quantized import _gemm_constraint, _gemm_resolver
from kernel_tuning_lab.catalog import get_catalog
from kernel_tuning_lab.kernel_runtime import AssetKernelAdapter, CompiledAsset, ExecutionPlan, ExecutionStep
from kernel_tuning_lab.models import KernelConfig, OperatorSpec, ShapeSpec


def main(output):
    import triton
    from kernel_tuning_lab.backends._triton_quantized import quantized_gemm_kernel

    catalog = get_catalog()
    rows = []
    for bits in (4, 3, 2):
        for dtype in ("float16", "bfloat16"):
            definition = catalog.resolve("operator", f"w{bits}a16_gemm")
            kernel = catalog.resolve("kernel", f"triton/w{bits}a16_gemm")
            adapter = AssetKernelAdapter(kernel, definition, accept_claims=False)
            hardware = adapter.probe_hardware()
            operator = OperatorSpec(definition.id, kind=definition.id, implementation=kernel.ref,
                                    version=definition.version, dtype=dtype, parameters=dict(definition.parameters))
            shape = ShapeSpec({"M": 1, "N": 65, "K": 129})
            config = KernelConfig({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "num_warps": 4,
                                   "num_stages": 2, "GROUP_M": 8, "SPLIT_K": 1})
            context = SpaceContext(hardware, operator, shape)
            reason = _gemm_constraint(config, context)
            prepared = adapter.prepare(hardware, operator, shape, 7)
            ctx = prepared.context
            ctx.config = config
            params = {**config.params, "BITS": bits, "GROUP_SIZE": 32}
            args = (ctx.inputs["a"], ctx.inputs["packed"], ctx.inputs["scales"],
                    ctx.inputs["zero_points"], ctx.outputs["out"], 1, 65, 129)
            grid = (3, 1)
            compiled = quantized_gemm_kernel.warmup(*args, grid=grid, **params)
            def launch():
                quantized_gemm_kernel[grid](*args, **params)
            wrapped = CompiledAsset(prepared, ExecutionPlan((ExecutionStep("direct_diagnostic", launch),)), 0, 0)
            correct, details = adapter.check_correctness(wrapped)
            rows.append({"operator": definition.ref, "dtype": dtype, "shape": shape.dims, "config": config.params,
                         "seed": 7, "environment": adapter.environment(), "legality_reason": reason,
                         "resolved_group_m": _gemm_resolver(context)["GROUP_M"], "direct_launch_correct": correct,
                         "correctness": details, "compiled_shared_bytes": compiled.metadata.shared})
            Path(output).write_text(json.dumps(rows, indent=2) + "\n")
            assert correct, details
            print(bits, dtype, "rejection:", reason, "direct launch:", correct, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
