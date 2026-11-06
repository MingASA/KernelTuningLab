"""Reproduce quantized-GEMM compiler resources without launching oversized CTAs."""

import argparse
import json
from pathlib import Path

from kernel_tuning_lab.catalog import get_catalog
from kernel_tuning_lab.kernel_runtime import AssetKernelAdapter, CompiledAsset, ExecutionPlan, ExecutionStep
from kernel_tuning_lab.models import KernelConfig, OperatorSpec, ShapeSpec


def main(output, dtype="bfloat16"):
    from kernel_tuning_lab.backends._triton_quantized import quantized_gemm_kernel
    rows = []
    catalog = get_catalog()
    for bits, bm in ((3, 64), (4, 16)):
        definition = catalog.resolve("operator", f"w{bits}a16_gemm")
        kernel = catalog.resolve("kernel", f"triton/w{bits}a16_gemm")
        adapter = AssetKernelAdapter(kernel, definition, accept_claims=False)
        hardware = adapter.probe_hardware()
        for group_size in (32, 64):
            operator = OperatorSpec(definition.id, kind=definition.id, implementation=kernel.ref,
                version=definition.version, dtype=dtype, parameters={"group_size": group_size})
            shape = ShapeSpec({"M": 2, "N": 7, "K": 65})
            prepared = adapter.prepare(hardware, operator, shape, 17)
            ctx = prepared.context
            for stages in (1, 2, 3, 4):
                config = KernelConfig({"BLOCK_M": bm, "BLOCK_N": 128, "BLOCK_K": 64,
                    "num_warps": 8, "num_stages": stages, "GROUP_M": 1, "SPLIT_K": 1})
                ctx.config = config
                args = (ctx.inputs["a"], ctx.inputs["packed"], ctx.inputs["scales"],
                        ctx.inputs["zero_points"], ctx.outputs["out"], 2, 7, 65)
                params = {**config.params, "BITS": bits, "GROUP_SIZE": group_size}
                compiled = quantized_gemm_kernel.warmup(*args, grid=(1, 1), **params)
                shared = compiled.metadata.shared
                correct = None
                if shared <= hardware.smem_per_block:
                    def launch():
                        quantized_gemm_kernel[(1, 1)](*args, **params)
                    wrapped = CompiledAsset(prepared, ExecutionPlan((ExecutionStep("resource_diagnostic", launch),)), 0, 0)
                    correct, _ = adapter.check_correctness(wrapped)
                    assert correct
                row = {"bits": bits, "shape": shape.dims, "dtype": operator.dtype, "seed": 17,
                       "group_size": group_size, "config": config.params, "compiled_shared_bytes": shared,
                       "physical_limit_bytes": hardware.smem_per_block, "direct_launch_correct": correct,
                       "launched": correct is not None, "environment": adapter.environment()}
                rows.append(row)
                Path(output).write_text(json.dumps(rows, indent=2) + "\n")
                print(bits, bm, group_size, stages, shared, "correct", correct, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16"))
    args = parser.parse_args()
    main(args.output, args.dtype)
