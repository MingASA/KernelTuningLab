"""Representative tutorial-style Tensor Core GEMM and a matched Torch baseline.

Registration is CPU-only; optional SDKs are imported only during actual build.
"""

from pathlib import Path

from ..sdk import KernelDefinition, ParameterSpec, SpaceDefinition


DOMAINS = {"BLOCK_M": (16, 32, 64, 128), "BLOCK_N": (32, 64, 128),
           "BLOCK_K": (32, 64), "num_warps": (4, 8), "num_stages": (2, 3), "GROUP_M": (1, 8)}
DESCRIPTIONS = {"BLOCK_M": "Output rows per CTA", "BLOCK_N": "Output columns per CTA",
    "BLOCK_K": "Reduction elements per pipelined dot iteration", "num_warps": "Triton compiler warps per CTA",
    "num_stages": "Triton compiler software-pipeline stages", "GROUP_M": "M tiles grouped before advancing N (clamped at tail)"}


def supports(hardware, operator, shape):
    if not hardware.architecture.startswith("sm_") or not hardware.architecture[3:].isdigit() or int(hardware.architecture[3:]) < 80:
        return "T1 requires NVIDIA sm_80 or later tensor cores; no Hopper-only instructions"
    if operator.kind != "gemm" or set(shape.dims) != {"M", "N", "K"} or operator.parameters:
        return "T1 implements contiguous, non-transposed, non-fused M/N/K GEMM"
    if max(shape.get("M") * shape.get("K"), shape.get("K") * shape.get("N"), shape.get("M") * shape.get("N")) >= 2**31:
        return "T1 address arithmetic is limited to signed int32 element offsets"


def resource_constraint(config, context):
    error = supports(context.hardware, context.operator, context.shape)
    if error:
        return error
    bm, bn, bk, stages, warps = (config.get(k) for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"))
    # Pipelined A16/B16 operands, before exact compiled-resource validation.
    estimate = 2 * (bm + bn) * bk * stages
    if context.hardware.smem_per_block and estimate > context.hardware.smem_per_block:
        return f"T1 pipelined operands need approximately {estimate} shared-memory bytes"
    if warps * context.hardware.warp_size > context.hardware.max_threads_per_block:
        return "T1 warp count exceeds device thread limit"


def build(ctx, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M):
    import triton
    from ..backends._triton_t1 import t1_gemm
    a, b, c = ctx.inputs["a"], ctx.inputs["b"], ctx.outputs["out"]
    m, n, k = (ctx.shape.get(key) for key in ("M", "N", "K"))
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    args = (a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride())
    config = dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                  num_warps=num_warps, num_stages=num_stages, GROUP_M=GROUP_M)
    compiled = t1_gemm.warmup(*args, grid=grid, **config)
    ctx.record_compiled_kernel("t1_gemm", compiled)
    ctx.metadata["execution_resources"]["t1_gemm"]["specialization"] = {
        "shape": dict(ctx.shape.dims), "strides": {"a": list(a.stride()), "b": list(b.stride()), "out": list(c.stride())},
        "dtype": ctx.operator.dtype, "config": config, "shape_arguments_constexpr": True}
    def launch():
        t1_gemm[grid](*args, **config)
    return launch


def build_torch(ctx):
    # Same preallocated C, same A/B and reference, same platform Graph boundary.
    def launch():
        ctx.torch.mm(ctx.inputs["a"], ctx.inputs["b"], out=ctx.outputs["out"])
    return launch


def register_assets(catalog):
    source = str(Path(__file__).parents[1] / "backends" / "_triton_t1.py")
    catalog.register(KernelDefinition("triton/t1_gemm", "1.0.0", "gemm@1.0.0", build,
        SpaceDefinition({name: ParameterSpec(values, description=DESCRIPTIONS[name]) for name, values in DOMAINS.items()},
                        constraints=(resource_constraint,), name="t1_gemm_v1"),
        runtime="cuda", dtypes=("float16", "bfloat16"), supports=supports, source_files=(source,),
        metadata={"requires_triton": True, "minimum_architecture": "sm_80",
            "description": "Explicitly tuned Tensor Core GEMM with grouped ordering and advancing pointers",
            "source": "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html",
            "license": "MIT (third_party/triton-tutorial-LICENSE)", "shape_specialization": "M/N/K and contiguous strides are constexpr"}))
    catalog.register(KernelDefinition("torch/mm", "1.0.0", "gemm@1.0.0", build_torch,
        SpaceDefinition({}, name="torch_mm_fixed"), runtime="cuda", dtypes=("float16", "bfloat16"), supports=supports,
        metadata={"description": "Torch mm library baseline with preallocated output; platform full-plan timing",
                  "compiled_identity": "unknown library internal dispatch, never estimated from config count"}))
