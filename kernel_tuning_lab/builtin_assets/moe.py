"""Ragged expert GEMM with compact scheduling, Split-K and optional TMA loads.

The inputs are already routed/grouped. Routing, top-k selection, scatter and the
rest of an MoE layer are separate operators and are not hidden in this timing.
"""

from __future__ import annotations

from bisect import bisect_right
from importlib.util import find_spec
import math
import random
import re
import struct

from ..torch_context import strict_fp32_reference
from ..sdk import (ExecutionPlan, ExecutionStep, KernelDefinition, OperatorDefinition,
                   ParameterSpec, ShapeSpec, SpaceDefinition)


_VERSION = "1.0.0"
_WORKSPACE_LIMIT = 256 * 1024**2
_SOURCES = (
    "https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/fused_moe/",
    "https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html",
    "https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html",
)


def _cast(value, dtype):
    return struct.unpack("f", struct.pack("f", value))[0] if dtype == "float32" else float(value)


def group_counts(total_rows: int, groups: int, distribution: str = "balanced") -> tuple[int, ...]:
    if (isinstance(total_rows, bool) or not isinstance(total_rows, int) or total_rows < 0
            or isinstance(groups, bool) or not isinstance(groups, int) or groups < 1):
        raise ValueError("group_counts requires nonnegative rows and positive groups")
    if distribution not in {"balanced", "skewed", "empty_first", "empty_last"}:
        raise ValueError("distribution must be balanced/skewed/empty_first/empty_last")
    if distribution == "skewed":
        return (total_rows,) + (0,) * (groups - 1)
    excluded = int(distribution in {"empty_first", "empty_last"})
    if excluded and groups < 2:
        raise ValueError("an explicitly empty expert requires at least two groups")
    active = groups - excluded
    base, remainder = divmod(total_rows, active)
    result = tuple(base + int(index < remainder) for index in range(active))
    return ((0,) + result if distribution == "empty_first" else result + (0,)
            if distribution == "empty_last" else result)


def validate_group_offsets(offsets, total_rows: int, groups: int) -> tuple[int, ...]:
    if not isinstance(offsets, (tuple, list)) or len(offsets) != groups + 1:
        raise ValueError("group_offsets must contain Groups+1 integers")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in offsets):
        raise ValueError("group_offsets must contain integer offsets")
    if offsets[0] != 0 or offsets[-1] != total_rows or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("group_offsets must be monotonic from zero through M; empty experts are allowed")
    return tuple(offsets)


def _counts(parameters, m, groups):
    if parameters.get("distribution", "balanced") not in {"balanced", "skewed", "empty_first", "empty_last"}:
        raise ValueError("distribution must be balanced/skewed/empty_first/empty_last")
    explicit = parameters.get("group_sizes")
    if explicit is None:
        return group_counts(m, groups, parameters.get("distribution", "balanced"))
    if (not isinstance(explicit, (tuple, list)) or len(explicit) != groups
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in explicit)
            or sum(explicit) != m):
        raise ValueError("group_sizes must contain Groups nonnegative integers summing to M")
    return tuple(explicit)


def _validation_cases(parameters, requested):
    if parameters.get("group_sizes") is not None:
        groups = len(parameters["group_sizes"])
        m = sum(parameters["group_sizes"])
        return (ShapeSpec({"M": m, "N": 5, "K": 7, "Groups": groups}),
                ShapeSpec({"M": m, "N": 1, "K": 1, "Groups": groups}))
    return (ShapeSpec({"M": 3, "N": 5, "K": 7, "Groups": 4}),
            ShapeSpec({"M": 1, "N": 1, "K": 1, "Groups": 2}))


def _validate(ctx):
    if set(ctx.shape.dims) != {"M", "N", "K", "Groups"}:
        return "MoE grouped GEMM requires M/N/K/Groups"
    m, n, k, groups = (ctx.shape.get(key) for key in ("M", "N", "K", "Groups"))
    try:
        _counts(ctx.operator.parameters, m, groups)
    except ValueError as exc:
        return str(exc)
    if max(m, n, k, groups) >= 2**31 or m * n >= 2**31:
        return "this implementation uses signed int32 scheduling/output indices"
    elements = m * k + groups * k * n + m * n
    if ctx.runtime == "python" and (elements > 2_000_000 or m * n * k > 50_000_000):
        return "stdlib CPU verification is limited to 2M elements and 50M multiply-add terms; use CUDA for production sizes"
    if ctx.runtime == "cuda" and ctx.operator.dtype not in {"float16", "bfloat16"}:
        return "CUDA MoE grouped GEMM supports float16/bfloat16"
    capability = re.fullmatch(r"sm_(\d+)", ctx.hardware.architecture)
    if ctx.runtime == "cuda" and capability and int(capability[1]) < 80:
        return "the current Triton runtime contract requires NVIDIA sm80+"
    if ctx.runtime == "python" and ctx.operator.dtype not in {"float32", "float64"}:
        return "Python MoE grouped GEMM supports float32/float64"
    if ctx.runtime == "cuda" and ctx.hardware.memory_bytes:
        peak = 2 * (m * k + groups * k * n) + 4 * m * n + 4 * k * n + 8 * groups
        if peak > ctx.hardware.memory_bytes * 0.65:
            return "input/output/reference estimate exceeds 65% of physical device capacity"
    return None


def _generate(ctx, seed):
    reason = _validate(ctx)
    if reason:
        raise ValueError(reason)
    m, n, k, groups = (ctx.shape.get(key) for key in ("M", "N", "K", "Groups"))
    counts = _counts(ctx.operator.parameters, m, groups)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    variant = "constant" if seed == 104729 else "offset" if seed == 130363 else "random"
    if ctx.runtime == "python":
        rng = random.Random(seed)
        def value(shift=0.0):
            return _cast(shift + (0.125 if variant == "constant" else rng.uniform(-0.25, 0.25)), ctx.operator.dtype)
        a = [[value(8.0 if variant == "offset" else 0.0) for _ in range(k)] for _ in range(m)]
        b = [[[value() for _ in range(n)] for _ in range(k)] for _ in range(groups)]
    else:
        torch, device = ctx.torch, f"cuda:{ctx.device}"
        dtype = getattr(torch, ctx.operator.dtype)
        generator = torch.Generator(device=device).manual_seed(seed)
        if variant == "constant":
            a = torch.full((m, k), 0.125, dtype=dtype, device=device)
            b = torch.full((groups, k, n), 0.125, dtype=dtype, device=device)
        else:
            a = torch.randn((m, k), dtype=dtype, device=device, generator=generator).mul_(0.25)
            b = torch.randn((groups, k, n), dtype=dtype, device=device, generator=generator).mul_(0.25)
            if variant == "offset":
                a.add_(8.0)
        offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    ctx.allocate_output("out", (m, n))
    ctx.metadata.update({"input_variant": variant, "expert_counts": list(counts), "routing_included": False})
    return {"a": a, "b": b, "group_offsets": offsets}


def _reference(ctx):
    m, n, k, groups = (ctx.shape.get(key) for key in ("M", "N", "K", "Groups"))
    a, b, offsets = (ctx.inputs[key] for key in ("a", "b", "group_offsets"))
    if ctx.runtime == "python":
        offsets = validate_group_offsets(offsets, m, groups)
        return {"out": [[_cast(math.fsum(a[row][kk] * b[bisect_right(offsets, row) - 1][kk][column]
                                         for kk in range(k)), ctx.operator.dtype)
                         for column in range(n)] for row in range(m)]}
    torch = ctx.torch
    host_offsets = validate_group_offsets(offsets.cpu().tolist(), m, groups)
    output = torch.empty((m, n), dtype=getattr(torch, ctx.operator.dtype), device=f"cuda:{ctx.device}")
    with strict_fp32_reference(torch):
        for expert in range(groups):
            start, stop = host_offsets[expert:expert + 2]
            if start == stop:
                continue
            weight = b[expert].float()
            for row in range(start, stop, 128):
                end = min(stop, row + 128)
                output[row:end] = (a[row:end].float() @ weight).to(output.dtype)
    return {"out": output}


def _build_python(ctx, tile_size):
    m, n, k, groups = (ctx.shape.get(key) for key in ("M", "N", "K", "Groups"))
    offsets = validate_group_offsets(ctx.inputs["group_offsets"], m, groups)
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("tile_size must be a positive integer")
    def launch():
        a, b, output = ctx.inputs["a"], ctx.inputs["b"], ctx.outputs["out"]
        for expert in range(groups):
            for block in range(offsets[expert], offsets[expert + 1], tile_size):
                for row in range(block, min(block + tile_size, offsets[expert + 1])):
                    for column in range(n):
                        value = 0.0
                        for kk in range(k):
                            value += a[row][kk] * b[expert][kk][column]
                        output[row][column] = _cast(value, ctx.operator.dtype)
    return launch


def _descriptor_available():
    try:
        return find_spec("triton.tools.tensor_descriptor") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _tma_allowed(hardware, shape):
    capability = re.fullmatch(r"sm_(\d+)", hardware.architecture)
    return bool(capability and int(capability[1]) >= 90 and shape.get("K") % 8 == 0
                and shape.get("N") % 8 == 0 and _descriptor_available())


def _workspace_bytes(shape, block_m, split_k):
    tiles = (shape.get("M") + block_m - 1) // block_m + shape.get("Groups")
    return 8 * tiles + 4 + (4 * split_k * shape.get("M") * shape.get("N") if split_k > 1 else 0)


def _constraint(config, context):
    hardware, operator, shape = context.hardware, context.operator, context.shape
    if operator.dtype not in {"float16", "bfloat16"}:
        return "CUDA MoE grouped GEMM requires float16/bfloat16"
    capability = re.fullmatch(r"sm_(\d+)", hardware.architecture)
    if capability and int(capability[1]) < 80:
        return "the current Triton runtime contract requires sm80+"
    bm, bn, bk = (config.get(key) for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
    warps, stages, split = (config.get(key) for key in ("num_warps", "num_stages", "SPLIT_K"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (bm, bn, bk, warps, stages, split)):
        return "tile/warp/stage/Split-K values must be positive integers"
    if any(value & (value - 1) for value in (bm, bn, bk, warps, split)) or min(bm, bn) < 16 or bk < 32:
        return "Tensor Core tiles require powers of two, M/N>=16, K>=32"
    if warps * hardware.warp_size > hardware.max_threads_per_block:
        return "warps exceed physical thread limit"
    for name in ("MAP_BLOCK_SIZE", "MAP_NUM_WARPS", "GROUP_M", "PERSISTENT_CTA_MULT", "REDUCE_BLOCK_SIZE"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value & (value - 1):
            return f"{name} must be a positive power of two"
    if config.get("MAP_NUM_WARPS") * hardware.warp_size > hardware.max_threads_per_block:
        return "schedule preprocessing warps exceed the physical thread limit"
    if type(config.get("USE_TMA")) is not bool:
        return "USE_TMA must be boolean"
    if split > (shape.get("K") + bk - 1) // bk:
        return "Split-K exceeds the number of nonempty K tiles"
    if _workspace_bytes(shape, bm, split) > context.workspace_memory_limit_bytes:
        return "Split-K/scheduling workspace exceeds the declared execution allowance"
    if config.get("USE_TMA") and not _tma_allowed(hardware, shape):
        return "TMA needs sm90+, TensorDescriptor software support, and 16-byte contiguous matrix strides"
    if config.get("SCHEDULE") not in {"grouped", "persistent"}:
        return "unknown expert tile scheduling strategy"
    return None


def _resource_advisory(config, hardware, operator, shape):
    estimate = 2 * (config.get('BLOCK_M') + config.get('BLOCK_N')) * config.get('BLOCK_K') * config.get('num_stages')
    return {'kind':'resource_estimate','model':'moe_pipelined_operands_v1',
            'estimated_shared_memory_bytes':estimate,'device_limit_bytes':hardware.smem_per_block or None,
            'excludes_candidate':False,'calibration':'Conservative model, compiler-layout dependent',
            'authoritative_check':'Actual compiler resources and execution'}


def _resolve_space(context):
    hardware, operator, shape = context.hardware, context.operator, context.shape
    k_tiles = (shape.get("K") + 31) // 32
    splits = tuple(value for value in (1, 2, 4, 8) if value <= k_tiles and _workspace_bytes(shape, 128, value) <= context.workspace_memory_limit_bytes)
    stages = (1, 2, 3, 4, 5)
    return {"USE_TMA": (False, True) if _tma_allowed(hardware, shape) else (False,),
            "SPLIT_K": splits or (1,), "num_stages": stages or (1,)}


def _record_resources(ctx, name, compiled):
    if hasattr(ctx, "record_compiled_kernel"):
        ctx.record_compiled_kernel(name, compiled)
        return
    shared = getattr(getattr(compiled, "metadata", None), "shared", None)
    if shared is not None and ctx.hardware.smem_per_block and shared > ctx.hardware.smem_per_block:
        raise RuntimeError("compiled Kernel exceeds shared-memory limit")
    ctx.metadata.setdefault("execution_resources", {})[name] = {"shared_memory_bytes": shared}


def _build_triton(ctx, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M,
                   SPLIT_K, SCHEDULE, PERSISTENT_CTA_MULT, REDUCE_BLOCK_SIZE, USE_TMA,
                   MAP_BLOCK_SIZE, MAP_NUM_WARPS):
    reason = _constraint(ctx.config, ctx)
    if reason:
        raise ValueError(reason)
    import triton
    from ..backends._triton_moe import build_expert_tile_map, grouped_expert_gemm, reduce_expert_split_k
    m, n, k, groups = (ctx.shape.get(key) for key in ("M", "N", "K", "Groups"))
    offsets = ctx.inputs["group_offsets"]
    if offsets.dtype != ctx.torch.int32 or offsets.device != ctx.inputs["a"].device or not offsets.is_contiguous():
        raise ValueError("group_offsets must be contiguous int32 on the input device")
    validate_group_offsets(offsets.cpu().tolist(), m, groups)
    max_tiles = triton.cdiv(m, BLOCK_M) + groups
    experts = ctx.allocate_workspace("tile_expert", (max_tiles,), "int32")
    rows = ctx.allocate_workspace("tile_row", (max_tiles,), "int32")
    counter = ctx.allocate_workspace("tile_count", (1,), "int32")
    partial = (ctx.allocate_workspace("split_k_partials", (SPLIT_K, m, n), "float32", fill=None)
               if SPLIT_K > 1 else ctx.outputs["out"])
    a, b = ctx.inputs["a"], ctx.inputs["b"]
    expected_dtype = getattr(ctx.torch, ctx.operator.dtype)
    if (tuple(a.shape) != (m, k) or tuple(b.shape) != (groups, k, n)
            or not a.is_contiguous() or not b.is_contiguous() or a.dtype != expected_dtype
            or b.dtype != expected_dtype or a.device != b.device):
        raise ValueError("a/b must be contiguous same-device/dtype tensors with shapes [M,K] and [Groups,K,N]")
    output = ctx.outputs["out"]
    if tuple(output.shape) != (m, n) or output.dtype != expected_dtype or output.device != a.device or not output.is_contiguous():
        raise ValueError("output must be contiguous [M,N] on the input device/dtype")
    a_desc, b_desc = a, b
    if USE_TMA:
        from triton.tools.tensor_descriptor import TensorDescriptor
        a_desc = TensorDescriptor(a, [m, k], [k, 1], [BLOCK_M, BLOCK_K])
        b_desc = TensorDescriptor(b, [groups, k, n], [k * n, n, 1], [1, BLOCK_K, BLOCK_N])
    physical_programs = max_tiles * triton.cdiv(n, BLOCK_N)
    if SCHEDULE == "persistent":
        physical_programs = min(physical_programs, max(1, ctx.hardware.sm_count) * PERSISTENT_CTA_MULT)
    map_args = (offsets, experts, rows, counter)
    map_params = {"GROUPS": groups, "BLOCK_M": BLOCK_M, "MAP_BLOCK_SIZE": MAP_BLOCK_SIZE, "num_warps": MAP_NUM_WARPS}
    mat_args = (a, b, partial, a_desc, b_desc, offsets, experts, rows, counter)
    mat_params = {"M": m, "N": n, "K": k, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
                  "BLOCK_K": BLOCK_K, "GROUP_M": GROUP_M, "SPLIT_K": SPLIT_K,
                  "NUM_PROGRAMS": physical_programs, "PIPELINE_STAGES": num_stages,
                  "USE_TMA": USE_TMA, "num_warps": num_warps, "num_stages": num_stages}
    _record_resources(ctx, "build_expert_tile_map", build_expert_tile_map.warmup(*map_args, grid=(groups,), **map_params))
    _record_resources(ctx, "grouped_expert_gemm", grouped_expert_gemm.warmup(*mat_args, grid=(physical_programs, SPLIT_K), **mat_params))

    def initialize():
        counter.zero_()
    def make_map():
        build_expert_tile_map[(groups,)](*map_args, **map_params)
    def multiply():
        grouped_expert_gemm[(physical_programs, SPLIT_K)](*mat_args, **mat_params)
    steps = [ExecutionStep("build_expert_tile_map", make_map), ExecutionStep("grouped_expert_gemm", multiply)]
    if SPLIT_K > 1:
        reduce_args = (partial, ctx.outputs["out"])
        reduce_params = {"ELEMENTS": m * n, "SPLIT_K": SPLIT_K, "BLOCK_SIZE": REDUCE_BLOCK_SIZE,
                         "num_warps": 4}
        grid = (triton.cdiv(m * n, REDUCE_BLOCK_SIZE),)
        _record_resources(ctx, "reduce_expert_split_k", reduce_expert_split_k.warmup(*reduce_args, grid=grid, **reduce_params))
        def reduce():
            reduce_expert_split_k[grid](*reduce_args, **reduce_params)
        steps.append(ExecutionStep("reduce_expert_split_k", reduce))
    ctx.metadata["expert_schedule"] = {"max_tiles": max_tiles, "physical_programs": physical_programs,
                                        "schedule": SCHEDULE, "tma": USE_TMA, "split_k": SPLIT_K,
                                        "preprocessing_in_timing": True}
    return ExecutionPlan(tuple(steps), initialize=initialize)


def register_assets(catalog):
    operator = OperatorDefinition(
        "moe_grouped_gemm", _VERSION, ("M", "N", "K", "Groups"),
        ("float16", "bfloat16", "float32", "float64"), _generate, _reference,
        {"float16": (0.02, 0.02), "bfloat16": (0.03, 0.03), "float32": (1e-5, 1e-4), "float64": (1e-10, 1e-9)},
        validation_shapes=_validation_cases({}, ()), parameters={"distribution": "balanced", "group_sizes": None},
        validate=_validate, validation_case_factory=_validation_cases,
        source_files=("../torch_context.py",),
        metadata={"input_names": ["a", "b", "group_offsets"], "output_names": ["out"],
                  "validation_seeds": [0, 104729, 130363], "proposal_family": "MoEGroupedGEMM",
                  "smoke_shapes": [{"M": 7, "N": 13, "K": 35, "Groups": 4},
                                   {"M": 13, "N": 32, "K": 64, "Groups": 4},
                                   {"M": 2, "N": 7, "K": 9, "Groups": 8}],
                  "validation_parameter_cases": [{"name": name, "parameters": {"distribution": name}}
                                                  for name in ("skewed", "empty_first", "empty_last")],
                  "scope": "already routed contiguous expert inputs; empty experts allowed; routing/scatter not included",
                  "primary_sources": list(_SOURCES),
                  "production_shapes": [{"M": 4096, "N": 4096, "K": 4096, "Groups": 8},
                                        {"M": 128, "N": 14336, "K": 4096, "Groups": 8}]})
    catalog.register(operator)
    catalog.register(KernelDefinition(
        "python/moe_grouped_gemm", _VERSION, operator.ref, _build_python,
        SpaceDefinition({"tile_size": ParameterSpec((1, 4, 16), description="CPU expert-row blocking", unit="rows")}),
        runtime="python", dtypes=("float32", "float64"), architectures=("cpu",),
        metadata={"scope": "small real CPU verification; not a GPU performance baseline"}))
    parameters = {
        "BLOCK_M": ParameterSpec((16, 32, 64, 128), description="Routed rows per matmul tile", unit="rows"),
        "BLOCK_N": ParameterSpec((32, 64, 128, 256), description="Expert output columns per tile", unit="columns"),
        "BLOCK_K": ParameterSpec((32, 64, 128), description="Tensor Core reduction tile", unit="elements"),
        "num_warps": ParameterSpec((4, 8), description="Warps per expert compute CTA"),
        "num_stages": ParameterSpec((1, 2, 3, 4, 5), description="Operand software pipeline stages"),
        "GROUP_M": ParameterSpec((1, 4, 8), description="Tile ordering group for expert-weight L2 reuse"),
        "SPLIT_K": ParameterSpec((1, 2, 4, 8), description="K partitions with FP32 workspace and timed reduction"),
        "SCHEDULE": ParameterSpec(("grouped", "persistent"), description="Full grid or SM-bounded persistent traversal"),
        "PERSISTENT_CTA_MULT": ParameterSpec((1, 2, 4), active_when={"SCHEDULE": ("persistent",)}, default=1,
                                               description="Persistent CTAs per physical SM"),
        "REDUCE_BLOCK_SIZE": ParameterSpec((256, 512, 1024), active_when={"SPLIT_K": (2, 4, 8)}, default=256,
                                           description="Split-K reduction vector length", unit="elements"),
        "USE_TMA": ParameterSpec((False, True), description="Actual TensorDescriptor loads; gated by hardware/software/alignment"),
        "MAP_BLOCK_SIZE": ParameterSpec((32, 64, 128, 256), description="Vectorized expert-schedule preprocessing block", unit="tiles"),
        "MAP_NUM_WARPS": ParameterSpec((1, 2, 4), description="Warps per expert-schedule preprocessing CTA"),
    }
    catalog.register(KernelDefinition(
        "triton/moe_grouped_gemm", _VERSION, operator.ref, _build_triton,
        SpaceDefinition(parameters, constraints=(_constraint,), name="ragged_expert_production_space", resolver=_resolve_space, advisories=(_resource_advisory,)),
        runtime="cuda", dtypes=("float16", "bfloat16"), architectures=("sm_*",),
        source_files=("../backends/_triton_moe.py",),
        metadata={"requires_triton": True, "primary_sources": list(_SOURCES),
                  "optimizations": ["compact expert tile map", "no padded expert matrices", "persistent traversal",
                                    "grouped L2 ordering", "Split-K with FP32 reduction", "optional real TMA operand loads"],
                  "tma_requirement": "compute capability >=9, Triton host TensorDescriptor API (3.4+), K/N 16-byte strides",
                  "default_workspace_limit_bytes": _WORKSPACE_LIMIT}))


__all__ = ["register_assets", "group_counts", "validate_group_offsets"]
