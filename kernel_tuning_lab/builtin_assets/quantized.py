"""W4/W3/W2 affine dequantization and A16 weight-only GEMM assets.

Only stdlib and CPU-safe project declarations are imported at discovery time.
The format is canonical_unsigned_affine_k_bitstream: signed int32 storage,
unsigned codes, K-contiguous LSB-first packing, including straddling W3 codes.
No automatic compatibility with GPTQ or AWQ checkpoint layouts is implied.
"""

from __future__ import annotations

import math
import random
import struct

from ..assets import KernelDefinition, OperatorDefinition, ParameterSpec, SpaceContext, SpaceDefinition
from ..kernel_runtime import ExecutionContext, ExecutionPlan, ExecutionStep
from ..models import KernelConfig, ShapeSpec
from ..quantization import PACKING_FORMAT, pack_bits, unpack_bits, dequantize_packed


_VERSION = "1.0.0"
_BITS = {name: bits for bits in (4, 3, 2)
         for name in (f"dequantize_w{bits}", f"w{bits}a16_gemm")}
_MAX_CPU_ELEMENTS = 2_000_000
_MAX_CPU_PRODUCTS = 20_000_000
_MAX_INDEX = (1 << 31) - 1
_DEFAULT_WORKSPACE_BYTES = 256 * 1024**2
_REFERENCE_TILE_ELEMENTS = 2_000_000
_DESIGN_SOURCES = (
    "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq_triton.py",
    "https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html",
)
_TOLERANCES = {"float16": (0.02, 0.02), "bfloat16": (0.04, 0.03),
               "float32": (1e-5, 1e-4), "float64": (1e-10, 1e-9)}
_GEMM_DOMAINS = {"BLOCK_M": (16, 32, 64, 128), "BLOCK_N": (32, 64, 128),
                 "BLOCK_K": (32, 64), "num_warps": (4, 8), "num_stages": (1, 2, 3, 4),
                 "GROUP_M": (1, 2, 4, 8), "SPLIT_K": (1, 2, 4, 8, 16)}
_DEQUANT_BLOCKS = (128, 256, 512, 1024)
_KNOB_DESCRIPTIONS = {
    "BLOCK_M": "Output row tile for each GEMM program", "BLOCK_N": "Contiguous output-column GEMM tile",
    "BLOCK_K": "Reduction tile decoded inside each fused GEMM loop",
    "num_warps": "Warps per Triton program", "num_stages": "Compiler software-pipeline stages for GEMM",
    "GROUP_M": "Number of M tiles grouped before advancing along N for L2 reuse",
    "SPLIT_K": "Independent K partitions; values above one add timed FP32 partial reduction",
}


def _cast(value, dtype):
    return struct.unpack("f", struct.pack("f", value))[0] if dtype == "float32" else float(value)


def _is_gemm(operator):
    return operator.kind.endswith("a16_gemm")


def _group_size(operator):
    return operator.parameters.get("group_size", 32)


def _semantic_error(operator, shape):
    if operator.kind not in _BITS:
        return "unknown built-in quantized operator"
    expected = {"M", "N", "K"} if _is_gemm(operator) else {"K", "N"}
    if set(shape.dims) != expected:
        return f"{operator.kind} requires exactly {'/'.join(sorted(expected))} dimensions"
    if operator.parameters.keys() - {"group_size"}:
        return "quantized operators accept only the group_size parameter"
    group_size = _group_size(operator)
    if isinstance(group_size, bool) or not isinstance(group_size, int) or not 0 < group_size <= _MAX_INDEX:
        return "group_size must be a positive integer <= 2**31-1; a partial last group is supported"
    return None


def _input_sizes(operator, shape):
    k, n = shape.get("K"), shape.get("N")
    m = shape.get("M") if _is_gemm(operator) else 0
    words = (k * _BITS[operator.kind] + 31) // 32
    groups = (k + _group_size(operator) - 1) // _group_size(operator)
    return m, n, k, words, groups


def _cuda_chunk_rows(n):
    # Whole 32-code units align to a word boundary for all supported bit widths.
    return max(32, (_REFERENCE_TILE_ELEMENTS // n // 32) * 32)


def _size_error(operator, shape, hardware, runtime):
    m, n, k, words, groups = _input_sizes(operator, shape)
    elements = words * n + 2 * groups * n + (m * k + m * n if m else k * n)
    if runtime == "python":
        if elements + k * n > _MAX_CPU_ELEMENTS:
            return "stdlib quantized input/reference generation is limited to 2,000,000 logical elements"
        if m and m * k * n > _MAX_CPU_PRODUCTS:
            return "stdlib quantized GEMM is limited to 20,000,000 scalar products; use a CUDA Kernel"
    else:
        if any(value > _MAX_INDEX for value in (k * _BITS[operator.kind], k * n, words * n,
                                                groups * n, m * k, m * n)):
            return "quantized CUDA inputs must fit signed 32-bit indexing"
        # Integer generation and the independent reference process bounded K
        # chunks; there is no full K*N int64 codes or FP32 dequantized matrix.
        width = 2 if operator.dtype in {"float16", "bfloat16"} else 8 if operator.dtype == "float64" else 4
        estimate = words * n * 4 + groups * n * (width + 4)
        estimate += min(k, _cuda_chunk_rows(n)) * n * 64
        estimate += (m * k * (width + 4) + m * n * (width * 3 + 4)) if m else k * n * width * 3
        if hardware.memory_bytes and estimate > hardware.memory_bytes * 0.75:
            return "quantized input/reference preparation estimate exceeds 75% of device memory"
    return None


def _validate_operator(ctx):
    reason = _semantic_error(ctx.operator, ctx.shape)
    if reason:
        return reason
    return _size_error(ctx.operator, ctx.shape, ctx.hardware, ctx.runtime)


def _python_constraint(config, hardware, operator, shape):
    reason = _semantic_error(operator, shape)
    if reason:
        return reason
    if operator.dtype not in {"float32", "float64"}:
        return "stdlib quantized Kernels support float32/float64"
    tile = config.get("tile_size")
    if isinstance(tile, bool) or not isinstance(tile, int) or tile <= 0:
        return "tile_size must be a positive integer"
    return _size_error(operator, shape, hardware, "python")


def _cuda_supports(hardware, operator, shape):
    reason = _semantic_error(operator, shape)
    if reason:
        return reason
    if operator.dtype not in {"float16", "bfloat16"}:
        return "quantized Triton Kernels implement float16/bfloat16 only"
    architecture = hardware.architecture
    if not architecture.startswith("sm_") or not architecture[3:].isdigit():
        return "quantized Triton Kernels require an NVIDIA sm_* architecture"
    if int(architecture[3:]) < 80:
        return "FP16/BF16 quantized Kernels require sm_80+ for the current Triton runtime contract"
    return _size_error(operator, shape, hardware, "cuda")


def _thread_error(config, hardware):
    warps = config.get("num_warps")
    if isinstance(warps, bool) or not isinstance(warps, int) or warps not in (4, 8):
        return "num_warps must be 4 or 8"
    if warps * hardware.warp_size > hardware.max_threads_per_block:
        return "num_warps exceeds the device's threads-per-block limit"
    return None


def _dequant_constraint(config, hardware, operator, shape):
    reason = _cuda_supports(hardware, operator, shape) or _thread_error(config, hardware)
    if reason:
        return reason
    block = config.get("BLOCK_SIZE")
    if isinstance(block, bool) or not isinstance(block, int) or block not in (128, 256, 512, 1024):
        return "dequantization BLOCK_SIZE must be 128, 256, 512, or 1024"
    return None


def _gemm_constraint(config, context, *, fused=True):
    hardware, operator, shape = context.hardware, context.operator, context.shape
    reason = _cuda_supports(hardware, operator, shape) or _thread_error(config, hardware)
    if reason:
        return reason
    if not _is_gemm(operator):
        return "the quantized GEMM space requires an A16 GEMM operator"
    bm, bn, bk = (config.get(name) for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value & (value - 1)
           for value in (bm, bn, bk)):
        return "GEMM block sizes must be positive powers of two"
    if bm < 16 or bn < 16 or bk < 32:
        return "A16 dot requires BLOCK_M/N >=16 and BLOCK_K >=32"
    stages = config.get("num_stages")
    if isinstance(stages, bool) or not isinstance(stages, int) or stages not in (1, 2, 3, 4):
        return "num_stages must be 1, 2, 3, or 4"
    group = config.get("GROUP_M")
    if isinstance(group, bool) or not isinstance(group, int) or group not in (1, 2, 4, 8):
        return "GROUP_M must be 1, 2, 4, or 8"
    # _program_coordinates clamps the final group to the remaining row tiles.
    # GROUP_M is an ordering preference, not a minimum number of M tiles.
    split = config.get("SPLIT_K")
    if isinstance(split, bool) or not isinstance(split, int) or split not in (1, 2, 4, 8, 16):
        return "SPLIT_K must be 1, 2, 4, 8, or 16"
    if split > (shape.get("K") + bk - 1) // bk:
        return "SPLIT_K cannot exceed the number of nonempty K tiles"
    if split > 1:
        partial_bytes = split * shape.get("M") * shape.get("N") * 4
        if partial_bytes > context.workspace_memory_limit_bytes:
            return f"Split-K FP32 partials exceed the {context.workspace_memory_limit_bytes}-byte execution workspace budget"
        if split * shape.get("M") * shape.get("N") > _MAX_INDEX:
            return "Split-K FP32 partials exceed signed 32-bit indexing"
    estimate = (bm * bk + bk * bn) * 2 * stages
    if fused:
        # The fused loop pipelines integer packed-word and zero-point loads,
        # not just the final A16 dot operands. W3 loads a second packed word.
        # Physical sm120/Triton 3.7 resource probes agree with this layout;
        # retain the compiler resource check for layout/version differences.
        operands = 2 * (bm * bk + bk * bn)
        group_size = _group_size(operator)
        zero_elements = bn if group_size >= bk and group_size % bk == 0 else bk * bn
        raw_loads = 4 * (bk * bn * (2 if _BITS[operator.kind] == 3 else 1) + zero_elements)
        iterations = (shape.get("K") + bk * split - 1) // (bk * split)
        estimate = operands + (stages - 1) * raw_loads if iterations > 1 else operands
    if hardware.smem_per_block and estimate > hardware.smem_per_block:
        return f"{'fused integer staging' if fused else 'staged A16 tile'} estimate {estimate} exceeds {hardware.smem_per_block} shared-memory bytes"
    return None


def _pipeline_constraint(config, context):
    reason = _gemm_constraint(config, context, fused=False)
    if reason:
        return reason
    shape = context.shape
    block = config.get("DEQUANT_BLOCK_SIZE")
    if isinstance(block, bool) or not isinstance(block, int) or block not in (128, 256, 512, 1024):
        return "DEQUANT_BLOCK_SIZE must be 128, 256, 512, or 1024"
    # The declared execution budget governs both materialized weights and
    # optional partials, matching ExecutionContext's allocation accounting.
    split = config.get("SPLIT_K")
    workspace_bytes = shape.get("K") * shape.get("N") * 2
    if split > 1:
        workspace_bytes += split * shape.get("M") * shape.get("N") * 4
    if workspace_bytes > context.workspace_memory_limit_bytes:
        return f"dequantized weights and Split-K partials exceed the {context.workspace_memory_limit_bytes}-byte execution workspace budget"
    return None


def _gemm_space_domains(context, *, pipeline=False):
    """Prune provably impossible individual values without enumerating products.

    Each shared-memory check uses the smallest remaining companion tiles and
    stage count. Correlated configurations still pass the space constraint.
    Unsupported cases preserve declarations so capability validation supplies
    the reason instead of inventing an empty domain.
    """
    hardware, operator, shape = context.hardware, context.operator, context.shape
    if _cuda_supports(hardware, operator, shape):
        return dict(_GEMM_DOMAINS)
    result = dict(_GEMM_DOMAINS)
    m, n, k = (shape.get(name) for name in ("M", "N", "K"))
    result["num_warps"] = tuple(w for w in result["num_warps"]
                                if w * hardware.warp_size <= hardware.max_threads_per_block)
    weight_bytes = k * n * 2 if pipeline else 0
    result["SPLIT_K"] = tuple(s for s in result["SPLIT_K"] if s <= (k + 31) // 32
                              and weight_bytes + (s * m * n * 4 if s > 1 else 0) <= context.workspace_memory_limit_bytes
                              and (s == 1 or s * m * n <= _MAX_INDEX))
    if hardware.smem_per_block:
        limit = hardware.smem_per_block
        result["BLOCK_M"] = tuple(bm for bm in result["BLOCK_M"] if (bm * 32 + 32 * 32) * 2 <= limit)
        result["BLOCK_N"] = tuple(bn for bn in result["BLOCK_N"] if (16 * 32 + 32 * bn) * 2 <= limit)
        result["BLOCK_K"] = tuple(bk for bk in result["BLOCK_K"] if (16 * bk + bk * 32) * 2 <= limit)
        result["num_stages"] = tuple(s for s in result["num_stages"] if (16 * 32 + 32 * 32) * 2 * s <= limit)
    return {name: values if values else _GEMM_DOMAINS[name] for name, values in result.items()}


def _gemm_resolver(context):
    return _gemm_space_domains(context)


def _pipeline_resolver(context):
    return {**_gemm_space_domains(context, pipeline=True),
            "DEQUANT_BLOCK_SIZE": _DEQUANT_BLOCKS}


def _dequant_resolver(hardware, operator, shape):
    warps = tuple(w for w in (4, 8) if w * hardware.warp_size <= hardware.max_threads_per_block)
    return {"BLOCK_SIZE": _DEQUANT_BLOCKS, "num_warps": warps or (4, 8)}


def _pattern(seed):
    return "zero_max_codes" if seed == 104729 else "crossword_tail_midpoint" if seed == 130363 else "random"


def _generate_python_inputs(ctx, seed):
    bits = _BITS[ctx.operator.kind]
    m, n, k, _, groups = _input_sizes(ctx.operator, ctx.shape)
    maximum = (1 << bits) - 1
    rng = random.Random(seed)
    variant = _pattern(seed)
    codes = [[((0 if (row + column) % 2 == 0 else maximum) if variant == "zero_max_codes" else
               ((row * 5 + column * 3) & maximum) if variant == "crossword_tail_midpoint" else
               rng.randrange(maximum + 1)) for column in range(n)] for row in range(k)]
    for row in range(k):
        if (row * bits) % 32 + bits > 32:
            codes[row] = [maximum if column % 2 == 0 else 0 for column in range(n)]
    codes[0][0] = 0
    codes[-1][-1] = maximum
    scales = [[_cast(rng.uniform(0.015625, 0.125), ctx.operator.dtype) for _ in range(n)] for _ in range(groups)]
    zero_points = [[((0 if (group + column) % 2 == 0 else maximum) if variant == "zero_max_codes" else
                     (1 << (bits - 1)) if variant == "crossword_tail_midpoint" else
                     rng.randrange(maximum + 1)) for column in range(n)] for group in range(groups)]
    inputs = {"packed": pack_bits(codes, bits), "scales": scales, "zero_points": zero_points}
    if m:
        inputs["a"] = [[_cast(rng.uniform(-0.25, 0.25), ctx.operator.dtype) for _ in range(k)] for _ in range(m)]
    return inputs


def _pack_cuda_codes(ctx, codes, bits, words):
    torch = ctx.torch
    k, n = codes.shape
    positions = torch.arange(k, device=codes.device, dtype=torch.int64) * bits
    indices = (positions // 32)[:, None].expand(k, n)
    shifts = (positions % 32)[:, None]
    packed = torch.zeros((words, n), device=codes.device, dtype=torch.int64)
    packed.scatter_add_(0, indices, (codes << shifts) & 0xFFFFFFFF)
    if bits == 3:
        crossing = (positions % 32) + bits > 32
        high_indices = ((positions[crossing] // 32) + 1)[:, None].expand(-1, n)
        packed.scatter_add_(0, high_indices, codes[crossing] >> (32 - shifts[crossing]))
    # Every logical word is <= 2**32-1; this conversion stores its int32 bit
    # pattern and avoids torch.tensor's out-of-range signed integer conversion.
    return packed.to(torch.int32)


def _generate_cuda_inputs(ctx, seed):
    torch = ctx.torch
    bits = _BITS[ctx.operator.kind]
    m, n, k, words, groups = _input_sizes(ctx.operator, ctx.shape)
    maximum = (1 << bits) - 1
    dtype = getattr(torch, ctx.operator.dtype)
    device = f"cuda:{ctx.device}"
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    columns = torch.arange(n, device=device, dtype=torch.int64)[None, :]
    variant = _pattern(seed)
    packed = torch.empty((words, n), device=device, dtype=torch.int32)
    chunk = _cuda_chunk_rows(n)
    for start in range(0, k, chunk):
        stop = min(start + chunk, k)
        rows = torch.arange(start, stop, device=device, dtype=torch.int64)[:, None]
        if variant == "zero_max_codes":
            codes = ((rows + columns) % 2) * maximum
        elif variant == "crossword_tail_midpoint":
            codes = (rows * 5 + columns * 3) & maximum
        else:
            codes = torch.randint(0, maximum + 1, (stop - start, n), device=device,
                                  dtype=torch.int64, generator=generator)
        crosses = ((rows * bits) % 32) + bits > 32
        codes = torch.where(crosses, torch.where(columns % 2 == 0, maximum, 0), codes)
        if start == 0:
            codes[0, 0] = 0
        if stop == k:
            codes[-1, -1] = maximum
        chunk_words = ((stop - start) * bits + 31) // 32
        word_start = start * bits // 32
        packed[word_start:word_start + chunk_words].copy_(_pack_cuda_codes(ctx, codes, bits, chunk_words))
    scales = (torch.rand((groups, n), device=device, dtype=torch.float32, generator=generator)
              * (0.125 - 0.015625) + 0.015625).to(dtype)
    if variant == "zero_max_codes":
        group_rows = torch.arange(groups, device=device, dtype=torch.int64)[:, None]
        zero_points = (((group_rows + columns) % 2) * maximum).to(torch.int32)
    elif variant == "crossword_tail_midpoint":
        zero_points = torch.full((groups, n), 1 << (bits - 1), device=device, dtype=torch.int32)
    else:
        zero_points = torch.randint(0, maximum + 1, (groups, n), device=device, dtype=torch.int32, generator=generator)
    inputs = {"packed": packed, "scales": scales, "zero_points": zero_points}
    if m:
        inputs["a"] = (torch.randn((m, k), device=device, dtype=torch.float32, generator=generator) * 0.25).to(dtype)
    return inputs


def _generate_inputs(ctx, seed):
    reason = _validate_operator(ctx)
    if reason:
        raise ValueError(reason)
    if ctx.runtime == "python":
        if ctx.operator.dtype not in {"float32", "float64"}:
            raise ValueError("stdlib quantized inputs support float32/float64")
        inputs = _generate_python_inputs(ctx, seed)
    elif ctx.runtime == "cuda":
        reason = _cuda_supports(ctx.hardware, ctx.operator, ctx.shape)
        if reason:
            raise ValueError(reason)
        inputs = _generate_cuda_inputs(ctx, seed)
    else:
        raise ValueError("quantized inputs require python or cuda runtime")
    output_shape = (ctx.shape.get("M"), ctx.shape.get("N")) if _is_gemm(ctx.operator) else (ctx.shape.get("K"), ctx.shape.get("N"))
    ctx.allocate_output("out", output_shape)
    ctx.metadata.update({"input_variant": _pattern(seed), "packing_format": PACKING_FORMAT,
                         "quantization_bits": _BITS[ctx.operator.kind], "group_size": _group_size(ctx.operator)})
    return inputs


def _reference_python(ctx):
    # Reconstruct each code one bit at a time, independently of the helpers and
    # the measured Python/Triton word-shift implementations.
    bits = _BITS[ctx.operator.kind]
    m, n, k, _, _ = _input_sizes(ctx.operator, ctx.shape)
    packed, scales, zeros = (ctx.inputs[name] for name in ("packed", "scales", "zero_points"))
    group_size = _group_size(ctx.operator)
    weights = []
    for row in range(k):
        group = row // group_size
        values = []
        for column in range(n):
            code = 0
            for bit in range(bits):
                word, offset = divmod(row * bits + bit, 32)
                code += ((packed[word][column] >> offset) & 1) * (1 << bit)
            values.append(_cast((code - zeros[group][column]) * scales[group][column], ctx.operator.dtype))
        weights.append(values)
    if not m:
        return {"out": weights}
    a = ctx.inputs["a"]
    return {"out": [[_cast(math.fsum(a[row][index] * weights[index][column] for index in range(k)), ctx.operator.dtype)
                      for column in range(n)] for row in range(m)]}


def _reference_cuda_weights(ctx, start, stop):
    torch = ctx.torch
    bits = _BITS[ctx.operator.kind]
    first_word = start * bits // 32
    last_word = (stop * bits + 31) // 32
    packed = ctx.inputs["packed"][first_word:last_word].to(torch.int64) & 0xFFFFFFFF
    positions = torch.arange(start, stop, device=packed.device, dtype=torch.int64) * bits
    word_indices = positions // 32 - first_word
    shifts = (positions % 32)[:, None]
    codes = packed.index_select(0, word_indices) >> shifts
    if bits == 3:
        high = packed.index_select(0, (word_indices + 1).clamp(max=last_word - first_word - 1))
        high = torch.where(shifts + bits > 32, high << ((32 - shifts) & 31), 0)
        codes = codes | high
    codes = codes & ((1 << bits) - 1)
    groups = torch.arange(start, stop, device=packed.device, dtype=torch.int64) // _group_size(ctx.operator)
    scales = ctx.inputs["scales"].index_select(0, groups).float()
    zeros = ctx.inputs["zero_points"].index_select(0, groups).float()
    return ((codes.float() - zeros) * scales).to(getattr(torch, ctx.operator.dtype))


def _reference_cuda(ctx):
    torch = ctx.torch
    m, n, k, _, _ = _input_sizes(ctx.operator, ctx.shape)
    chunk = _cuda_chunk_rows(n)
    if not m:
        result = torch.empty_like(ctx.outputs["out"])
        for start in range(0, k, chunk):
            stop = min(start + chunk, k)
            result[start:stop].copy_(_reference_cuda_weights(ctx, start, stop))
        return {"out": result}
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        result = torch.zeros((m, n), device=ctx.inputs["a"].device, dtype=torch.float32)
        for start in range(0, k, chunk):
            stop = min(start + chunk, k)
            weights = _reference_cuda_weights(ctx, start, stop)
            result.add_(ctx.inputs["a"][:, start:stop].float() @ weights.float())
        return {"out": result.to(getattr(torch, ctx.operator.dtype))}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _reference(ctx):
    return _reference_cuda(ctx) if ctx.runtime == "cuda" else _reference_python(ctx)


def _build_python(ctx, tile_size=1):
    reason = _python_constraint(KernelConfig({"tile_size": tile_size}), ctx.hardware, ctx.operator, ctx.shape)
    if reason:
        raise ValueError(reason)
    bits = _BITS[ctx.operator.kind]
    m, n, k, _, _ = _input_sizes(ctx.operator, ctx.shape)
    group_size = _group_size(ctx.operator)
    maximum = (1 << bits) - 1
    packed, scales, zeros = (ctx.inputs[name] for name in ("packed", "scales", "zero_points"))
    def launch():
        weights = [[0.0] * n for _ in range(k)]
        for start in range(0, k, tile_size):
            for row in range(start, min(start + tile_size, k)):
                word, shift = divmod(row * bits, 32)
                group = row // group_size
                for column in range(n):
                    code = (packed[word][column] & 0xFFFFFFFF) >> shift
                    if shift + bits > 32:
                        code |= (packed[word + 1][column] & 0xFFFFFFFF) << (32 - shift)
                    weights[row][column] = _cast(((code & maximum) - zeros[group][column]) * scales[group][column], ctx.operator.dtype)
        if not m:
            ctx.outputs["out"] = weights
            return
        a = ctx.inputs["a"]
        output = [[0.0] * n for _ in range(m)]
        for row in range(m):
            for column in range(n):
                total = 0.0
                for start in range(0, k, tile_size):
                    for index in range(start, min(start + tile_size, k)):
                        total += a[row][index] * weights[index][column]
                output[row][column] = _cast(total, ctx.operator.dtype)
        ctx.outputs["out"] = output
    return launch


def _check_compiled_resources(ctx, compiled, warps, label):
    ctx.record_compiled_kernel(label, compiled)


def _build_triton_dequantize(ctx, BLOCK_SIZE, num_warps):
    reason = _dequant_constraint(KernelConfig({"BLOCK_SIZE": BLOCK_SIZE, "num_warps": num_warps}),
                                 ctx.hardware, ctx.operator, ctx.shape)
    if reason:
        raise ValueError(reason)
    import triton
    from ..backends._triton_quantized import dequantize_kernel
    k, n = ctx.shape.get("K"), ctx.shape.get("N")
    grid = (triton.cdiv(k * n, BLOCK_SIZE),)
    args = (ctx.inputs["packed"], ctx.inputs["scales"], ctx.inputs["zero_points"], ctx.outputs["out"], k, n)
    parameters = {"BITS": _BITS[ctx.operator.kind], "GROUP_SIZE": _group_size(ctx.operator),
                  "BLOCK_SIZE": BLOCK_SIZE, "num_warps": num_warps}
    compiled = dequantize_kernel.warmup(*args, grid=grid, **parameters)
    _check_compiled_resources(ctx, compiled, num_warps, "dequantize")
    def launch():
        dequantize_kernel[grid](*args, **parameters)
    return launch


def _gemm_destination(ctx, split):
    if split == 1:
        return ctx.outputs["out"]
    # Every partition writes every logical [M,N] value on every invocation.
    # No atomic update, stale read, or untimed zero-initialization is involved.
    return ctx.allocate_workspace("split_k_partials", (split, ctx.shape.get("M"), ctx.shape.get("N")),
                                  dtype="float32", fill=None)


def _split_k_reduction(ctx, partials, split, num_warps):
    import triton
    from ..backends._triton_quantized import split_k_reduce_kernel
    elements = ctx.shape.get("M") * ctx.shape.get("N")
    grid = (triton.cdiv(elements, 256),)
    args = (partials, ctx.outputs["out"], elements)
    parameters = {"SPLIT_K": split, "BLOCK_SIZE": 256, "num_warps": num_warps}
    compiled = split_k_reduce_kernel.warmup(*args, grid=grid, **parameters)
    _check_compiled_resources(ctx, compiled, num_warps, "split_k_reduce")
    def reduce():
        split_k_reduce_kernel[grid](*args, **parameters)
    return ExecutionStep("split_k_reduce", reduce)


def _build_triton_gemm(ctx, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, SPLIT_K):
    knobs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
             "num_warps": num_warps, "num_stages": num_stages, "GROUP_M": GROUP_M, "SPLIT_K": SPLIT_K}
    context = SpaceContext(ctx.hardware, ctx.operator, ctx.shape,
                           workspace_memory_limit_bytes=ctx.workspace_memory_limit_bytes)
    reason = _gemm_constraint(KernelConfig(knobs), context)
    if reason:
        raise ValueError(reason)
    import triton
    from ..backends._triton_quantized import quantized_gemm_kernel
    m, n, k = (ctx.shape.get(name) for name in ("M", "N", "K"))
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N), SPLIT_K)
    destination = _gemm_destination(ctx, SPLIT_K)
    args = (ctx.inputs["a"], ctx.inputs["packed"], ctx.inputs["scales"], ctx.inputs["zero_points"],
            destination, m, n, k)
    parameters = {"BITS": _BITS[ctx.operator.kind], "GROUP_SIZE": _group_size(ctx.operator),
                  "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
                  "num_warps": num_warps, "num_stages": num_stages, "GROUP_M": GROUP_M, "SPLIT_K": SPLIT_K}
    compiled = quantized_gemm_kernel.warmup(*args, grid=grid, **parameters)
    _check_compiled_resources(ctx, compiled, num_warps, "fused_dequantize_gemm")
    def launch():
        quantized_gemm_kernel[grid](*args, **parameters)
    steps = (ExecutionStep("fused_dequantize_gemm", launch),)
    if SPLIT_K > 1:
        steps += (_split_k_reduction(ctx, destination, SPLIT_K, num_warps),)
    return ExecutionPlan(steps)


def _build_triton_pipeline(ctx, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages,
                            GROUP_M, SPLIT_K, DEQUANT_BLOCK_SIZE):
    knobs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
             "num_warps": num_warps, "num_stages": num_stages, "GROUP_M": GROUP_M, "SPLIT_K": SPLIT_K,
             "DEQUANT_BLOCK_SIZE": DEQUANT_BLOCK_SIZE}
    context = SpaceContext(ctx.hardware, ctx.operator, ctx.shape,
                           workspace_memory_limit_bytes=ctx.workspace_memory_limit_bytes)
    reason = _pipeline_constraint(KernelConfig(knobs), context)
    if reason:
        raise ValueError(reason)
    import triton
    from ..backends._triton_quantized import dequantize_kernel, dense_gemm_kernel
    m, n, k = (ctx.shape.get(name) for name in ("M", "N", "K"))
    weights = ctx.allocate_workspace("dequantized_weights", (k, n), fill=None)
    destination = _gemm_destination(ctx, SPLIT_K)
    dequant_grid = (triton.cdiv(k * n, DEQUANT_BLOCK_SIZE),)
    dequant_args = (ctx.inputs["packed"], ctx.inputs["scales"], ctx.inputs["zero_points"], weights, k, n)
    dequant_parameters = {"BITS": _BITS[ctx.operator.kind], "GROUP_SIZE": _group_size(ctx.operator),
                          "BLOCK_SIZE": DEQUANT_BLOCK_SIZE, "num_warps": num_warps}
    gemm_grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N), SPLIT_K)
    gemm_args = (ctx.inputs["a"], weights, destination, m, n, k)
    gemm_parameters = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
                       "num_warps": num_warps, "num_stages": num_stages, "GROUP_M": GROUP_M, "SPLIT_K": SPLIT_K}
    compiled_dequant = dequantize_kernel.warmup(*dequant_args, grid=dequant_grid, **dequant_parameters)
    _check_compiled_resources(ctx, compiled_dequant, num_warps, "dequantize")
    compiled_gemm = dense_gemm_kernel.warmup(*gemm_args, grid=gemm_grid, **gemm_parameters)
    _check_compiled_resources(ctx, compiled_gemm, num_warps, "gemm")
    def dequantize():
        dequantize_kernel[dequant_grid](*dequant_args, **dequant_parameters)
    def gemm():
        dense_gemm_kernel[gemm_grid](*gemm_args, **gemm_parameters)
    steps = (ExecutionStep("dequantize", dequantize), ExecutionStep("gemm", gemm))
    if SPLIT_K > 1:
        steps += (_split_k_reduction(ctx, destination, SPLIT_K, num_warps),)
    return ExecutionPlan(steps)


def register_assets(catalog):
    """Register six semantics, six CPU kernels and nine real CUDA kernels."""
    python_space = SpaceDefinition({"tile_size": ParameterSpec((1, 4, 16))},
                                   constraints=(_python_constraint,), name="python_quantized_tiles_v1")
    dequant_space = SpaceDefinition({"BLOCK_SIZE": ParameterSpec(_DEQUANT_BLOCKS,
                                     description="Standalone flattened K*N dequantization tile", unit="elements"),
                                    "num_warps": ParameterSpec((4, 8), description="Warps per dequantization program")},
                                    constraints=(_dequant_constraint,), name="triton_packed_dequantize_v1", resolver=_dequant_resolver)
    gemm_parameters = {name: ParameterSpec(values, description=_KNOB_DESCRIPTIONS[name])
                       for name, values in _GEMM_DOMAINS.items()}
    gemm_space = SpaceDefinition(gemm_parameters, constraints=(_gemm_constraint,),
                                 name="triton_fused_quantized_gemm_v1", resolver=_gemm_resolver)
    pipeline_space = SpaceDefinition({**gemm_parameters, "DEQUANT_BLOCK_SIZE": ParameterSpec(_DEQUANT_BLOCKS,
                                      description="Materialized dequantization tile independent of GEMM BLOCK_K", unit="elements")},
                                     constraints=(_pipeline_constraint,), name="triton_dequantize_gemm_pipeline_v1", resolver=_pipeline_resolver)
    shared_sources = ("../quantization.py",)
    gpu_sources = (*shared_sources, "../backends/_triton_quantized.py")
    for name, bits in _BITS.items():
        gemm = name.endswith("a16_gemm")
        shapes = (ShapeSpec({"M": 3, "N": 5, "K": 35}), ShapeSpec({"M": 1, "N": 1, "K": 1}),
                  ShapeSpec({"M": 2, "N": 7, "K": 65})) if gemm else (
                  ShapeSpec({"K": 35, "N": 5}), ShapeSpec({"K": 1, "N": 1}), ShapeSpec({"K": 65, "N": 7}))
        operator = OperatorDefinition(
            id=name, version=_VERSION, shape_dims=("M", "N", "K") if gemm else ("K", "N"),
            dtypes=("float16", "bfloat16", "float32", "float64"), generate_inputs=_generate_inputs,
            reference=_reference, tolerances=_TOLERANCES, validation_shapes=shapes,
            parameters={"group_size": 32}, validate=_validate_operator, source_files=shared_sources,
            metadata={"description": "Groupwise unsigned affine packed weight-only GEMM" if gemm else "Standalone groupwise unsigned affine packed dequantization",
                      "proposal_family": "quantized",
                      "quantization_bits": bits, "packing_format": PACKING_FORMAT,
                      "packing_layout": "int32[ceil(K*bits/32),N], LSB-first contiguous K stream per column; W3 straddles words",
                      "dequantization": "(code-zero_points[k//group_size,n])*scales[k//group_size,n]",
                      "group_shape": "[ceil(K/group_size),N]; partial last group is allowed",
                      "zero_points": "int32 unsigned codes; midpoint 2**(bits-1) supports symmetric convention",
                      "checkpoint_compatibility": "Explicit canonical format; GPTQ/AWQ conversion is not automatic",
                      "a16_rounding": "dequantized weights round to the input/output dtype before GEMM",
                      "production_shapes": [{"M": 1, "N": 4096, "K": 4096}, {"M": 128, "N": 4096, "K": 4096},
                                            {"M": 512, "N": 4096, "K": 11008}] if gemm else [{"K": 4096, "N": 4096}, {"K": 11008, "N": 4096}],
                      "validation_seeds": [0, 104729, 130363],
                      "input_names": ["a", "packed", "scales", "zero_points"] if gemm else ["packed", "scales", "zero_points"],
                      "output_names": ["out"]})
        catalog.register(operator)
        catalog.register(KernelDefinition(
            id=f"python/{name}", version=_VERSION, operator=operator.ref, build=_build_python,
            space=python_space, runtime="python", dtypes=("float32", "float64"), architectures=("cpu",),
            source_files=shared_sources,
            metadata={"description": "Executable small CPU bitstream implementation with independent bitwise reference"}))
        catalog.register(KernelDefinition(
            id=f"triton/{name}", version=_VERSION, operator=operator.ref,
            build=_build_triton_gemm if gemm else _build_triton_dequantize,
            space=gemm_space if gemm else dequant_space, runtime="cuda",
            dtypes=("float16", "bfloat16"), architectures=("sm_*",), supports=_cuda_supports,
            source_files=gpu_sources,
            metadata={"requires_triton": True, "minimum_architecture": "sm_80", "packing_format": PACKING_FORMAT,
                      "runtime_compatibility_source": "https://github.com/triton-lang/triton#compatibility",
                      "description": "Fused unpack/affine-dequantize/A16 dot with FP32 accumulation" if gemm else "Standalone Triton bitstream unpack and affine dequantization",
                      "design_sources": _DESIGN_SOURCES,
                      "optimization_design": "N-contiguous packed loads; aligned group scale broadcast; grouped M program order; FP32 Split-K partials and timed reduction" if gemm else "N-contiguous flattened unpack tiles",
                      "split_k_initialization": "No initialization: all FP32 partials are overwritten in every invocation",
                      "default_workspace_memory_limit_bytes": _DEFAULT_WORKSPACE_BYTES,
                      "resource_validation": "Conservative staged A16 tile estimate plus compiled shared-memory/register limits"}))
        if gemm:
            catalog.register(KernelDefinition(
                id=f"triton_pipeline/{name}", version=_VERSION, operator=operator.ref, build=_build_triton_pipeline,
                space=pipeline_space, runtime="cuda", dtypes=("float16", "bfloat16"),
                architectures=("sm_*",), supports=_cuda_supports,
                source_files=gpu_sources,
                metadata={"requires_triton": True, "minimum_architecture": "sm_80", "packing_format": PACKING_FORMAT,
                          "runtime_compatibility_source": "https://github.com/triton-lang/triton#compatibility",
                          "description": "Timed dequantization to declared workspace, then grouped dense GEMM and optional Split-K reduction",
                          "design_sources": _DESIGN_SOURCES,
                          "execution_steps": ["dequantize", "gemm", "split_k_reduce (only SPLIT_K>1)"],
                          "workspace": "K*N input-dtype weights plus optional SPLIT_K*M*N FP32 partials; bounded by SpaceContext execution budget",
                          "default_workspace_memory_limit_bytes": _DEFAULT_WORKSPACE_BYTES}))


__all__ = ["register_assets", "pack_bits", "unpack_bits", "dequantize_packed"]
