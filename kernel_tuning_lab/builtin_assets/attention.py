"""Inference SDPA assets with independent, bounded CPU and CUDA references.

The public tensor layout is Q[B, Hq, Sq, D], K/V[B, Hkv, Sk, D].
Grouped query heads map to KV heads by ``hq // (Hq // Hkv)``.  Bottom-right
causality permits key j for query i exactly when j <= i + Sk - Sq; rows with
no permitted keys produce zeros.  Importing this module needs no GPU SDK.
"""

from __future__ import annotations

import math
import random
import struct
from itertools import product
from typing import TYPE_CHECKING

from ..torch_context import strict_fp32_reference
from ..sdk import (HardwareSpec, KernelConfig, KernelDefinition, OperatorDefinition,
                   OperatorSpec, ParameterSpec, ShapeSpec, SpaceDefinition)

if TYPE_CHECKING:
    from ..sdk import ExecutionContext


_DIMS = ("Batch", "Heads", "KVHeads", "SeqQ", "SeqKV", "D")
_DEFAULTS = {"causal": True, "causal_alignment": "bottom_right",
             "softmax_scale": None, "dropout": 0.0}
_CPU_MAX_ELEMENTS = 262_144
_CPU_MAX_DOT_WORK = 8_000_000
_REFERENCE_QUERY_CHUNK = 128
_REFERENCE_SCORE_ELEMENTS = 1_048_576
_CUDA_MAX_LIVE_BYTES = 4 * 1024**3
_FP32_MAX = 3.4028234663852886e38
_LOG2_E = 1.4426950408889634
_TRITON_DOMAINS = {"BLOCK_M": (16, 32, 64, 128), "BLOCK_N": (16, 32, 64, 128),
                   "num_warps": (4, 8), "num_stages": (1, 2, 3),
                   "USE_EXP2": (True, False), "PRE_LOAD_V": (False, True)}
_TRITON_KNOB_DESCRIPTIONS = {
    "BLOCK_M": "Query rows owned by one program; Q stays resident across K/V tiles",
    "BLOCK_N": "K/V tokens per online-softmax iteration",
    "num_warps": "GPU warps assigned to each Q tile",
    "num_stages": "Compiler and tl.range pipeline depth for the K/V tile loop",
    "USE_EXP2": "Use log2-scaled logits with exp2 instead of natural exponentiation",
    "PRE_LOAD_V": "Load the V tile before computing scores, trading registers for load overlap",
}


def _semantics_reason(operator: OperatorSpec, shape: ShapeSpec) -> str | None:
    if operator.kind != "attention" or set(shape.dims) != set(_DIMS):
        return "attention requires exactly Batch/Heads/KVHeads/SeqQ/SeqKV/D dimensions"
    if shape.get("Heads") % shape.get("KVHeads"):
        return "attention requires Heads to be divisible by KVHeads"
    if not 1 <= shape.get("D") <= 256:
        return "attention supports head dimensions D from 1 through 256"
    unknown = operator.parameters.keys() - _DEFAULTS.keys()
    if unknown:
        return "unsupported attention parameters: " + ", ".join(sorted(unknown))
    parameters = {**_DEFAULTS, **operator.parameters}
    if type(parameters["causal"]) is not bool:
        return "attention causal must be a boolean"
    if parameters["causal_alignment"] not in ("bottom_right", "top_left"):
        return "attention causal_alignment must be bottom_right or top_left"
    scale = parameters["softmax_scale"]
    if scale is not None:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            return "attention softmax_scale must be None or a finite real number"
        try:
            if not math.isfinite(scale):
                return "attention softmax_scale must be None or a finite real number"
        except OverflowError:
            return "attention softmax_scale must be representable as a finite real number"
    dropout = parameters["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or dropout != 0:
        return "attention is inference-only and requires dropout=0"
    return None


def _sizes(shape: ShapeSpec) -> tuple[int, int]:
    b, hq, hkv, sq, sk, d = (shape.get(name) for name in _DIMS)
    return b * hq * sq * d, b * hkv * sk * d


def _python_supports(hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec) -> str | None:
    reason = _semantics_reason(operator, shape)
    if reason:
        return reason
    if operator.dtype not in ("float32", "float64"):
        return "Python attention supports float32/float64 only"
    q_elements, kv_elements = _sizes(shape)
    # Include outputs and independent reference copies, not just logical inputs.
    if 4 * q_elements + 2 * kv_elements > _CPU_MAX_ELEMENTS:
        return "Python attention exceeds the 262,144 live logical element limit; use CUDA"
    work = math.prod(shape.get(name) for name in ("Batch", "Heads", "SeqQ", "SeqKV", "D"))
    if work > _CPU_MAX_DOT_WORK:
        return "Python attention exceeds the 8,000,000 dot-product work limit; use CUDA"
    return None


def _reference_chunk_rows(seq_kv: int) -> int:
    return max(1, min(_REFERENCE_QUERY_CHUNK, _REFERENCE_SCORE_ELEMENTS // seq_kv))


def _exp2_can_prescale(scale: float, dtype: str = "float16", head_dimension: int = 256) -> bool:
    """Keep the coefficient and finite A16 QK scores within FP32 range.

    A finite coefficient alone is insufficient: its product with QK may still
    overflow while the natural-domain logit is finite.  FP16 has a bounded QK
    magnitude (with extra rounding headroom); finite BF16 QK can fill FP32.
    """
    coefficient = abs(scale * _LOG2_E)
    dot_bound = 2.0 * head_dimension * 65504.0**2 if dtype == "float16" else _FP32_MAX
    return coefficient <= _FP32_MAX / dot_bound


def _cuda_supports(hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec) -> str | None:
    reason = _semantics_reason(operator, shape)
    if reason:
        return reason
    if operator.dtype not in ("float16", "bfloat16"):
        return "Triton attention supports float16/bfloat16 only"
    scale = operator.parameters.get("softmax_scale")
    if scale is not None and abs(scale) > _FP32_MAX:
        return "CUDA attention softmax_scale must lie within the finite FP32 range"
    architecture = hardware.architecture
    if not architecture.startswith("sm_") or not architecture[3:].isdigit():
        return "Triton attention requires an NVIDIA sm_* architecture"
    capability = int(architecture[3:])
    if capability < 80:
        return "Triton attention requires sm_80+ for the current Triton runtime contract"
    sk, d = shape.get("SeqKV"), shape.get("D")
    if sk > _REFERENCE_SCORE_ELEMENTS:
        return "attention SeqKV exceeds the bounded-reference limit of 1,048,576"
    q_elements, kv_elements = _sizes(shape)
    scores = min(shape.get("SeqQ"), _reference_chunk_rows(sk)) * sk
    # Cover generated inputs, output/reference clones, FP32 K/V for one head,
    # query chunks, score/softmax/mask temporaries and key position indices.
    estimated = 16 * (q_elements + kv_elements) + 8 * sk * d + 32 * scores + 8 * sk
    limit = _CUDA_MAX_LIVE_BYTES
    if hardware.memory_bytes is not None:
        limit = min(limit, int(hardware.memory_bytes) // 2)
    if estimated > limit:
        return f"attention conservative input/reference memory estimate {estimated} exceeds {limit} bytes"
    return None


def _validate_attention(ctx: ExecutionContext) -> str | None:
    reason = _semantics_reason(ctx.operator, ctx.shape)
    if reason:
        return reason
    if ctx.runtime == "python":
        return _python_supports(ctx.hardware, ctx.operator, ctx.shape)
    if ctx.runtime == "cuda":
        return _cuda_supports(ctx.hardware, ctx.operator, ctx.shape)
    return "attention supports python and cuda runtimes only"


def _settings(operator: OperatorSpec, shape: ShapeSpec) -> tuple[bool, int, float]:
    parameters = {**_DEFAULTS, **operator.parameters}
    offset = shape.get("SeqKV") - shape.get("SeqQ") if parameters["causal_alignment"] == "bottom_right" else 0
    scale = parameters["softmax_scale"]
    return parameters["causal"], offset, 1.0 / math.sqrt(shape.get("D")) if scale is None else float(scale)


def _cast(value: float, dtype: str) -> float:
    return struct.unpack("f", struct.pack("f", value))[0] if dtype == "float32" else float(value)


def _cpu_tensor(batch: int, heads: int, rows: int, width: int, rng: random.Random,
                dtype: str, variant: str, offset: float = 0.0):
    return [[[[_cast(offset + (0.125 if variant == "constant" else rng.uniform(-0.25, 0.25)), dtype)
               for _ in range(width)] for _ in range(rows)] for _ in range(heads)] for _ in range(batch)]


def _generate_inputs(ctx: ExecutionContext, seed: int):
    reason = _validate_attention(ctx)
    if reason:
        raise ValueError(reason)
    b, hq, hkv, sq, sk, d = (ctx.shape.get(name) for name in _DIMS)
    variant = "constant" if seed == 104729 else "offset" if seed == 130363 else "random"
    if ctx.runtime == "python":
        rng = random.Random(seed)
        q = _cpu_tensor(b, hq, sq, d, rng, ctx.operator.dtype, variant, 8.0 if variant == "offset" else 0.0)
        k = _cpu_tensor(b, hkv, sk, d, rng, ctx.operator.dtype, variant)
        v = _cpu_tensor(b, hkv, sk, d, rng, ctx.operator.dtype, variant, -0.5 if variant == "offset" else 0.0)
    else:
        torch = ctx.torch
        dtype, device = getattr(torch, ctx.operator.dtype), f"cuda:{ctx.device}"
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        def make(dimensions, offset=0.0):
            if variant == "constant":
                return torch.full(dimensions, 0.125, dtype=dtype, device=device)
            values = torch.randn(dimensions, dtype=dtype, device=device, generator=generator)
            values.mul_(0.25)
            if offset:
                values.add_(offset)
            return values

        q = make((b, hq, sq, d), 8.0 if variant == "offset" else 0.0)
        k = make((b, hkv, sk, d))
        v = make((b, hkv, sk, d), -0.5 if variant == "offset" else 0.0)
    ctx.allocate_output("out", (b, hq, sq, d))
    ctx.metadata["input_variant"] = variant
    return {"q": q, "k": k, "v": v}


def _reference_cpu(ctx: ExecutionContext):
    q, k, v = (ctx.inputs[name] for name in ("q", "k", "v"))
    b, hq, hkv, sq, sk, d = (ctx.shape.get(name) for name in _DIMS)
    causal, offset, scale = _settings(ctx.operator, ctx.shape)
    result = []
    for batch in range(b):
        heads = []
        for head in range(hq):
            kv_head = head // (hq // hkv)
            rows = []
            for query in range(sq):
                count = max(0, min(sk, query + offset + 1)) if causal else sk
                if not count:
                    rows.append([0.0] * d)
                    continue
                logits = [scale * math.fsum(x * y for x, y in zip(q[batch][head][query], key))
                          for key in k[batch][kv_head][:count]]
                maximum = max(logits)
                exponents = [math.exp(value - maximum) for value in logits]
                denominator = math.fsum(exponents)
                probabilities = [value / denominator for value in exponents]
                rows.append([_cast(math.fsum(probabilities[key] * v[batch][kv_head][key][column]
                                            for key in range(count)), ctx.operator.dtype) for column in range(d)])
            heads.append(rows)
        result.append(heads)
    return {"out": result}


def _reference_cuda(ctx: ExecutionContext):
    torch = ctx.torch
    q, k, v = (ctx.inputs[name] for name in ("q", "k", "v"))
    b, hq, hkv, sq, sk, _ = (ctx.shape.get(name) for name in _DIMS)
    causal, offset, scale = _settings(ctx.operator, ctx.shape)
    chunk_rows = _reference_chunk_rows(sk)
    ctx.metadata["reference_query_chunk"] = chunk_rows
    result = torch.empty_like(q)
    with strict_fp32_reference(torch):
        with torch.no_grad():
            keys = torch.arange(sk, device=q.device)
            # Never repeat K/V across GQA heads or form a full Sq-by-Sk matrix.
            for batch in range(b):
                for kv_head in range(hkv):
                    key = k[batch, kv_head].to(torch.float32).transpose(0, 1)
                    value = v[batch, kv_head].to(torch.float32)
                    for head in range(kv_head * (hq // hkv), (kv_head + 1) * (hq // hkv)):
                        for begin in range(0, sq, chunk_rows):
                            end = min(begin + chunk_rows, sq)
                            scores = q[batch, head, begin:end].to(torch.float32) @ key
                            scores.mul_(scale)
                            if causal:
                                queries = torch.arange(begin, end, device=q.device)
                                allowed = keys[None, :] <= queries[:, None] + offset
                                has_keys = allowed.any(dim=-1)
                                scores.masked_fill_(~allowed, -math.inf)
                                # Softmax of a fully masked row is undefined;
                                # use a finite dummy row, then explicitly zero it.
                                scores.masked_fill_(~has_keys[:, None], 0.0)
                            probabilities = torch.softmax(scores, dim=-1)
                            output = probabilities @ value
                            if causal:
                                output.masked_fill_(~has_keys[:, None], 0.0)
                            result[batch, head, begin:end].copy_(output)
    return {"out": result}


def _reference(ctx: ExecutionContext):
    return _reference_cuda(ctx) if ctx.runtime == "cuda" else _reference_cpu(ctx)


def _build_python(ctx: ExecutionContext, tile_size: int = 1):
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("attention tile_size must be a positive integer")
    reason = _python_supports(ctx.hardware, ctx.operator, ctx.shape)
    if reason:
        raise ValueError(reason)
    q, k, v = (ctx.inputs[name] for name in ("q", "k", "v"))
    output = ctx.outputs["out"]
    b, hq, hkv, sq, sk, d = (ctx.shape.get(name) for name in _DIMS)
    causal, offset, scale = _settings(ctx.operator, ctx.shape)

    def launch():
        for batch in range(b):
            for head in range(hq):
                kv_head = head // (hq // hkv)
                for begin in range(0, sq, tile_size):
                    for query in range(begin, min(begin + tile_size, sq)):
                        count = max(0, min(sk, query + offset + 1)) if causal else sk
                        if not count:
                            output[batch][head][query][:] = [0.0] * d
                            continue
                        # Ordinary summation is independent of the fsum reference.
                        scores = [scale * sum(q[batch][head][query][column] * k[batch][kv_head][key][column]
                                              for column in range(d)) for key in range(count)]
                        maximum = max(scores)
                        weights = [math.exp(score - maximum) for score in scores]
                        denominator = sum(weights)
                        for column in range(d):
                            numerator = sum(weights[key] * v[batch][kv_head][key][column] for key in range(count))
                            output[batch][head][query][column] = _cast(numerator / denominator, ctx.operator.dtype)

    return launch


def _triton_constraint(config: KernelConfig, hardware: HardwareSpec,
                       operator: OperatorSpec, shape: ShapeSpec) -> str | None:
    reason = _cuda_supports(hardware, operator, shape)
    if reason:
        return reason
    bm, bn, warps, stages = (config.get(name) for name in ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages"))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (bm, bn, warps, stages)):
        return "attention tile/warp/stage knobs must be integers"
    if bm not in _TRITON_DOMAINS["BLOCK_M"] or bn not in _TRITON_DOMAINS["BLOCK_N"] or warps not in (4, 8) or stages not in (1, 2, 3):
        return "attention knobs are outside the supported tile/warp/stage domains"
    if type(config.get("USE_EXP2")) is not bool or type(config.get("PRE_LOAD_V")) is not bool:
        return "attention USE_EXP2 and PRE_LOAD_V knobs must be booleans"
    grid_x = shape.get("Batch") * shape.get("Heads") * ((shape.get("SeqQ") + bm - 1) // bm)
    if grid_x >= 2**31:
        return "attention flattened launch grid.x must be smaller than 2**31"
    threads = warps * hardware.warp_size
    if threads > hardware.max_threads_per_block:
        return "attention warp count exceeds the hardware thread limit"
    bd = max(16, 1 << (shape.get("D") - 1).bit_length())
    shared = (bm * bd + 2 * bn * bd) * 2 * stages + bm * bn * 4
    if hardware.smem_per_block and shared > hardware.smem_per_block:
        return f"attention conservative shared-memory estimate {shared} exceeds {hardware.smem_per_block} bytes"
    # Q/K/V fragments use packed 16-bit operands; scores and the accumulated
    # output use FP32.  Include fixed per-thread headroom for masks/statistics.
    registers = (3 * bm * bd) // 2 + bm * bn + (2 if config.get("PRE_LOAD_V") else 1) * bn * bd // 2 + 32 * threads
    if (registers + threads - 1) // threads > 255:
        return "attention conservative register estimate exceeds 255 registers per thread"
    if hardware.registers_per_sm and registers > hardware.registers_per_sm:
        return "attention conservative block register estimate exceeds the SM register capacity"
    return None


def _space_resolver(hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec):
    """Shrink each knob to values appearing in some resource-legal combination.

    The space is small (384 combinations); this setup-only enumeration does no
    GPU work.  Correlated resource limits remain authoritative constraints.
    A wholly unsupported case retains its domains and is rejected by supports.
    """
    if _cuda_supports(hardware, operator, shape):
        return dict(_TRITON_DOMAINS)
    kept = {name: set() for name in _TRITON_DOMAINS}
    for values in product(*_TRITON_DOMAINS.values()):
        config = KernelConfig(dict(zip(_TRITON_DOMAINS, values)))
        if _triton_constraint(config, hardware, operator, shape) is None:
            for name, value in config.params.items():
                kept[name].add(value)
    return {name: tuple(value for value in domain if value in kept[name]) if kept[name] else domain
            for name, domain in _TRITON_DOMAINS.items()}


def _build_triton(ctx: ExecutionContext, BLOCK_M: int, BLOCK_N: int,
                  num_warps: int, num_stages: int, USE_EXP2: bool, PRE_LOAD_V: bool):
    knobs = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "num_warps": num_warps,
             "num_stages": num_stages, "USE_EXP2": USE_EXP2, "PRE_LOAD_V": PRE_LOAD_V}
    reason = _triton_constraint(KernelConfig(knobs), ctx.hardware, ctx.operator, ctx.shape)
    if reason:
        raise ValueError(reason)
    import triton
    from ..backends._triton_attention import attention_kernel

    b, hq, hkv, sq, sk, d = (ctx.shape.get(name) for name in _DIMS)
    causal, offset, scale = _settings(ctx.operator, ctx.shape)
    args = (ctx.inputs["q"], ctx.inputs["k"], ctx.inputs["v"], ctx.outputs["out"])
    parameters = {**knobs, "HEADS": hq, "KV_HEADS": hkv, "SEQ_Q": sq,
                  "SEQ_KV": sk, "D": d, "SOFTMAX_SCALE": scale, "CAUSAL": causal,
                  "CAUSAL_OFFSET": offset, "BLOCK_D": max(16, triton.next_power_of_2(d)),
                  "PIPELINE_STAGES": num_stages,
                  "EXP2_PRE_SCALE": _exp2_can_prescale(scale, ctx.operator.dtype, d)}
    grid = (triton.cdiv(sq, BLOCK_M) * b * hq,)
    compiled = attention_kernel.warmup(*args, grid=grid, **parameters)
    ctx.record_compiled_kernel("attention", compiled)

    def launch():
        attention_kernel[grid](*args, **parameters)

    return launch


def register_assets(catalog) -> None:
    """Register one semantic asset and actual CPU/Triton implementations."""
    operator = OperatorDefinition(
        id="attention", version="1.0.0", shape_dims=_DIMS,
        dtypes=("float16", "bfloat16", "float32", "float64"),
        generate_inputs=_generate_inputs, reference=_reference, validate=_validate_attention,
        source_files=("../torch_context.py",),
        tolerances={"float16": (0.02, 0.02), "bfloat16": (0.03, 0.03),
                    "float32": (1e-5, 1e-4), "float64": (1e-10, 1e-9)},
        parameters=_DEFAULTS,
        validation_shapes=tuple(ShapeSpec(dict(zip(_DIMS, values))) for values in (
            (1, 2, 2, 3, 5, 7), (1, 4, 2, 5, 7, 17), (2, 4, 1, 1, 9, 33),
            (1, 2, 1, 7, 3, 5), (1, 1, 1, 1, 1, 1), (1, 2, 1, 17, 19, 65))),
        metadata={"description": "Contiguous inference SDPA for MHA/GQA/MQA, prefill and decode",
                  "proposal_family": "attention",
                  "input_names": ["q", "k", "v"], "output_names": ["out"],
                  "validation_seeds": [0, 104729, 130363], "head_dimension_range": [1, 256],
                  "fully_masked_rows": "zeros", "causal_bottom_right_offset": "SeqKV-SeqQ",
                  "reference": "stdlib math.fsum; CUDA FP32 matmul/softmax with explicit mask and bounded query chunks",
                  "reference_max_score_elements": _REFERENCE_SCORE_ELEMENTS,
                  "python_max_live_elements": _CPU_MAX_ELEMENTS, "python_max_dot_work": _CPU_MAX_DOT_WORK,
                  "cuda_max_estimated_live_bytes": _CUDA_MAX_LIVE_BYTES,
                  "cuda_softmax_scale_max_abs": _FP32_MAX,
                  "production_shapes": [
                      {"Batch": 1, "Heads": 32, "KVHeads": 8, "SeqQ": 2048, "SeqKV": 2048, "D": 128},
                      {"Batch": 1, "Heads": 32, "KVHeads": 8, "SeqQ": 16384, "SeqKV": 16384, "D": 128},
                      {"Batch": 1, "Heads": 32, "KVHeads": 8, "SeqQ": 1, "SeqKV": 16384, "D": 128}],
                  "scope": "contiguous, non-paged, inference forward only; no dropout, backward or attention bias",
                  "primary_sources": ["https://arxiv.org/abs/2307.08691",
                                      "https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html",
                                      "https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton_og.py"],
                  "validation_parameter_cases": [
                      {"name": "bottom_right_gqa", "parameters": {"causal": True, "causal_alignment": "bottom_right"}},
                      {"name": "top_left_gqa", "parameters": {"causal": True, "causal_alignment": "top_left"}},
                      {"name": "noncausal", "parameters": {"causal": False}},
                      {"name": "custom_scale", "parameters": {"softmax_scale": 0.25}},
                      {"name": "zero_scale", "parameters": {"softmax_scale": 0.0}},
                      {"name": "negative_scale", "parameters": {"softmax_scale": -0.5}}]})
    catalog.register(operator)
    catalog.register(KernelDefinition(
        id="python/attention", version="1.0.0", operator=operator.ref, build=_build_python,
        runtime="python", dtypes=("float32", "float64"), architectures=("cpu",), supports=_python_supports,
        space=SpaceDefinition({"tile_size": ParameterSpec((1, 4, 16))}, name="python_attention_v1"),
        metadata={"description": "Small stdlib SDPA implementation for acceptance; not a GPU performance baseline"}))
    catalog.register(KernelDefinition(
        id="triton/attention", version="1.0.0", operator=operator.ref, build=_build_triton,
        runtime="cuda", dtypes=("float16", "bfloat16"), architectures=("sm_*",), supports=_cuda_supports,
        space=SpaceDefinition({name: ParameterSpec(values, description=_TRITON_KNOB_DESCRIPTIONS[name])
                               for name, values in _TRITON_DOMAINS.items()},
                              constraints=(_triton_constraint,), name="triton_attention_v1", resolver=_space_resolver),
        source_files=("../backends/_triton_attention.py",),
        metadata={"requires_triton": True, "description": "Fused tiled SDPA with FP32 online softmax accumulation",
                  "tensor_layout": "contiguous B,H,S,D", "minimum_architecture": "sm_80",
                  "launch_grid": "one-dimensional query-tiles times batch-heads",
                  "optimizations": ["causal KV tile upper-bound pruning", "online softmax without quadratic storage",
                                    "selectable base-2 exponentiation", "selectable V preload", "pipelined K/V tile loop"],
                  "exp2_scaling": "Pre-scale with FP32 headroom for A16 QK scores; otherwise center natural logits before base-2 conversion",
                  "resource_model": "conservative shared-memory and register estimates; actual compilation is authoritative"}))


__all__ = ["register_assets"]
