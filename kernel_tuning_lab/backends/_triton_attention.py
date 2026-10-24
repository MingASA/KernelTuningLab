"""Fused inference SDPA; imported only after optional GPU runtime discovery.

Each program owns one Q tile and one (batch, query-head), reads K/V tiles, and
maintains FP32 softmax statistics and output without storing a score matrix.
"""

import triton
import triton.language as tl


@triton.jit
def attention_kernel(Q, K, V, Out,
                     HEADS: tl.constexpr, KV_HEADS: tl.constexpr,
                     SEQ_Q: tl.constexpr, SEQ_KV: tl.constexpr, D: tl.constexpr,
                     SOFTMAX_SCALE: tl.constexpr, CAUSAL: tl.constexpr, CAUSAL_OFFSET: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                     PIPELINE_STAGES: tl.constexpr, USE_EXP2: tl.constexpr, PRE_LOAD_V: tl.constexpr,
                     EXP2_PRE_SCALE: tl.constexpr):
    program = tl.program_id(0)
    query_tiles = tl.cdiv(SEQ_Q, BLOCK_M)
    query_tile = program % query_tiles
    batch_head = program // query_tiles
    batch = batch_head // HEADS
    head = batch_head % HEADS
    kv_head = head // (HEADS // KV_HEADS)
    q_base = batch_head.to(tl.int64) * SEQ_Q * D
    kv_base = (batch.to(tl.int64) * KV_HEADS + kv_head.to(tl.int64)) * SEQ_KV * D
    rows = query_tile.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, BLOCK_D)
    q = tl.load(Q + q_base + rows[:, None] * D + dims[None, :],
                mask=(rows[:, None] < SEQ_Q) & (dims[None, :] < D), other=0.0)
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    end_kv = SEQ_KV
    if CAUSAL:
        last_query = tl.minimum((query_tile.to(tl.int64) + 1) * BLOCK_M, SEQ_Q) - 1
        end_kv = tl.minimum(SEQ_KV, tl.maximum(last_query + CAUSAL_OFFSET + 1, 0))
    for begin in tl.range(0, end_kv, BLOCK_N, num_stages=PIPELINE_STAGES):
        keys = (begin + tl.arange(0, BLOCK_N)).to(tl.int64)
        key = tl.load(K + kv_base + keys[None, :] * D + dims[:, None],
                      mask=(keys[None, :] < SEQ_KV) & (dims[:, None] < D), other=0.0)
        if PRE_LOAD_V:
            value = tl.load(V + kv_base + keys[:, None] * D + dims[None, :],
                            mask=(keys[:, None] < SEQ_KV) & (dims[None, :] < D), other=0.0)
        scores = tl.dot(q, key)
        if USE_EXP2 and EXP2_PRE_SCALE:
            scores = scores * (SOFTMAX_SCALE * 1.4426950408889634)
        else:
            scores = scores * SOFTMAX_SCALE
        visible = (rows[:, None] < SEQ_Q) & (keys[None, :] < SEQ_KV)
        if CAUSAL:
            visible = visible & (keys[None, :] <= rows[:, None] + CAUSAL_OFFSET)
        scores = tl.where(visible, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        # A row may have no visible key, including entire early Q tiles when
        # Sq > Sk.  Replacing -inf with zero avoids the undefined -inf - -inf.
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        if USE_EXP2:
            correction_exponent = maximum - safe_maximum
            probability_exponents = scores - safe_maximum[:, None]
            if not EXP2_PRE_SCALE:
                # Large scales or QK scores may overflow pre-scaled logits.
                # These centered differences are nonpositive: conversion can
                # only underflow to -inf, whose exponential is correctly zero.
                correction_exponent = correction_exponent * 1.4426950408889634
                probability_exponents = probability_exponents * 1.4426950408889634
            correction = tl.exp2(correction_exponent)
            probabilities = tl.exp2(probability_exponents)
        else:
            correction = tl.exp(maximum - safe_maximum)
            probabilities = tl.exp(scores - safe_maximum[:, None])
        normalizer = normalizer * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        if not PRE_LOAD_V:
            value = tl.load(V + kv_base + keys[:, None] * D + dims[None, :],
                            mask=(keys[:, None] < SEQ_KV) & (dims[None, :] < D), other=0.0)
        # Keep both dot operands in the input dtype, for FP16 and BF16 alike.
        accumulator = tl.dot(probabilities.to(V.dtype.element_ty), value, accumulator)
        maximum = next_maximum

    denominator = tl.where(normalizer > 0.0, normalizer, 1.0)
    output = accumulator / denominator[:, None]
    output = tl.where(normalizer[:, None] > 0.0, output, 0.0)
    tl.store(Out + q_base + rows[:, None] * D + dims[None, :], output,
             mask=(rows[:, None] < SEQ_Q) & (dims[None, :] < D))
