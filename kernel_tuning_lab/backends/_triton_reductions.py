"""Contiguous row reductions. Loaded only after NVIDIA Triton is available."""

import triton
import triton.language as tl


@triton.jit
def softmax_kernel(X, Y, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < N
    values = tl.load(X + row * N + columns, mask=valid, other=-float("inf")).to(tl.float32)
    shifted = values - tl.max(values, axis=0)
    numerators = tl.exp(shifted)
    probabilities = numerators / tl.sum(numerators, axis=0)
    tl.store(Y + row * N + columns, probabilities, mask=valid)


@triton.jit
def layernorm_kernel(X, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < N
    values = tl.load(X + row * N + columns, mask=valid, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / N
    centered = tl.where(valid, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / N
    normalized = centered * tl.rsqrt(variance + EPS)
    tl.store(Y + row * N + columns, normalized, mask=valid)
