"""Portable low-bit affine dequantization and fused A16 GEMM kernels.

Imported only by CUDA build callbacks after the runtime capability check.
Packed values use the canonical K-contiguous bitstream, not GPTQ/AWQ layouts.
"""

import triton
import triton.language as tl


@triton.jit
def _program_coordinates(M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr,
                          BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    program = tl.program_id(0)
    row_tiles = tl.cdiv(M, BLOCK_M)
    column_tiles = tl.cdiv(N, BLOCK_N)
    group_tiles = GROUP_M * column_tiles
    group = program // group_tiles
    first_row = group * GROUP_M
    rows_in_group = tl.minimum(row_tiles - first_row, GROUP_M)
    position = program - group * group_tiles
    return first_row + position % rows_in_group, position // rows_in_group


@triton.jit
def _load_codes(Packed, kk, nn, K: tl.constexpr, N: tl.constexpr, BITS: tl.constexpr):
    bit_positions = kk * BITS
    words = bit_positions // 32
    shifts = bit_positions % 32
    valid = (kk < K) & (nn < N)
    low = tl.load(Packed + words * N + nn, mask=valid, other=0).to(tl.uint32)
    codes = low >> shifts
    if BITS == 3:
        # Only W3 straddles word boundaries: K positions 10 and 21 mod 32.
        crosses = shifts + BITS > 32
        high = tl.load(Packed + (words + 1) * N + nn,
                       mask=valid & crosses, other=0).to(tl.uint32)
        # Mask the shift amount so inactive lanes never request a shift by 32.
        codes = codes | (high << ((32 - shifts) & 31))
    return codes & ((1 << BITS) - 1)


@triton.jit
def _load_weights(Packed, Scales, Zeros, kk, nn, K: tl.constexpr, N: tl.constexpr,
                  BITS: tl.constexpr, GROUP_SIZE: tl.constexpr):
    valid = (kk < K) & (nn < N)
    codes = _load_codes(Packed, kk, nn, K, N, BITS).to(tl.float32)
    groups = kk // GROUP_SIZE
    scales = tl.load(Scales + groups * N + nn, mask=valid, other=0.0).to(tl.float32)
    zeros = tl.load(Zeros + groups * N + nn, mask=valid, other=0).to(tl.float32)
    return (codes - zeros) * scales


@triton.jit
def _load_weight_tile(Packed, Scales, Zeros, k_start, columns,
                      K: tl.constexpr, N: tl.constexpr, BITS: tl.constexpr,
                      GROUP_SIZE: tl.constexpr, BLOCK_K: tl.constexpr):
    kk = k_start + tl.arange(0, BLOCK_K)
    if GROUP_SIZE >= BLOCK_K and GROUP_SIZE % BLOCK_K == 0:
        # A tile cannot cross a group here. Load each scale/zero only once per
        # N lane, then broadcast across the unpacked K tile.
        group = k_start // GROUP_SIZE
        valid = (k_start < K) & (columns < N)
        scales = tl.load(Scales + group * N + columns, mask=valid, other=0.0).to(tl.float32)
        zeros = tl.load(Zeros + group * N + columns, mask=valid, other=0).to(tl.float32)
        codes = _load_codes(Packed, kk[:, None], columns[None, :], K, N, BITS).to(tl.float32)
        weights = (codes - zeros[None, :]) * scales[None, :]
        return tl.where(kk[:, None] < K, weights, 0.0)
    else:
        return _load_weights(Packed, Scales, Zeros, kk[:, None], columns[None, :],
                             K, N, BITS, GROUP_SIZE)


@triton.jit
def dequantize_kernel(Packed, Scales, Zeros, Out, K: tl.constexpr, N: tl.constexpr,
                      BITS: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    kk = offsets // N
    nn = offsets % N
    weights = _load_weights(Packed, Scales, Zeros, kk, nn, K, N, BITS, GROUP_SIZE)
    tl.store(Out + offsets, weights, mask=offsets < K * N)


@triton.jit
def quantized_gemm_kernel(A, Packed, Scales, Zeros, Out,
                          M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                          BITS: tl.constexpr, GROUP_SIZE: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr):
    row_tile, column_tile = _program_coordinates(M, N, BLOCK_M, BLOCK_N, GROUP_M)
    partition = tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = column_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    reductions = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        k_start = (block * SPLIT_K + partition) * BLOCK_K
        kk = k_start + reductions
        lhs = tl.load(A + rows[:, None] * K + kk[None, :],
                      mask=(rows[:, None] < M) & (kk[None, :] < K), other=0.0)
        rhs = _load_weight_tile(Packed, Scales, Zeros, k_start, columns,
                                K, N, BITS, GROUP_SIZE, BLOCK_K).to(lhs.dtype)
        # Round dequantized weights to A16 before dot, matching the reference
        # and the materialized two-step path. Accumulation is always FP32.
        accumulator = tl.dot(lhs, rhs, accumulator, input_precision="ieee")
    if SPLIT_K > 1:
        Out += partition * M * N
    tl.store(Out + rows[:, None] * N + columns[None, :], accumulator,
             mask=(rows[:, None] < M) & (columns[None, :] < N))


@triton.jit
def dense_gemm_kernel(A, Weights, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr):
    row_tile, column_tile = _program_coordinates(M, N, BLOCK_M, BLOCK_N, GROUP_M)
    partition = tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = column_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    reductions = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        kk = (block * SPLIT_K + partition) * BLOCK_K + reductions
        lhs = tl.load(A + rows[:, None] * K + kk[None, :],
                      mask=(rows[:, None] < M) & (kk[None, :] < K), other=0.0)
        rhs = tl.load(Weights + kk[:, None] * N + columns[None, :],
                      mask=(kk[:, None] < K) & (columns[None, :] < N), other=0.0)
        accumulator = tl.dot(lhs, rhs, accumulator, input_precision="ieee")
    if SPLIT_K > 1:
        Out += partition * M * N
    tl.store(Out + rows[:, None] * N + columns[None, :], accumulator,
             mask=(rows[:, None] < M) & (columns[None, :] < N))


@triton.jit
def split_k_reduce_kernel(Partials, Out, ELEMENTS: tl.constexpr, SPLIT_K: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    partitions = tl.arange(0, SPLIT_K)
    partials = tl.load(Partials + partitions[:, None] * ELEMENTS + offsets[None, :],
                       mask=offsets[None, :] < ELEMENTS, other=0.0)
    output = tl.sum(partials, axis=0)
    tl.store(Out + offsets, output, mask=offsets < ELEMENTS)
