"""Small contiguous GEMM. Imported only after the optional GPU runtime is checked."""

import triton
import triton.language as tl


@triton.jit
def gemm_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    batch = tl.program_id(2)
    A += batch * M * K
    B += batch * K * N
    C += batch * M * N
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    reductions = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block in range(tl.cdiv(K, BLOCK_K)):
        kk = block * BLOCK_K + reductions
        a = tl.load(A + rows[:, None] * K + kk[None, :],
                    mask=(rows[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B + kk[:, None] * N + cols[None, :],
                    mask=(kk[:, None] < K) & (cols[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
    tl.store(C + rows[:, None] * N + cols[None, :], accumulator,
             mask=(rows[:, None] < M) & (cols[None, :] < N))
