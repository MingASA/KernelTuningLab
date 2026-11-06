"""T1 blocked GEMM, adapted from Triton's MIT-licensed tutorial 03.

Copyright 2018-2020 Philippe Tillet; Copyright 2020-2022 OpenAI.
See third_party/triton-tutorial-LICENSE and docs/RESEARCH_PHASE1.md.
No Triton autotuner: every launch uses exactly the platform's explicit config.
"""

import triton
import triton.language as tl


@triton.jit
def t1_gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            stride_am: tl.constexpr, stride_ak: tl.constexpr,
            stride_bk: tl.constexpr, stride_bn: tl.constexpr,
            stride_cm: tl.constexpr, stride_cn: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    grid_m, grid_n = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    group_width = GROUP_M * grid_n
    group = pid // group_width
    first_m = group * GROUP_M
    actual_group_m = tl.minimum(grid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_width) % actual_group_m
    pid_n = (pid % group_width) // actual_group_m
    rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    cols = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    kk = tl.arange(0, BLOCK_K)
    a_ptrs = A + rows[:, None] * stride_am + kk[None, :] * stride_ak
    b_ptrs = B + kk[:, None] * stride_bk + cols[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=kk[None, :] < K - block * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=kk[:, None] < K - block * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    out_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(C + out_rows[:, None] * stride_cm + out_cols[None, :] * stride_cn,
             acc.to(C.dtype.element_ty), mask=(out_rows[:, None] < M) & (out_cols[None, :] < N))
