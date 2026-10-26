"""Runtime-shape T1 comparison, derived from the same MIT tutorial-style T1.

See third_party/triton-tutorial-LICENSE. Shapes/strides are int32 runtime
arguments, explicitly excluded from value *and alignment* specialization.
"""
import triton
import triton.language as tl

RUNTIME_SCALARS = ('M','N','K','stride_am','stride_ak','stride_bk','stride_bn','stride_cm','stride_cn')


@triton.jit(do_not_specialize=RUNTIME_SCALARS, do_not_specialize_on_alignment=RUNTIME_SCALARS)
def t1_gemm_generic(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    grid_m, grid_n = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    if GROUP_M == 1:
        pid_m, pid_n = pid // grid_n, pid % grid_n
    else:
        group_width = GROUP_M * grid_n
        first_m = (pid // group_width) * GROUP_M
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
