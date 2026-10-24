"""Real compact/persistent ragged expert GEMM; imported only on a CUDA host."""

import triton
import triton.language as tl


@triton.jit
def build_expert_tile_map(Offsets, Experts, Rows, Count, GROUPS: tl.constexpr,
                          BLOCK_M: tl.constexpr, MAP_BLOCK_SIZE: tl.constexpr):
    expert = tl.program_id(0)
    first = tl.load(Offsets + expert)
    stop = tl.load(Offsets + expert + 1)
    tiles = tl.cdiv(stop - first, BLOCK_M)
    base = tl.atomic_add(Count, tiles)
    lanes = tl.arange(0, MAP_BLOCK_SIZE)
    for block in range(tl.cdiv(tiles, MAP_BLOCK_SIZE)):
        indices = block * MAP_BLOCK_SIZE + lanes
        tl.store(Experts + base + indices, expert, mask=indices < tiles)
        tl.store(Rows + base + indices, first + indices * BLOCK_M, mask=indices < tiles)


@triton.jit
def grouped_expert_gemm(A, B, Partial, ADesc, BDesc, Offsets, Experts, Rows, Count,
                        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, NUM_PROGRAMS: tl.constexpr,
                        PIPELINE_STAGES: tl.constexpr, USE_TMA: tl.constexpr):
    program = tl.program_id(0)
    partition = tl.program_id(1)
    m_tiles = tl.load(Count)
    n_tiles = tl.cdiv(N, BLOCK_N)
    total_tiles = m_tiles * n_tiles
    for tile in range(program, total_tiles, NUM_PROGRAMS):
        group = tile // (GROUP_M * n_tiles)
        first_m = group * GROUP_M
        group_rows = tl.minimum(m_tiles - first_m, GROUP_M)
        local = tile % (GROUP_M * n_tiles)
        tile_m = first_m + local % group_rows
        tile_n = local // group_rows
        expert = tl.load(Experts + tile_m)
        first = tl.load(Rows + tile_m)
        stop = tl.load(Offsets + expert + 1)
        rr = first.to(tl.int64) + tl.arange(0, BLOCK_M)
        cc = tile_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
        kk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for block in tl.range(partition, tl.cdiv(K, BLOCK_K), SPLIT_K, num_stages=PIPELINE_STAGES):
            reduction = block * BLOCK_K + kk
            if USE_TMA:
                lhs = ADesc.load([first, block * BLOCK_K])
                rhs = BDesc.load([expert, block * BLOCK_K, tile_n * BLOCK_N]).reshape((BLOCK_K, BLOCK_N))
                lhs = tl.where((rr[:, None] < stop) & (reduction[None, :] < K), lhs, 0.0)
                rhs = tl.where((reduction[:, None] < K) & (cc[None, :] < N), rhs, 0.0)
            else:
                lhs = tl.load(A + rr[:, None] * K + reduction[None, :].to(tl.int64),
                              mask=(rr[:, None] < stop) & (reduction[None, :] < K), other=0.0)
                rhs = tl.load(B + expert.to(tl.int64) * K * N + reduction[:, None].to(tl.int64) * N + cc[None, :],
                              mask=(reduction[:, None] < K) & (cc[None, :] < N), other=0.0)
            acc += tl.dot(lhs, rhs)
        tl.store(Partial + partition.to(tl.int64) * M * N + rr[:, None] * N + cc[None, :], acc,
                 mask=(rr[:, None] < stop) & (cc[None, :] < N))


@triton.jit
def reduce_expert_split_k(Partial, Output, ELEMENTS: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    indices = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    partitions = tl.arange(0, SPLIT_K)
    values = tl.load(Partial + partitions[:, None] * ELEMENTS + indices[None, :], mask=indices[None, :] < ELEMENTS, other=0.0)
    tl.store(Output + indices, tl.sum(values, axis=0), mask=indices < ELEMENTS)
