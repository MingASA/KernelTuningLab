"""Reproducible CPU scale exercise without pretending synthetic results are GPU data."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

from .backends import MockAdapter
from .experiments import demo_hardware
from .l3 import PerformanceMatrix, ShapeDispatcher, bucket_baseline, greedy, greedy_local_swap, oracle
from .models import KernelConfig, ShapeSpec, SearchBudget, WorkloadDistribution
from .operators import operator_spec
from .runner import BenchmarkRunner
from .search import SEARCHER_NAMES, make_searcher
from .space import ConfigSpace, gemm_constraint
from .storage import ExperimentStore
from .tuning import tune


def run_scale_demo(output_dir, *, budget=32, candidate_pool_size=256, seed=19,
                   matrix_shapes=128, matrix_configs=64) -> dict:
    if matrix_shapes < 1 or matrix_configs < 1:
        raise ValueError("matrix shape/config counts must be positive")
    SearchBudget(budget)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hardware = demo_hardware()[0]
    operator = operator_spec("gemm")
    tiles = tuple(2**i for i in range(4, 13))
    space = ConfigSpace({"BLOCK_M": tiles, "BLOCK_N": tiles, "BLOCK_K": tiles,
                         "num_warps": (1, 2, 4, 8, 16, 32), "num_stages": tuple(range(1, 33))},
                        constraints=(gemm_constraint,), name="synthetic_scale_gemm_139968")
    shapes = tuple(ShapeSpec({"M": m, "N": 1024, "K": 512}) for m in (16, 127, 512, 1024))
    adapter = MockAdapter(compile_failure_rate=0, correctness_failure_rate=0, runtime_failure_rate=0)
    online = []
    with ExperimentStore(output / "online.sqlite") as store:
        runner = BenchmarkRunner(adapter, store, repetitions=5, seed=seed, experiment_id=f"mock-scale-v1-{seed}")
        for shape in shapes:
            for name in SEARCHER_NAMES:
                # Independent algorithm runs share no source observations or cache allowance.
                searcher = make_searcher(name, seed=seed, candidate_pool_size=candidate_pool_size)
                result = tune(runner, hardware, operator, shape, space, searcher, SearchBudget(budget), resume=False)
                payload = result.to_dict()
                payload["shape"] = shape.dims
                payload["provenance"] = "mock"
                payload["regret"] = None
                payload["regret_unavailable_reason"] = "full large-space optimum was not measured"
                online.append(payload)
        store.export_jsonl(output / "online.jsonl")

    # A separate, fully observed synthetic matrix exercises distribution-level scaling.
    workload = WorkloadDistribution(tuple(ShapeSpec({"M": i + 1, "N": 1024}) for i in range(matrix_shapes)),
                                    tuple(1.0 + (i % 7) for i in range(matrix_shapes)), name="synthetic-scale-workload")
    configs = tuple(KernelConfig({"variant": j}) for j in range(matrix_configs))
    rng = random.Random(seed)
    latencies = tuple(tuple(0.01 + 0.002 * abs((i / max(1, matrix_shapes - 1)) - (j / max(1, matrix_configs - 1)))
                            + 0.0002 * rng.random() for j in range(matrix_configs)) for i in range(matrix_shapes))
    matrix = PerformanceMatrix(workload, configs, latencies, provenance="mock",
                               metadata={"generator": "seeded synthetic matrix; not a kernel measurement", "seed": seed})
    l3_rows = []
    for k in (1, 2, 4, 8, 16):
        baseline = greedy(matrix, k)
        improved = greedy_local_swap(matrix, k, max_passes=10)
        if improved.expected_latency_ms > baseline.expected_latency_ms + 1e-12:
            raise AssertionError("local swap worsened the greedy objective")
        for result in (baseline, improved, bucket_baseline(matrix, k)):
            dispatcher = ShapeDispatcher.from_result(matrix, result)
            for shape in workload.shapes:
                dispatcher.select_config(shape)
            l3_rows.append(result.to_dict())
    # Exactness is independently checked on a small projection, without a huge brute force.
    small_shapes, small_configs = min(8, matrix_shapes), min(10, matrix_configs)
    reference = PerformanceMatrix(WorkloadDistribution(workload.shapes[:small_shapes], workload.weights[:small_shapes]),
                                  configs[:small_configs], tuple(row[:small_configs] for row in latencies[:small_shapes]), provenance="mock")
    exact = [oracle(reference, k).to_dict() for k in (1, 2, 4)]
    artifacts = {"online_results.json": online, "l3_matrix.json": matrix.to_dict(),
                 "l3_results.json": l3_rows, "small_exact_reference.json": exact,
                 "space.json": {"name": space.name, "cardinality": space.cardinality, "domains": space.domains,
                                "hardware": asdict(hardware), "operator": asdict(operator)}}
    for name, payload in artifacts.items():
        (output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {"provenance": "mock", "space_cardinality": space.cardinality, "candidate_pool_size": candidate_pool_size,
               "algorithms": list(SEARCHER_NAMES), "online_trials": len(online),
               "evaluations": sum(row["evaluations"] for row in online), "matrix_shapes": matrix_shapes,
               "matrix_configs": matrix_configs, "l3_runs": len(l3_rows), "exact_reference_runs": len(exact),
               "wall_time_s": time.perf_counter() - started, "output": str(output.resolve()),
               "notice": "CPU scale exercise only; large-space global regret unknown; no real GPU validation"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# CPU scale exercise", "", "**MOCK / SYNTHETIC; not measured GPU performance.**", "",
             f"Configuration space: {space.cardinality:,}; bounded candidate pool: {candidate_pool_size}.", "",
             "| Algorithm | Trials | New evaluations |", "| --- | ---: | ---: |"]
    for name in SEARCHER_NAMES:
        subset = [row for row in online if row["algorithm"] == name]
        lines.append(f"| {name} | {len(subset)} | {sum(row['evaluations'] for row in subset)} |")
    lines.extend(["", f"L3: {matrix_shapes} shapes × {matrix_configs} configs; 15 approximate selections and 3 small exact references.", "",
                  "The full large-space optimum is unknown. The matrix is a separate synthetic scaling problem; its latencies are not physical measurements."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
