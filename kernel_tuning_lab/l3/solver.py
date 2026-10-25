"""Finite config-budget selection, not compiled-binary or code-size budgeting."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Any, Iterable

from ..models import KernelConfig
from .matrix import PerformanceMatrix


class OracleLimitError(ValueError):
    """An exact request exceeds the caller's declared enumeration budget."""


@dataclass(frozen=True)
class SelectionResult:
    """At most k distinct config dictionaries; compiled specializations are separate."""
    method: str
    k: int
    selected_indices: tuple[int, ...]
    configs: tuple[KernelConfig, ...]
    assignment_indices: tuple[int | None, ...]
    expected_latency_ms: float | None
    conditional_latency_ms: float | None
    coverage: float
    relative_loss: float | None
    reference_latency_ms: float | None
    provenance: str
    exact: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_configs(self) -> tuple[KernelConfig, ...]:
        return self.configs

    @property
    def config_count(self) -> int:
        return len(self.selected_indices)

    @property
    def variant_count(self) -> int:
        """Legacy alias for config_count; never a physical compiled-kernel count."""
        return self.config_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method, "k": self.k, "selected_indices": list(self.selected_indices),
            "configs": [dict(config.params) for config in self.configs],
            "assignment_indices": list(self.assignment_indices), "config_count": self.config_count,
            "budget_kind": "config_count", "compiled_variant_count": None, "binary_size_bytes": None,
            "variant_count": self.variant_count,
            "expected_latency_ms": self.expected_latency_ms,
            "conditional_latency_ms": self.conditional_latency_ms, "coverage": self.coverage,
            "relative_loss": self.relative_loss, "reference_latency_ms": self.reference_latency_ms,
            "provenance": self.provenance, "exact": self.exact, "metadata": dict(self.metadata),
        }


def _validate_k(k: int, matrix: PerformanceMatrix) -> int:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("variant budget k must be a positive integer")
    return min(k, len(matrix.configs))


def _assignment(matrix: PerformanceMatrix, selected: Iterable[int]) -> tuple[int | None, ...]:
    indices = tuple(sorted(selected))
    return tuple(min((j for j in indices if row[j] is not None),
                     key=lambda j: (row[j], j), default=None) for row in matrix.latencies_ms)


def _score(
    matrix: PerformanceMatrix, selected: Iterable[int], rows: Iterable[int] | None = None,
) -> tuple[float, float]:
    """Lexicographic: minimize uncovered workload mass, then observed weighted cost."""
    selected = tuple(selected)
    weights = matrix.workload.normalized_weights
    uncovered = []
    costs = []
    for i in range(len(weights)) if rows is None else rows:
        observed = [matrix.latencies_ms[i][j] for j in selected if matrix.latencies_ms[i][j] is not None]
        if observed:
            costs.append(weights[i] * min(observed))
        else:
            uncovered.append(weights[i])
    return math.fsum(uncovered), math.fsum(costs)


def _improves(candidate: tuple[float, float], incumbent: tuple[float, float]) -> bool:
    # Strict comparison is intentional: tolerances could erase real tiny workload weights.
    return candidate < incumbent


def _result(
    matrix: PerformanceMatrix, k: int, selected: Iterable[int], method: str,
    *, exact: bool = False, metadata: dict[str, Any] | None = None,
) -> SelectionResult:
    indices = tuple(sorted(selected))
    assignment = _assignment(matrix, indices)
    uncovered, weighted_cost = _score(matrix, indices)
    covered_weights = [weight for weight, choice in zip(matrix.workload.normalized_weights, assignment)
                       if choice is not None]
    coverage = math.fsum(covered_weights)
    expected = weighted_cost if uncovered == 0 else None
    conditional = weighted_cost / coverage if coverage > 0 else None
    reference = matrix.reference_latency_ms
    relative_loss = max(0.0, expected / reference - 1.0) if expected is not None and reference else None
    details = {
        "matrix_complete": matrix.complete, "matrix_cell_coverage": matrix.coverage,
        "reference_scope": "all_candidates" if matrix.complete else "observed_cells_only",
        "objective": "minimum_uncovered_weight_then_observed_weighted_latency",
        "uncovered_weight": uncovered,
        "weighted_observed_latency_ms": weighted_cost,
        "hardware_id": matrix.metadata.get("hardware_id"),
        "operator_id": matrix.metadata.get("operator_id"),
        "shape_ids": [shape.id for shape in matrix.workload.shapes],
        "candidate_config_ids": [config.id for config in matrix.configs],
        "budget_kind": "config_count",
        "compiled_variants": "unknown unless joined with actual compiler artifact identities",
        "mock_data": matrix.provenance == "mock",
    }
    details.update(metadata or {})
    return SelectionResult(method, k, indices, tuple(matrix.configs[j] for j in indices), assignment,
                           expected, conditional, coverage, relative_loss, reference,
                           matrix.provenance, exact, details)


def oracle(matrix: PerformanceMatrix, k: int, *, max_combinations: int = 200_000) -> SelectionResult:
    """Exact finite optimum; refuse excessive enumeration before any expensive work.

    Adding variants cannot worsen coverage or latency. Enumerating all subsets of
    size min(k, |C|) therefore solves the constraint |S| <= k exactly, including
    partially observed matrices under the documented coverage-first objective.
    """
    count = _validate_k(k, matrix)
    if not isinstance(max_combinations, int) or isinstance(max_combinations, bool) or max_combinations < 1:
        raise ValueError("max_combinations must be a positive integer")
    # Reaching the unrestricted row-wise lower bound is an exact certificate,
    # frequently avoiding enumeration for K >= the number of workload shapes.
    unrestricted = _assignment(matrix, range(len(matrix.configs)))
    certificate = {j for weight, j in zip(matrix.workload.normalized_weights, unrestricted)
                   if weight > 0 and j is not None}
    if len(certificate) <= count:
        return _result(matrix, k, certificate, "oracle", exact=True, metadata={
            "subsets_evaluated": 1, "optimality_certificate": "unrestricted_observed_row_minima",
            "exact_scope": "complete_matrix" if matrix.complete else "observed_coverage_objective",
        })
    combinations_needed = math.comb(len(matrix.configs), count)
    if combinations_needed > max_combinations:
        raise OracleLimitError(
            f"exact oracle requires {combinations_needed:,} subsets for {len(matrix.configs)} configs, "
            f"k={k}; limit={max_combinations:,}. Reduce candidates/k or explicitly raise the limit."
        )
    best = None
    best_score = (math.inf, math.inf)
    for selected in combinations(range(len(matrix.configs)), count):
        score = _score(matrix, selected)
        if _improves(score, best_score):
            best, best_score = selected, score
    return _result(matrix, k, best or (), "oracle", exact=True,
                   metadata={"subsets_evaluated": combinations_needed,
                             "exact_scope": "complete_matrix" if matrix.complete else "observed_coverage_objective"})


def evaluate_selection(
    matrix: PerformanceMatrix, selected_indices: Iterable[int], *, k: int | None = None,
) -> SelectionResult:
    """Score an externally supplied set with the same objective as all solvers."""
    indices = tuple(selected_indices)
    if any(not isinstance(j, int) or isinstance(j, bool) or not 0 <= j < len(matrix.configs) for j in indices):
        raise ValueError("selected_indices must be valid integer matrix column indices")
    if len(set(indices)) != len(indices):
        raise ValueError("selected_indices must not contain duplicates")
    budget = max(1, len(indices)) if k is None else k
    _validate_k(budget, matrix)
    if len(indices) > budget:
        raise ValueError("selected set exceeds the variant budget")
    return _result(matrix, budget, indices, "provided")


def greedy(matrix: PerformanceMatrix, k: int) -> SelectionResult:
    count = _validate_k(k, matrix)
    selected: list[int] = []
    current = _score(matrix, selected)
    evaluations = 0
    for _ in range(count):
        best_index = None
        best_score = current
        for j in range(len(matrix.configs)):
            if j in selected:
                continue
            score = _score(matrix, (*selected, j))
            evaluations += 1
            if _improves(score, best_score):
                best_index, best_score = j, score
        if best_index is None:
            break
        selected.append(best_index)
        current = best_score
    return _result(matrix, k, selected, "greedy", metadata={"subsets_evaluated": evaluations})


def greedy_local_swap(matrix: PerformanceMatrix, k: int, *, max_passes: int = 100) -> SelectionResult:
    """Greedy initialization followed by deterministic best improving one-for-one swaps."""
    _validate_k(k, matrix)
    if not isinstance(max_passes, int) or isinstance(max_passes, bool) or max_passes < 0:
        raise ValueError("max_passes must be a nonnegative integer")
    initial = greedy(matrix, k)
    selected = set(initial.selected_indices)
    current = _score(matrix, selected)
    swaps = 0
    evaluations = 0
    converged = False
    for _ in range(max_passes):
        best = None
        best_score = current
        for old in sorted(selected):
            for new in range(len(matrix.configs)):
                if new in selected:
                    continue
                proposal = selected - {old} | {new}
                score = _score(matrix, proposal)
                evaluations += 1
                if _improves(score, best_score):
                    best, best_score = proposal, score
        if best is None:
            converged = True
            break
        selected, current = best, best_score
        swaps += 1
    return _result(matrix, k, selected, "greedy_local_swap", metadata={
        "swaps": swaps, "converged": converged, "max_passes": max_passes,
        "swap_subsets_evaluated": evaluations,
        "initial_expected_latency_ms": initial.expected_latency_ms,
    })


def bucket_baseline(matrix: PerformanceMatrix, k: int, *, bucket_dimension: str = "M") -> SelectionResult:
    """Select one representative config for each contiguous shape quantile bucket.

    Selection uses only observed cells and workload weights. Final dispatch picks
    the best observed member of the selected set, just like the other solvers.
    Bucket boundaries describe selection, never extrapolated performance claims.
    """
    _validate_k(k, matrix)
    if not isinstance(bucket_dimension, str) or not bucket_dimension:
        raise ValueError("bucket_dimension must be a nonempty dimension name")
    if any(bucket_dimension not in shape.dims for shape in matrix.workload.shapes):
        raise ValueError(f"every workload shape must contain bucket dimension {bucket_dimension!r}")
    bucket_count = min(k, len(matrix.workload.shapes))
    ordered = sorted(range(len(matrix.workload.shapes)),
                     key=lambda i: (matrix.workload.shapes[i].get(bucket_dimension), matrix.workload.shapes[i].id))
    selected = set()
    buckets = []
    for b in range(bucket_count):
        group = ordered[b * len(ordered) // bucket_count:(b + 1) * len(ordered) // bucket_count]
        scores = [(_score(matrix, (j,), group), j) for j in range(len(matrix.configs))]
        score, chosen = min(scores)
        # A bucket without any observed positive-weight execution adds no evidence.
        if any(matrix.workload.normalized_weights[i] > 0 and matrix.latencies_ms[i][chosen] is not None for i in group):
            selected.add(chosen)
        buckets.append({"shape_indices": group, "config_index": chosen,
                        "min_value": matrix.workload.shapes[group[0]].get(bucket_dimension),
                        "max_value": matrix.workload.shapes[group[-1]].get(bucket_dimension)})
    return _result(matrix, k, selected, "bucket", metadata={"bucket_dimension": bucket_dimension, "buckets": buckets})


def k_loss_curve(
    matrix: PerformanceMatrix,
    ks: Iterable[int] = (1, 2, 4, 8, 16),
    methods: Iterable[str] = ("greedy", "greedy_local_swap", "bucket"),
    *,
    max_combinations: int = 200_000,
    bucket_dimension: str | None = None,
) -> list[SelectionResult]:
    algorithms = {"oracle": oracle, "greedy": greedy,
                  "greedy_local_swap": greedy_local_swap, "bucket": bucket_baseline}
    requested = tuple(methods)
    unknown = set(requested) - algorithms.keys()
    if unknown:
        raise ValueError(f"unknown L3 methods: {', '.join(sorted(unknown))}")
    if "bucket" in requested and bucket_dimension is None:
        dimensions = set(matrix.workload.shapes[0].dims)
        for shape in matrix.workload.shapes[1:]:
            dimensions.intersection_update(shape.dims)
        if not dimensions:
            raise ValueError("bucket curves need a shared shape dimension")
        bucket_dimension = "M" if "M" in dimensions else sorted(dimensions)[0]
    results = []
    for k in ks:
        for name in requested:
            if name == "oracle":
                results.append(oracle(matrix, k, max_combinations=max_combinations))
            elif name == "bucket":
                results.append(bucket_baseline(matrix, k, bucket_dimension=bucket_dimension))
            else:
                results.append(algorithms[name](matrix, k))
    return results
