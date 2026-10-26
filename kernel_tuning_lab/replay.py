"""Offline scoring with a private observation table and explicit measurement budgets."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import islice
import math
import statistics
import time
from typing import Iterable

from .datasets import require_training_records, validate_split_isolation
from .measurement_context import source_context as _source_context, source_context_id
from .models import KernelConfig, MeasurementRecord, SearchBudget, canonical_json, stable_id
from .search import SEARCHER_NAMES, SearchHistory, Searcher, make_searcher
from .space import ConfigSpace


@dataclass(frozen=True)
class ReplayStep:
    evaluation: int
    config: KernelConfig
    status: str
    latency_ms: float | None
    best_latency_ms: float | None
    source_cost_s: float | None
    cpu_elapsed_s: float
    regret: float | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["config"] = dict(self.config.params)
        return result


@dataclass(frozen=True)
class ReplayResult:
    algorithm: str
    case_id: str
    hardware_id: str
    operator_id: str
    shape_id: str
    split: str
    seed: int
    budget: int
    evaluations: int
    best_latency_ms: float | None
    regret: float | None
    oracle_best_latency_ms: float | None
    oracle_coverage: float | None
    source_measurement_cost_s: float | None
    known_measurement_cost_s: float
    cost_coverage: float
    cpu_wall_time_s: float
    target_ratio: float
    target_evaluations: int | None
    provenance: str
    trace: tuple[ReplayStep, ...]
    best_config: KernelConfig | None = None
    source_records: int = 0
    missing_evaluations: int = 0
    failed_evaluations: int = 0
    stop_reason: str = "evaluation_budget"
    hardware_name: str = ""
    shape_dims: dict | None = None
    operator_name: str = ""
    operator_kind: str = ""
    measurement_mode: str = "unspecified"
    source_context_id: str = ""
    max_wall_time_s: float | None = None
    coverage_scope: str = "legal_space"
    known_coverage_raw_space: float = 0.0
    space_cardinality: int = 0
    candidate_pool_size: int = 512
    candidate_pool_approximate: bool = False
    exhaustive_threshold: int = 4096

    def to_dict(self) -> dict:
        result = asdict(self)
        result["trace"] = [step.to_dict() for step in self.trace]
        result["best_config"] = dict(self.best_config.params) if self.best_config is not None else None
        return result


def _source_cost(record: MeasurementRecord | None) -> float | None:
    if record is None or record.metadata.get("measurement_cost_known") is False:
        return None
    if record.wall_time_s > 0 or record.metadata.get("measurement_cost_known") is True:
        return record.wall_time_s
    # Partial compile/sample timings are not labelled as total measurement cost.
    return None


def _require_compatible_contexts(records: tuple[MeasurementRecord, ...]) -> str:
    contexts = {canonical_json(_source_context(record)): _source_context(record) for record in records}
    if len(contexts) > 1:
        values = tuple(contexts.values())
        differing = [key for key in values[0]
                     if len({canonical_json(value[key]) for value in values}) > 1]
        raise ValueError("replay source contexts are incompatible in " + ", ".join(differing)
                         + "; filter to one provenance/measurement protocol/environment before replay")
    return stable_id(next(iter(contexts.values())))


class _ReplayOracle:
    """Never passed to a searcher. Known optimum is used only by final scoring."""

    def __init__(self, records: tuple[MeasurementRecord, ...], legal_ids: set[str], legal_count: int | None):
        self.__table: dict[str, MeasurementRecord] = {}
        # Duplicate measurements select by stable record ID, never by latency.
        for record in sorted(records, key=lambda r: r.id):
            # Unverified *successes* cannot form a performance oracle. A failed
            # correctness check is still a real observed failure with a cost;
            # filtering its false correctness flag would turn it into missing
            # data and understate the observed coverage and measurement cost.
            if record.successful and not record.correctness_verified:
                continue
            # Interrupted work is durable failure evidence, not a completed
            # source observation with which to simulate a research evaluation.
            if record.metadata.get("terminated_by_parent") is True:
                continue
            self.__table.setdefault(record.config.id, deepcopy(record))
        valid = [r for key, r in self.__table.items() if key in legal_ids and r.successful]
        self.__optimum = min((r.median_ms for r in valid), default=None)
        self.__coverage = (len(legal_ids & self.__table.keys()) / legal_count if legal_count else 0.0) if legal_count is not None else None

    def reveal(self, config: KernelConfig) -> MeasurementRecord | None:
        record = self.__table.get(config.id)
        # Models are frozen dataclasses with mutable nested dictionaries.
        return deepcopy(record) if record is not None else None

    def score(self, latency: float | None) -> float | None:
        if latency is None or self.__optimum is None:
            return None
        return max(0.0, latency / self.__optimum - 1)

    def reference(self) -> tuple[float | None, float | None]:
        return self.__optimum, self.__coverage


def run_replay(
    records: Iterable[MeasurementRecord],
    searcher: Searcher,
    space: ConfigSpace,
    budget: SearchBudget = SearchBudget(),
    seed: int = 0,
    target_ratio: float = 1.05,
) -> ReplayResult:
    """Replay one case; only requested observations enter the searcher's history.

    Missing and invalid requests consume attempts but do not create measurement
    records. In a time budget, known source measurement cost and actual CPU time
    both count; missing source costs remain unknown and are reported explicitly.
    An atomic observation can finish after the wall-time limit.
    """
    started = time.perf_counter()
    rows = tuple(records)
    if not rows:
        raise ValueError("replay needs at least one source record")
    if len({r.case_id for r in rows}) != 1:
        raise ValueError("run_replay accepts a single (hardware, operator, shape) case")
    if len({r.split for r in rows}) != 1:
        raise ValueError("one case cannot span dataset splits")
    context_id = _require_compatible_contexts(rows)
    if not math.isfinite(target_ratio) or target_ratio < 1:
        raise ValueError("target_ratio is a latency multiplier and must be at least 1")
    first = rows[0]
    hardware, operator, shape = first.hardware, first.operator, first.shape
    # Only public hardware/operator/shape/space go to the algorithm.
    cardinality = getattr(space, "cardinality", None)
    if cardinality is None:
        cardinality = len(space)
    approximate = cardinality > getattr(searcher, "exhaustive_threshold", 4096)
    if approximate:
        # Score only provided observations. Neither evaluator nor algorithm expands C.
        legal_ids = {record.config.id for record in rows
                     if space.validate(record.config, hardware, operator, shape).valid}
        legal_count = None
    else:
        legal_ids = {config.id for config in space.legal_configs(hardware, operator, shape)}
        legal_count = len(legal_ids)
    domain_values = {key: {canonical_json(value) for value in domain} for key, domain in space.domains.items()}
    raw_observed = {record.config.id for record in rows
                    if record.config.params.keys() == domain_values.keys()
                    and all(canonical_json(value) in domain_values[key] for key, value in record.config.params.items())}
    oracle = _ReplayOracle(rows, legal_ids, legal_count)
    observed: list[MeasurementRecord] = []
    attempted: set[str] = set()
    raw_trace: list[ReplayStep] = []
    known_cost = 0.0
    known_cost_count = 0
    best_latency = None
    best_config = None
    stop_reason = "evaluation_budget"
    algorithm_space = space.read_only() if callable(getattr(space, "read_only", None)) else space
    # Suggestions are delivered serially by default: each gets a fresh history.
    while not budget.exhausted(len(raw_trace), time.perf_counter() - started + known_cost):
        elapsed = time.perf_counter() - started + known_cost
        history = SearchHistory(tuple(deepcopy(observed)), frozenset(attempted))
        remaining = budget.remaining(len(raw_trace), elapsed)
        try:
            # Consume at most one item beyond the requested bound. In
            # particular, do not materialize a plugin's unbounded generator.
            before_space = getattr(space, "fingerprint", None)
            proposals = list(islice(iter(searcher.suggest(
                deepcopy(hardware), deepcopy(operator), deepcopy(shape),
                algorithm_space, history, remaining, batch_size=1)), 2))
            if len(proposals) > 1:
                raise ValueError("searcher exceeded requested batch_size=1")
            if getattr(space, "fingerprint", None) != before_space:
                raise ValueError("searcher modified the authoritative configuration space")
        except Exception as exc:
            stop_reason = f"searcher_error:{type(exc).__name__}:{exc}"
            break
        if not proposals:
            stop_reason = "searcher_exhausted"
            break
        for proposal in proposals:
            if budget.exhausted(len(raw_trace), time.perf_counter() - started + known_cost):
                break
            if not isinstance(proposal, KernelConfig):
                proposal = KernelConfig({"__invalid_proposal_type__": type(proposal).__name__})
            attempted.add(proposal.id)
            valid = space.validate(proposal, hardware, operator, shape)
            record = oracle.reveal(proposal) if valid.valid else None
            if not valid.valid:
                status, failure = "invalid_config", "; ".join(valid.reasons)
            elif record is None:
                status, failure = "missing", "configuration has no source measurement"
            else:
                status, failure = record.status.value, record.failure_reason
                observed.append(record)
            latency = record.median_ms if record is not None else None
            if latency is not None and (best_latency is None or latency < best_latency):
                best_latency, best_config = latency, deepcopy(proposal)
            cost = _source_cost(record)
            if cost is not None:
                known_cost += cost
                known_cost_count += 1
            raw_trace.append(ReplayStep(len(raw_trace) + 1, deepcopy(proposal), status,
                                        latency, best_latency, cost, time.perf_counter() - started,
                                        failure_reason=failure))
    cpu_elapsed = time.perf_counter() - started
    if budget.max_wall_time_s is not None and cpu_elapsed + known_cost >= budget.max_wall_time_s:
        stop_reason = "wall_time_budget"
    optimum, coverage = oracle.reference()
    # Score only after the last algorithm call; regret never enters history.
    trace = tuple(ReplayStep(step.evaluation, step.config, step.status, step.latency_ms,
                             step.best_latency_ms, step.source_cost_s, step.cpu_elapsed_s,
                             oracle.score(step.best_latency_ms), step.failure_reason)
                  for step in raw_trace)
    target_evaluations = next((step.evaluation for step in trace
                               if optimum is not None and step.best_latency_ms is not None
                               and step.best_latency_ms <= optimum * target_ratio), None)
    evaluations = len(trace)
    return ReplayResult(
        algorithm=getattr(searcher, "name", type(searcher).__name__),
        case_id=first.case_id, hardware_id=hardware.id, operator_id=operator.id,
        shape_id=shape.id, split=first.split, seed=seed, budget=budget.max_evaluations,
        evaluations=evaluations, best_latency_ms=best_latency, regret=oracle.score(best_latency),
        oracle_best_latency_ms=optimum, oracle_coverage=coverage,
        source_measurement_cost_s=known_cost if known_cost_count == evaluations else None,
        known_measurement_cost_s=known_cost,
        cost_coverage=known_cost_count / evaluations if evaluations else 1.0,
        cpu_wall_time_s=cpu_elapsed, target_ratio=target_ratio, target_evaluations=target_evaluations,
        provenance=first.provenance, trace=trace,
        best_config=best_config, source_records=len(rows),
        missing_evaluations=sum(step.status == "missing" for step in trace),
        failed_evaluations=sum(step.status not in {"success", "missing"} for step in trace),
        stop_reason=stop_reason, hardware_name=hardware.name, shape_dims=dict(shape.dims),
        operator_name=operator.name, operator_kind=operator.kind,
        measurement_mode=first.measurement_mode, source_context_id=context_id,
        max_wall_time_s=budget.max_wall_time_s,
        coverage_scope=(space.identity.get("inference_scope", "known_observations_only" if approximate else "legal_space") if isinstance(getattr(space, "identity", None), dict)
                        else "known_observations_only" if approximate else "legal_space"),
        known_coverage_raw_space=len(raw_observed) / cardinality if cardinality else 0.0,
        space_cardinality=cardinality, candidate_pool_size=getattr(searcher, "candidate_pool_size", 512),
        candidate_pool_approximate=approximate,
        exhaustive_threshold=getattr(searcher, "exhaustive_threshold", 4096),
    )


def compare_searchers(
    records: Iterable[MeasurementRecord],
    space: ConfigSpace,
    budgets: Iterable[int] = (8, 16, 32, 64, 128),
    seeds: Iterable[int] = (0, 1, 2),
    searcher_names: Iterable[str] = SEARCHER_NAMES,
    split: str = "test",
    training_records: Iterable[MeasurementRecord] = (),
    target_ratio: float = 1.05,
    candidate_pool_size: int = 512,
    exhaustive_threshold: int = 4096,
) -> list[ReplayResult]:
    """A fresh seeded searcher is used for every independent budget/case trial."""
    rows = tuple(records)
    validate_split_isolation(rows)
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    grouped: dict[tuple[str, str], list[MeasurementRecord]] = defaultdict(list)
    for record in rows:
        if record.split == split:
            grouped[record.case_id, source_context_id(record)].append(record)
    target_case_ids = {case for case, context in grouped}
    supplied_training = tuple(training_records)
    training = supplied_training or tuple(r for r in rows if r.split == "train" and r.case_id not in target_case_ids)
    require_training_records(training, forbidden_case_ids=target_case_ids)
    budget_values, seed_values, names = tuple(budgets), tuple(seeds), tuple(searcher_names)
    for count in budget_values:
        SearchBudget(count)  # Validate even when the requested split is empty.
    results = []
    for case in sorted(grouped):
        for name in names:
            for seed in seed_values:
                for count in budget_values:
                    searcher = make_searcher(name, seed=seed, training_records=training,
                                             candidate_pool_size=candidate_pool_size, exhaustive_threshold=exhaustive_threshold)
                    results.append(run_replay(grouped[case], searcher, space,
                                              SearchBudget(count), seed=seed, target_ratio=target_ratio))
    return results


def summarize_results(results: Iterable[ReplayResult]) -> list[dict]:
    """Summarize shapes/seeds within comparable hardware/operator/protocol cohorts."""
    groups: dict[tuple, list[ReplayResult]] = defaultdict(list)
    for result in results:
        groups[result.hardware_id, result.operator_id, result.provenance, result.source_context_id,
               result.split, result.algorithm, result.budget,
               -1.0 if result.max_wall_time_s is None else result.max_wall_time_s,
               result.target_ratio, result.space_cardinality, result.candidate_pool_size,
               result.exhaustive_threshold].append(result)
    summaries = []
    for cohort, rows in sorted(groups.items()):
        first = rows[0]
        regrets = [r.regret for r in rows if r.regret is not None]
        latencies = [r.best_latency_ms for r in rows if r.best_latency_ms is not None]
        costs = [r.source_measurement_cost_s for r in rows if r.source_measurement_cost_s is not None]
        targets = [r.target_evaluations for r in rows if r.target_evaluations is not None]
        case_regrets: dict[str, list[float]] = defaultdict(list)
        for result in rows:
            if result.regret is not None:
                case_regrets[result.case_id].append(result.regret)
        within_case_variances = [statistics.pvariance(values) for values in case_regrets.values() if len(values) > 1]
        coverages = [r.oracle_coverage for r in rows if r.oracle_coverage is not None]
        summaries.append({
            "algorithm": first.algorithm, "budget": first.budget, "runs": len(rows),
            "hardware_id": first.hardware_id, "hardware_name": first.hardware_name,
            "operator_id": first.operator_id, "operator_name": first.operator_name,
            "operator_kind": first.operator_kind, "provenance": first.provenance,
            "source_context_id": first.source_context_id, "measurement_mode": first.measurement_mode,
            "split": first.split, "max_wall_time_s": first.max_wall_time_s,
            "target_ratio": first.target_ratio,
            "space_cardinality": first.space_cardinality,
            "candidate_pool_size": first.candidate_pool_size,
            "exhaustive_threshold": first.exhaustive_threshold,
            "candidate_pool_approximate": any(r.candidate_pool_approximate for r in rows),
            "coverage_scope": first.coverage_scope,
            "successful_runs": len(latencies),
            "mean_regret": statistics.mean(regrets) if regrets else None,
            "std_regret": statistics.pstdev(regrets) if regrets else None,
            "cases": len({r.case_id for r in rows}),
            "mean_within_case_seed_variance": statistics.mean(within_case_variances) if within_case_variances else None,
            "mean_best_latency_ms": statistics.mean(latencies) if latencies else None,
            "mean_evaluations": statistics.mean(r.evaluations for r in rows),
            "mean_source_measurement_cost_s": statistics.mean(costs) if len(costs) == len(rows) else None,
            "known_cost_runs": len(costs),
            "mean_cpu_wall_time_s": statistics.mean(r.cpu_wall_time_s for r in rows),
            "target_reached_runs": len(targets),
            "mean_target_evaluations": statistics.mean(targets) if targets else None,
            "mean_oracle_coverage": statistics.mean(coverages) if len(coverages) == len(rows) else None,
            "mean_known_coverage_raw_space": statistics.mean(r.known_coverage_raw_space for r in rows),
        })
    return summaries


__all__ = ["ReplayStep", "ReplayResult", "run_replay", "compare_searchers", "summarize_results", "source_context_id"]
