"""Online budgeted tuning against any BenchmarkRunner-compatible adapter."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import islice
import time

from .datasets import case_id
from .models import HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec, SearchBudget, ShapeSpec
from .search import SearchHistory, Searcher
from .space import ConfigSpace


@dataclass(frozen=True)
class TuningResult:
    algorithm: str
    case_id: str
    records: tuple[MeasurementRecord, ...]
    best_record: MeasurementRecord | None
    evaluations: int
    attempts: int
    cache_hits: int
    wall_time_s: float
    stop_reason: str
    budget: SearchBudget
    space_cardinality: int = 0
    candidate_pool_size: int = 512
    candidate_pool_approximate: bool = False

    @property
    def best_config(self) -> KernelConfig | None:
        return self.best_record.config if self.best_record is not None else None

    @property
    def best_latency_ms(self) -> float | None:
        return self.best_record.median_ms if self.best_record is not None else None

    def to_dict(self) -> dict:
        return {"algorithm": self.algorithm, "case_id": self.case_id,
                "records": [r.to_dict() for r in self.records],
                "best_record_id": self.best_record.id if self.best_record is not None else None,
                "best_config": dict(self.best_config.params) if self.best_config is not None else None,
                "best_latency_ms": self.best_latency_ms,
                "recommendation_verified": self.best_record is not None and self.best_record.correctness_verified,
                "evaluations": self.evaluations, "attempts": self.attempts, "cache_hits": self.cache_hits,
            "wall_time_s": self.wall_time_s, "stop_reason": self.stop_reason,
            "space_cardinality": self.space_cardinality, "candidate_pool_size": self.candidate_pool_size,
            "candidate_pool_approximate": self.candidate_pool_approximate,
                "budget": {"max_evaluations": self.budget.max_evaluations,
                           "max_wall_time_s": self.budget.max_wall_time_s},
                "budget_semantics": "new physical attempts; failures count, cache hits do not; atomic evaluation may overrun wall limit"}


def tune(runner, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec,
         space: ConfigSpace, searcher: Searcher, budget: SearchBudget,
         split: str = "train", resume: bool = True) -> TuningResult:
    """Adaptively collect measurements, exposing only this run's revealed history.

    Every new runner attempt, including legality/compile/correctness failures,
    consumes a physical evaluation. Cached records reveal observations without
    new physical measurements; lookup time still counts against wall time.
    After physical budget exhaustion no further request is admitted. Repeated
    already-revealed requests end search to prevent cache-only loops.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    started = time.perf_counter()
    target = case_id(hardware, operator, shape)
    records: list[MeasurementRecord] = []
    attempted: set[str] = set()
    evaluations = 0
    cache_hits = 0
    best = None
    stop_reason = "evaluation_budget"
    algorithm_space = space.read_only() if callable(getattr(space, "read_only", None)) else space
    while not budget.exhausted(evaluations, time.perf_counter() - started):
        elapsed = time.perf_counter() - started
        history = SearchHistory(tuple(deepcopy(records)), frozenset(attempted))
        try:
            before_space = getattr(space, "fingerprint", None)
            proposals = list(islice(iter(searcher.suggest(
                deepcopy(hardware), deepcopy(operator), deepcopy(shape), algorithm_space,
                history, budget.remaining(evaluations, elapsed), batch_size=1)), 2))
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
        config = proposals[0]
        if not isinstance(config, KernelConfig):
            config = KernelConfig({"__invalid_proposal_type__": type(config).__name__})
        if config.id in attempted:
            stop_reason = "repeated_candidate"
            break
        if budget.exhausted(evaluations, time.perf_counter() - started):
            stop_reason = "wall_time_budget"
            break
        attempted.add(config.id)
        valid = space.validate(config, hardware, operator, shape)
        record = runner.measure(hardware, operator, shape, deepcopy(config), space,
                                split=split, resume=resume and valid.valid)
        if record.case_id != target or record.config.id != config.id or record.split != split:
            raise ValueError("runner returned a measurement from a different target/config/split")
        cached = bool(runner.last_cache_hit)
        cache_hits += int(cached)
        evaluations += int(not cached)
        records.append(deepcopy(record))
        if record.correctness_verified and valid.valid and (best is None or record.median_ms < best.median_ms):
            best = deepcopy(record)
    elapsed = time.perf_counter() - started
    if budget.max_wall_time_s is not None and elapsed >= budget.max_wall_time_s:
        stop_reason = "wall_time_budget"
    cardinality = getattr(space, "cardinality", None)
    if cardinality is None:
        cardinality = len(space)
    return TuningResult(getattr(searcher, "name", type(searcher).__name__), target, tuple(records),
                        best, evaluations, len(records), cache_hits, elapsed, stop_reason, budget,
                        cardinality, getattr(searcher, "candidate_pool_size", 512),
                        cardinality > getattr(searcher, "exhaustive_threshold", 4096))


__all__ = ["TuningResult", "tune"]
