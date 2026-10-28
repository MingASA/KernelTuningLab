"""Portable Python policy contract, guarded execution, independent scoring and logs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Protocol

from ..datasets import require_training_records
from ..models import HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec, SearchBudget, ShapeSpec, canonical_json, stable_id
from ..replay import ReplayResult, run_replay, source_context_id
from ..search import HeuristicSearch, SearchHistory, make_searcher
from ..space import ConfigSpace


@dataclass(frozen=True)
class PolicyDecision:
    candidates: tuple[KernelConfig, ...] = ()
    recommendation: KernelConfig | None = None


class Policy(Protocol):
    name: str
    version: str

    def select_config(self, hardware: HardwareSpec, op: OperatorSpec, shape: ShapeSpec) -> tuple[KernelConfig, ...]: ...

    def search_config(self, hardware: HardwareSpec, op: OperatorSpec, shape: ShapeSpec,
                      history: Sequence[MeasurementRecord], budget: SearchBudget) -> PolicyDecision: ...


def _incumbent(history) -> KernelConfig | None:
    best = min((r for r in history if r.successful), key=lambda r: (r.median_ms, r.config.id), default=None)
    return best.config if best is not None else None


class SearchPolicy:
    """A reusable algorithm policy; search calls return next measurements and incumbent."""

    version = "1"

    def __init__(self, space: ConfigSpace, algorithm: str = "surrogate", seed: int = 0, training_records=(),
                 candidate_pool_size: int = 512, exhaustive_threshold: int = 50_000):
        self.space = space
        self.algorithm = algorithm
        self.seed = seed
        self.candidate_pool_size = candidate_pool_size
        self.exhaustive_threshold = exhaustive_threshold
        self.training_records = require_training_records(training_records)
        self.name = f"{algorithm}_policy"
        self.searcher = make_searcher(algorithm, seed=seed, training_records=self.training_records,
                                      candidate_pool_size=candidate_pool_size, exhaustive_threshold=exhaustive_threshold)

    @property
    def specification(self) -> dict:
        return {"algorithm": self.algorithm, "seed": self.seed, "space": self.space.name,
                "domains": self.space.domains,
                "candidate_pool_size": self.candidate_pool_size, "exhaustive_threshold": self.exhaustive_threshold,
                "training_record_ids": sorted(r.id for r in self.training_records)}

    def select_config(self, hardware, op, shape):
        # Zero-shot transfer reuses training observations; other policies use the
        # analytic ordering since no new-hardware measurements are available.
        searcher = make_searcher("transfer" if self.algorithm == "transfer" else "heuristic",
                                 seed=self.seed, training_records=self.training_records,
                                 candidate_pool_size=self.candidate_pool_size, exhaustive_threshold=self.exhaustive_threshold)
        return tuple(searcher.suggest(hardware, op, shape, self.space, (), SearchBudget(1)))

    def search_config(self, hardware, op, shape, history, budget):
        candidates = tuple(self.searcher.suggest(hardware, op, shape, self.space, history, budget))
        return PolicyDecision(candidates, _incumbent(history))


class HeuristicPolicy(SearchPolicy):
    def __init__(self, space: ConfigSpace, seed: int = 0, candidate_pool_size: int = 512, exhaustive_threshold: int = 50_000):
        super().__init__(space, algorithm="heuristic", seed=seed,
                         candidate_pool_size=candidate_pool_size, exhaustive_threshold=exhaustive_threshold)


class TransferPolicy(SearchPolicy):
    def __init__(self, space: ConfigSpace, training_records=(), seed: int = 0,
                 candidate_pool_size: int = 512, exhaustive_threshold: int = 50_000):
        super().__init__(space, algorithm="transfer", seed=seed, training_records=training_records,
                         candidate_pool_size=candidate_pool_size, exhaustive_threshold=exhaustive_threshold)


class FunctionPolicy:
    """Adapt generated rules/functions without requiring a particular AI SDK."""

    def __init__(self, select: Callable, search: Callable | None = None,
                 name: str = "function_policy", version: str = "1"):
        self.select = select
        self.search = search
        self.name = name
        self.version = version
        self.training_records = ()

    def select_config(self, hardware, op, shape):
        value = self.select(hardware, op, shape)
        return (value,) if isinstance(value, KernelConfig) else tuple(value)

    def search_config(self, hardware, op, shape, history, budget):
        if self.search is not None:
            return self.search(hardware, op, shape, history, budget)
        seen = {r.config.id for r in history} | set(getattr(history, "attempted_config_ids", ()))
        candidates = tuple(c for c in self.select_config(hardware, op, shape) if c.id not in seen)
        return PolicyDecision(candidates[:budget.max_evaluations], _incumbent(history))


class SafePolicy:
    """Catch contract errors and validate outputs, then use a legal heuristic fallback.

    This is output validation for trusted Python code, not a security sandbox.
    """

    def __init__(self, policy: Policy, space: ConfigSpace, seed: int = 0):
        self.policy = policy
        self.space = space
        self.seed = seed
        self.name = getattr(policy, "name", type(policy).__name__)
        self.version = getattr(policy, "version", "unversioned")
        self.diagnostics: list[dict] = []
        self.training_records = getattr(policy, "training_records", ())

    def _valid(self, values, hardware, op, shape, history=(), limit=1):
        if isinstance(values, KernelConfig):
            values = (values,)
        seen = {r.config.id for r in history} | set(getattr(history, "attempted_config_ids", ()))
        result = []
        for value in values:
            if not isinstance(value, KernelConfig):
                self.diagnostics.append({"event": "invalid_output_type", "type": type(value).__name__})
                continue
            validation = self.space.validate(value, hardware, op, shape)
            if not validation.valid:
                self.diagnostics.append({"event": "invalid_config", "reasons": list(validation.reasons)})
                continue
            if value.id in seen:
                continue
            result.append(value)
            seen.add(value.id)
            if len(result) >= limit:
                break
        return tuple(result)

    def _fallback(self, hardware, op, shape, history, budget):
        self.diagnostics.append({"event": "heuristic_fallback"})
        return tuple(HeuristicSearch(self.seed).suggest(hardware, op, shape, self.space, history, budget))

    def select_config(self, hardware, op, shape):
        try:
            result = self._valid(self.policy.select_config(hardware, op, shape), hardware, op, shape, limit=16)
        except Exception as exc:
            self.diagnostics.append({"event": "policy_exception", "error": f"{type(exc).__name__}: {exc}"})
            result = ()
        return result or self._fallback(hardware, op, shape, (), SearchBudget(1))

    def search_config(self, hardware, op, shape, history, budget):
        incumbent = _incumbent(history)
        if budget.exhausted(0, 0):
            return PolicyDecision((), incumbent)
        failed = False
        try:
            value = self.policy.search_config(hardware, op, shape, history, budget)
            if not isinstance(value, PolicyDecision):
                raise TypeError("search_config must return PolicyDecision")
            candidates = self._valid(value.candidates, hardware, op, shape, history,
                                     limit=budget.max_evaluations)
            # An intentional empty decision ends search; invalid nonempty outputs fall back.
            failed = bool(value.candidates) and not candidates
            recommendation = value.recommendation
            if recommendation is not None and (not isinstance(recommendation, KernelConfig)
                    or not self.space.validate(recommendation, hardware, op, shape).valid):
                self.diagnostics.append({"event": "invalid_recommendation"})
                recommendation = incumbent
        except Exception as exc:
            self.diagnostics.append({"event": "policy_exception", "error": f"{type(exc).__name__}: {exc}"})
            candidates, recommendation, failed = (), incumbent, True
        if failed:
            candidates = self._fallback(hardware, op, shape, history, budget)
        # A recommendation is a prediction unless it already appears as successful history.
        return PolicyDecision(candidates, incumbent if incumbent is not None else recommendation)


class _PolicySearcher:
    def __init__(self, policy: SafePolicy, selected=()):
        self.policy = policy
        self.selected = tuple(selected)
        self.select_only = bool(self.selected)
        self.name = f"policy:{policy.name}"
        self.candidate_pool_size = getattr(policy.policy, "candidate_pool_size", 512)
        self.exhaustive_threshold = getattr(policy.policy, "exhaustive_threshold", 50_000)

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        if self.select_only:
            attempted = set(getattr(history, "attempted_config_ids", ()))
            return [config for config in self.selected if config.id not in attempted][:batch_size]
        return list(self.policy.search_config(hardware, operator, shape, history, budget).candidates)[:batch_size]


@dataclass(frozen=True)
class PolicyEvaluation:
    policy_name: str
    policy_version: str
    mode: str
    replay: ReplayResult
    diagnostics: tuple[dict, ...]
    deployment_evaluations: int
    scorer_lookups: int
    training_case_ids: tuple[str, ...]
    predicted_candidates: tuple[KernelConfig, ...] = ()

    def to_dict(self) -> dict:
        return {"policy_name": self.policy_name, "policy_version": self.policy_version,
                "mode": self.mode, "replay": self.replay.to_dict(),
                "diagnostics": list(self.diagnostics), "deployment_evaluations": self.deployment_evaluations,
                "scorer_lookups": self.scorer_lookups, "training_case_ids": list(self.training_case_ids),
                "predicted_candidates": [dict(config.params) for config in self.predicted_candidates]}


def evaluate_policy(policy: Policy, records, space: ConfigSpace, mode: str = "search",
                    budget: SearchBudget = SearchBudget(), seed: int = 0) -> PolicyEvaluation:
    """The scorer owns target records; a policy sees only revealed observations.

    Select-mode lookups evaluate predictions offline, with zero deployment
    measurements. Search-mode recommendations are scored from measured history.
    """
    if mode not in {"select", "search"}:
        raise ValueError("mode must be select or search")
    rows = tuple(records)
    if not rows or len({r.case_id for r in rows}) != 1:
        raise ValueError("policy evaluation requires exactly one nonempty target case")
    training = require_training_records(getattr(policy, "training_records", ()),
                                        forbidden_case_ids=(rows[0].case_id,))
    base_policy = policy
    while isinstance(base_policy, SafePolicy):
        base_policy = base_policy.policy
    guarded = SafePolicy(deepcopy(base_policy), space, seed=seed)
    selected = ()
    predicted_candidates = ()
    if mode == "select":
        first = rows[0]
        predicted_candidates = guarded.select_config(deepcopy(first.hardware), deepcopy(first.operator), deepcopy(first.shape))
        selected = predicted_candidates[:1]
        # Score the ranked first prediction; no oracle chooses among predictions.
        evaluation_budget = SearchBudget(len(selected))
    else:
        evaluation_budget = budget
    searcher = _PolicySearcher(guarded, selected=selected)
    if mode == "select" and not selected:
        evaluation_budget = SearchBudget(0)
    result = run_replay(rows, searcher, space, budget=evaluation_budget, seed=seed)
    return PolicyEvaluation(guarded.name, guarded.version, mode, result, tuple(guarded.diagnostics),
                            result.evaluations if mode == "search" else 0, result.evaluations,
                            tuple(sorted({r.case_id for r in training})), predicted_candidates)


def evaluate_holdout(policy: Policy, records, space: ConfigSpace, mode: str = "search",
                     budget: SearchBudget = SearchBudget(), seed: int = 0,
                     split: str = "test", unseen: str = "case") -> list[PolicyEvaluation]:
    """Evaluate held-out cases, optionally requiring unseen shape or GPU identity."""
    if unseen not in {"case", "shape", "hardware"}:
        raise ValueError("unseen must be case, shape, or hardware")
    if split not in {"validation", "test"}:
        raise ValueError("holdout evaluation requires validation or test split")
    rows = tuple(r for r in records if r.split == split)
    kwargs = {"forbidden_case_ids": {r.case_id for r in rows}}
    if unseen == "shape":
        kwargs["forbidden_shape_ids"] = {r.shape.id for r in rows}
    if unseen == "hardware":
        kwargs["forbidden_hardware_ids"] = {r.hardware.id for r in rows}
    require_training_records(getattr(policy, "training_records", ()), **kwargs)
    groups: dict[tuple[str, str], list[MeasurementRecord]] = {}
    for record in rows:
        groups.setdefault((record.case_id, source_context_id(record)), []).append(record)
    return [evaluate_policy(policy, groups[key], space, mode=mode, budget=budget, seed=seed)
            for key in sorted(groups)]


class PolicyRepository:
    """Immutable version manifests plus append-only local JSONL experiment logs."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, policy: Policy, version: str, metadata: dict | None = None) -> dict:
        registered_policy = policy
        while isinstance(registered_policy, SafePolicy):
            registered_policy = registered_policy.policy
        name = getattr(registered_policy, "name", type(registered_policy).__name__)
        if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", value) and value not in {".", ".."}
                   for value in (name, version)):
            raise ValueError("policy names/versions must be safe local identifiers")
        source_path = inspect.getsourcefile(type(registered_policy))
        source = Path(source_path).read_text(encoding="utf-8") if source_path and Path(source_path).exists() else inspect.getsource(type(registered_policy))
        if isinstance(registered_policy, FunctionPolicy):
            source += "\n" + inspect.getsource(registered_policy.select)
            if registered_policy.search is not None:
                source += "\n" + inspect.getsource(registered_policy.search)
        training = require_training_records(getattr(registered_policy, "training_records", ()))
        content = {"name": name, "version": version,
                   "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                   "source_file": str(Path(source_path).resolve()) if source_path else None,
                   "specification": getattr(registered_policy, "specification", {}),
                   "training_case_ids": sorted({r.case_id for r in training}),
                   "metadata": metadata or {}}
        content["artifact_id"] = stable_id(content)
        destination = self.root / name / version
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != json.loads(canonical_json(content)):
                raise ValueError("registered policy versions are immutable; choose a new version")
        else:
            manifest_path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (destination / "source.py").write_text(source, encoding="utf-8")
        policy.version = version
        registered_policy.version = version
        return content

    def log_evaluation(self, result: PolicyEvaluation) -> Path:
        destination = self.root / "evaluations.jsonl"
        entry = {"logged_at": datetime.now(timezone.utc).isoformat(), **result.to_dict()}
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(entry) + "\n")
        return destination


def load_policy(path, space: ConfigSpace, seed: int = 0) -> SafePolicy:
    """Load a trusted local Python module exporting ``create_policy(space, seed)``.

    Generated source is ordinary Python: review it before loading. The wrapper
    validates outputs and provides fallbacks; process isolation is not claimed.
    """
    source = Path(path).resolve()
    module_name = "ktlab_policy_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise ValueError(f"cannot import policy source {source}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not callable(getattr(module, "create_policy", None)):
        raise ValueError("policy module must export create_policy(space, seed=0)")
    policy = module.create_policy(space, seed=seed)
    if not all(callable(getattr(policy, method, None)) for method in ("select_config", "search_config")):
        raise TypeError("policy must implement select_config and search_config")
    return SafePolicy(policy, space, seed=seed)


__all__ = ["Policy", "PolicyDecision", "SearchPolicy", "HeuristicPolicy", "TransferPolicy", "FunctionPolicy",
           "SafePolicy", "PolicyEvaluation", "evaluate_policy", "evaluate_holdout", "PolicyRepository", "load_policy"]
