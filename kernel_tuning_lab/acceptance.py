"""Executable extension checks, bounded proposals and reproducible space probes."""

from __future__ import annotations

from copy import deepcopy
import inspect
from itertools import islice
import random
from typing import Any

from .asset_space import SpaceView
from .models import HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec, RecordStatus, SearchBudget, ShapeSpec
from .search import SearchHistory
from .space import ConfigSpace


class ContractError(ValueError):
    """A loaded implementation violated the public extension contract."""


def inspect_signature(callback, *arguments, **keywords):
    try:
        inspect.signature(callback).bind(*arguments, **keywords)
    except (ValueError, TypeError) as exc:
        raise ContractError(f"entry point {getattr(callback, '__name__', type(callback).__name__)} signature: {exc}") from exc


def create_searcher(definition, options: dict[str, Any], seed: int):
    arguments = {"seed": seed, **options}
    try:
        inspect.signature(definition.factory).bind(**arguments)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"algorithm {definition.ref} factory arguments: {exc}") from exc
    searcher = definition.factory(**arguments)
    if not callable(getattr(searcher, "suggest", None)):
        raise ContractError(f"algorithm {definition.ref} factory did not return a Searcher")
    return searcher


def request_candidates(searcher, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec,
                       space: ConfigSpace, history: SearchHistory, budget: SearchBudget,
                       batch_size: int = 1) -> list[KernelConfig]:
    """The algorithm never receives the mutable, authoritative configuration space."""
    before = space.fingerprint
    result = searcher.suggest(deepcopy(hardware), deepcopy(operator), deepcopy(shape), SpaceView(space),
                              deepcopy(history), budget, batch_size=batch_size)
    if isinstance(result, (str, bytes, dict)) or result is None:
        raise ContractError("suggest must return an iterable of KernelConfig values")
    proposals = list(islice(iter(result), batch_size + 1))
    if len(proposals) > batch_size:
        raise ContractError(f"suggest exceeded requested batch_size={batch_size}")
    if space.fingerprint != before:
        raise ContractError("algorithm modified the authoritative configuration space")
    if budget.exhausted(0, 0) and proposals:
        raise ContractError("algorithm returned proposals for an exhausted budget")
    seen = set(history.attempted_config_ids) | {row.config.id for row in history}
    for config in proposals:
        if not isinstance(config, KernelConfig):
            raise ContractError(f"suggest returned {type(config).__name__}; expected KernelConfig")
        if config.id in seen:
            raise ContractError("algorithm repeated an already revealed or attempted configuration")
        result = space.validate(config, hardware, operator, shape)
        if not result.valid:
            raise ContractError("algorithm proposed an illegal configuration: " + "; ".join(result.reasons))
        seen.add(config.id)
    return proposals


def inspect_kernel_entrypoint(kernel, space: ConfigSpace) -> None:
    mapping = kernel.parameter_map
    if set(mapping) - set(space.domains):
        raise ContractError("kernel parameter_map refers to undeclared parameters")
    config = space.config_at(0)
    parameters = {mapping.get(key, key): value for key, value in config.params.items()}
    if len(parameters) != len(config.params):
        raise ContractError("kernel parameter_map maps multiple knobs to the same argument")
    try:
        inspect.signature(kernel.build).bind(object(), **parameters)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"kernel {kernel.ref} build signature/parameter mapping: {exc}") from exc


def check_operator_request(definition, dtype: str, parameters: dict, shape: ShapeSpec) -> dict:
    if dtype not in definition.dtypes:
        raise ContractError(f"operator {definition.ref} does not support dtype={dtype}")
    if set(shape.dims) != set(definition.shape_dims):
        raise ContractError(f"operator {definition.ref} requires shape dimensions {definition.shape_dims}")
    unknown = set(parameters) - set(definition.parameters)
    if unknown:
        raise ContractError("undeclared operator parameters: " + ", ".join(sorted(unknown)))
    return {**definition.parameters, **parameters}


def probe_legal_configs(space: ConfigSpace, hardware: HardwareSpec, operator: OperatorSpec,
                        shape: ShapeSpec, *, seed: int, count: int = 8, max_inspected: int = 512) -> dict:
    """Bound sparse-space rejection; retain examples and counts of invalid points."""
    if count < 1 or max_inspected < 1:
        return {"configs": [], "inspected": 0, "rejected": 0, "rejection_examples": [], "exhaustive": False}
    limit = min(space.cardinality, max_inspected)
    if space.cardinality <= max_inspected:
        candidates = (space.config_at(index) for index in range(limit))
    else:
        candidates = iter(space.sample_configs(random.Random(seed), limit))
    legal, rejected, inspected, examples = [], 0, 0, []
    for config in candidates:
        inspected += 1
        result = space.validate(config, hardware, operator, shape)
        if result.valid:
            legal.append(config)
        else:
            rejected += 1
            if len(examples) < 8:
                examples.append({"config": dict(config.params), "reasons": list(result.reasons)})
    # Shuffle only the bounded pool, including small spaces, so samples are seeded.
    rng = random.Random(seed)
    rng.shuffle(legal)
    return {"configs": legal[:count], "inspected": inspected, "rejected": rejected,
            "rejection_examples": examples, "exhaustive": inspected == space.cardinality,
            "legal_found": len(legal), "selected": min(count, len(legal))}


def _synthetic_interactions(definition, options, seed, hardware, operator, shape, space, label) -> dict:
    searcher = create_searcher(definition, options, seed)
    empty = SearchHistory()
    request_candidates(searcher, hardware, operator, shape, space, empty, SearchBudget(0), 1)
    searcher = create_searcher(definition, options, seed)
    rows, attempted = [], set()
    for index in range(min(3, space.cardinality)):
        proposals = request_candidates(searcher, hardware, operator, shape, space,
                                       SearchHistory(tuple(rows), frozenset(attempted)), SearchBudget(3 - index), 1)
        if not proposals:
            if not rows:
                raise ContractError(f"algorithm returned no candidate for nonempty {label} space")
            break
        config = proposals[0]
        attempted.add(config.id)
        failure = index == 1
        rows.append(MeasurementRecord(hardware, operator, shape, config,
                                      RecordStatus.COMPILE_ERROR if failure else RecordStatus.SUCCESS,
                                      runtime_samples_ms=() if failure else (1.0 + int(config.id[:4], 16) / 65536,),
                                      failure_reason="synthetic acceptance failure observation" if failure else None,
                                      provenance="mock", environment={"purpose": "algorithm_acceptance_only"},
                                      metadata={"synthetic": True}))
    return {"space": label, "status": "passed", "suggestions": len(rows),
            "synthetic": True, "checks": ["zero_budget", "bounded_batch", "legal_configs", "revealed_history", "no_repeats"]}


def check_algorithm(definition, options, seed, hardware, operator, shape, space) -> dict:
    """Exercise real space plus advertised portable features without running a Kernel."""
    from .asset_space import check_algorithm_compatibility
    compatibility = check_algorithm_compatibility(definition, space)
    if not compatibility.valid:
        raise ContractError("algorithm/space capability mismatch: " + "; ".join(compatibility.reasons))
    cases = [("resolved_kernel_space", space)]
    if not definition.capabilities.required_parameters:
        singleton = ConfigSpace({key: (values[0],) for key, values in space.domains.items()}, name="acceptance_singleton", allow_empty=True)
        cases.append(("singleton", singleton))
        kinds = set(definition.capabilities.parameter_kinds)
        if {"integer", "categorical", "boolean"} <= kinds:
            cases.append(("mixed_finite", ConfigSpace({"tile": (1, 2), "mode": ("a", "b"), "enabled": (False, True)},
                                                      name="acceptance_mixed_finite")))
    results = [_synthetic_interactions(definition, options, seed + index, hardware, operator, shape, case_space, label)
               for index, (label, case_space) in enumerate(cases)]
    return {"status": "passed", "algorithm": definition.ref, "cases": results,
            "notice": "Synthetic protocol acceptance; no kernel performance or GPU correctness evidence."}
