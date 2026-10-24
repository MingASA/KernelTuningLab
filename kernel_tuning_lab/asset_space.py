"""Resolve Kernel-owned spaces and expose a defensive search algorithm view."""

from __future__ import annotations

from copy import deepcopy
from fnmatch import fnmatchcase
import inspect
from types import MappingProxyType
from typing import Any, Mapping

from .assets import KernelDefinition, ParameterSpec, SearchAlgorithmDefinition, SpaceContext, callable_identity
from .models import HardwareSpec, KernelConfig, OperatorSpec, ShapeSpec, canonical_json
from .space import ConfigSpace, ValidationResult


def _validation_reason(result: Any, fallback: str) -> str | None:
    if isinstance(result, ValidationResult):
        return None if result.valid else "; ".join(result.reasons) or fallback
    if result is None or result is True:
        return None
    if result is False:
        return fallback
    if isinstance(result, str):
        return result or None
    raise TypeError("validation callbacks must return ValidationResult, a reason string, bool, or None")


def _bind_space_callback(callback, legacy_arguments, context_arguments, label):
    """Select a callback contract by binding, never by catching its execution.

    Legacy arguments win for callbacks accepting both forms, including *args.
    A TypeError raised inside a resolver/constraint therefore remains its own
    failure and is never mistaken for a request to retry another signature.
    """
    signature = inspect.signature(callback)
    try:
        signature.bind(*legacy_arguments)
    except TypeError:
        try:
            signature.bind(*context_arguments)
        except TypeError as exc:
            raise TypeError(f"{label} must accept legacy arguments or the SpaceContext contract: {exc}") from exc
        return context_arguments, "context"
    return legacy_arguments, "legacy"


def constraint_arguments(callback, config: KernelConfig, context: SpaceContext):
    """Return the already-bound 4-argument or 2-argument constraint inputs."""
    arguments, _ = _bind_space_callback(callback,
        (deepcopy(config), context.hardware, context.operator, context.shape),
        (deepcopy(config), context), "space constraint")
    return arguments


def build_effective_space(kernel: KernelDefinition, hardware: HardwareSpec, operator: OperatorSpec,
                          shape: ShapeSpec, restrictions: Mapping[str, Any] | None = None,
                          *, context: SpaceContext | None = None) -> ConfigSpace:
    """Intersect the declaration and experiment; never invent undeclared values.

    Hardware/input-dependent and conditional constraints remain lazy predicates,
    so resolving a million-point search space does not enumerate it.
    """
    if not isinstance(kernel, KernelDefinition):
        raise TypeError("effective spaces require a KernelDefinition")
    context_source = "default" if context is None else "provided"
    context = SpaceContext(hardware, operator, shape) if context is None else context
    if not isinstance(context, SpaceContext):
        raise TypeError("effective space context must be a SpaceContext")
    if (context.hardware.id, context.operator.id, context.shape.id) != (hardware.id, operator.id, shape.id):
        raise ValueError("SpaceContext hardware/operator/shape identities must match the effective-space inputs")
    restrictions = dict(restrictions or {})
    unknown = restrictions.keys() - kernel.space.parameters.keys()
    if unknown:
        raise ValueError("experiment restricts undeclared Kernel parameters: " + ", ".join(sorted(unknown)))
    context_domains = kernel.space.domains
    resolved = {}
    resolver_style = None
    if kernel.space.resolver is not None:
        arguments, resolver_style = _bind_space_callback(kernel.space.resolver,
            (context.hardware, context.operator, context.shape), (context,), "space resolver")
        resolved = kernel.space.resolver(*arguments)
        if not isinstance(resolved, Mapping):
            raise ValueError("space resolver must return a mapping of parameters to finite subdomains")
        if resolved.keys() - context_domains.keys():
            raise ValueError("space resolver returned undeclared parameters")
        for name, selected in resolved.items():
            if isinstance(selected, (str, bytes)) or not isinstance(selected, (tuple, list)) or not selected:
                raise ValueError(f"space resolver must return a nonempty finite subdomain for {name!r}")
            declared = {canonical_json(value) for value in context_domains[name]}
            identities = [canonical_json(value) for value in selected]
            if len(set(identities)) != len(identities) or any(value not in declared for value in identities):
                raise ValueError(f"space resolver widens or duplicates parameter {name!r}")
            context_domains[name] = tuple(selected)
    domains = {}
    pruned = {}
    for name, specification in kernel.space.parameters.items():
        selected = restrictions.get(name, specification.values)
        if isinstance(selected, Mapping):
            if set(selected) != {"values"}:
                raise ValueError(f"restriction for {name!r} only supports a values list")
            selected = selected["values"]
        if isinstance(selected, (str, bytes)) or not isinstance(selected, (tuple, list)):
            raise ValueError(f"restriction for {name!r} must be a finite list or tuple")
        selected = tuple(selected)
        if not selected:
            raise ValueError(f"restriction makes parameter {name!r} empty")
        declared = {canonical_json(value) for value in specification.values}
        if any(canonical_json(value) not in declared for value in selected):
            raise ValueError(f"restriction widens Kernel parameter {name!r} beyond its declared domain")
        if len({canonical_json(value) for value in selected}) != len(selected):
            raise ValueError(f"restriction for {name!r} contains duplicate values")
        available = {canonical_json(value) for value in context_domains[name]}
        pruned[name] = [value for value in selected if canonical_json(value) not in available]
        selected = tuple(value for value in selected if canonical_json(value) in available)
        if not selected:
            raise ValueError(f"hardware/input space resolver leaves no allowed values for parameter {name!r}")
        # Preserve experiment order: it is a research variable for Grid search.
        domains[name] = selected

    def capability_constraint(config, current_hardware, current_operator, current_shape):
        if kernel.dtypes and current_operator.dtype not in kernel.dtypes:
            return f"Kernel {kernel.ref} does not support dtype {current_operator.dtype!r}"
        if kernel.architectures and not any(fnmatchcase(current_hardware.architecture, pattern) for pattern in kernel.architectures):
            return f"Kernel {kernel.ref} does not support architecture {current_hardware.architecture!r}"
        if kernel.supports is not None:
            return _validation_reason(kernel.supports(current_hardware, current_operator, current_shape), "Kernel rejected this input/hardware")
        return None

    def conditional_constraint(config, current_hardware, current_operator, current_shape):
        active = {}
        def is_active(name):
            if name in active:
                return active[name]
            specification = kernel.space.parameters[name]
            enabled = True
            for parent, condition in specification.active_when.items():
                choices = condition if isinstance(condition, (tuple, list)) else (condition,)
                if not is_active(parent) or canonical_json(config.get(parent)) not in {canonical_json(value) for value in choices}:
                    enabled = False
                    break
            active[name] = enabled
            return enabled
        for name, specification in kernel.space.parameters.items():
            if not is_active(name) and canonical_json(config.get(name)) != canonical_json(specification.default):
                return f"inactive parameter {name!r} must retain its declared default {specification.default!r}"
        return None

    def wrap_constraint(constraint):
        # Validate the contract while resolving, even if no candidate is run.
        constraint_arguments(constraint, KernelConfig({}), context)
        def validate(config, current_hardware, current_operator, current_shape):
            current_context = SpaceContext(current_hardware, current_operator, current_shape,
                environment=context.environment, workspace_memory_limit_bytes=context.workspace_memory_limit_bytes)
            arguments = constraint_arguments(constraint, config, current_context)
            return _validation_reason(constraint(*arguments), "Kernel constraint rejected this configuration")
        return validate

    constraints = [capability_constraint]
    if any(specification.active_when for specification in kernel.space.parameters.values()):
        constraints.append(conditional_constraint)
    constraints.extend(wrap_constraint(constraint) for constraint in kernel.space.constraints)
    identity = {"kernel_ref": kernel.ref, "kernel_fingerprint": kernel.fingerprint,
                "space_fingerprint": kernel.space.fingerprint,
                "hardware_id": hardware.id, "operator_id": operator.id, "shape_id": shape.id,
                "context_fingerprint": context.fingerprint}
    space = ConfigSpace(domains, constraints=tuple(constraints), name=kernel.space.name, identity=identity, allow_empty=True)
    space.parameter_specs = MappingProxyType(dict(kernel.space.parameters))
    space.conditional = any(specification.active_when for specification in kernel.space.parameters.values())
    # Capability checks do not turn an otherwise unconstrained Cartesian space
    # into an algorithmically constrained space.
    space.has_constraints = bool(kernel.space.constraints or space.conditional)
    space.asset_identity = identity
    space.context = context
    space.resolution = {"context_domains": context_domains,
                        "pruned_by_context": {name: values for name, values in pruned.items() if values},
                        "has_resolver": kernel.space.resolver is not None,
                        "context": context.to_dict(), "context_fingerprint": context.fingerprint,
                        "context_source": context_source,
                        "limits": {"workspace_memory_limit_bytes": context.workspace_memory_limit_bytes},
                        "resolver_style": resolver_style,
                        "resolver_callback": callable_identity(kernel.space.resolver) if kernel.space.resolver else None}
    return space


def check_algorithm_compatibility(definition: SearchAlgorithmDefinition, space) -> ValidationResult:
    if not isinstance(definition, SearchAlgorithmDefinition):
        return ValidationResult(False, ("algorithm must be a SearchAlgorithmDefinition",))
    capabilities = definition.capabilities
    specifications = getattr(space, "parameter_specs", None)
    if specifications is None:
        specifications = {name: ParameterSpec(values) for name, values in space.domains.items()}
    reasons = []
    unsupported = {specification.kind for specification in specifications.values()} - set(capabilities.parameter_kinds)
    if unsupported:
        reasons.append("algorithm does not support parameter kinds: " + ", ".join(sorted(unsupported)))
    if getattr(space, "conditional", False) and not capabilities.supports_conditional:
        reasons.append("algorithm does not support conditional parameters")
    if getattr(space, "has_constraints", bool(getattr(space, "constraints", ()))) and not capabilities.supports_constraints:
        reasons.append("algorithm does not support constrained spaces")
    missing = set(capabilities.required_parameters) - set(space.domains)
    if missing:
        reasons.append("algorithm requires missing parameters: " + ", ".join(sorted(missing)))
    if capabilities.max_dimensions is not None and len(space.domains) > capabilities.max_dimensions:
        reasons.append(f"space has more than {capabilities.max_dimensions} dimensions supported by the algorithm")
    if capabilities.max_cardinality is not None and space.cardinality > capabilities.max_cardinality:
        reasons.append(f"space cardinality exceeds algorithm limit {capabilities.max_cardinality}")
    return ValidationResult(not reasons, tuple(reasons))


class SpaceView:
    """Read-only, defensively copied facade over the authoritative finite space.

    This protects platform state from accidental plugin mutation.  Python plugin
    loading is trusted code execution, not a security sandbox.
    """

    __slots__ = ("__space",)

    def __new__(cls, space: ConfigSpace):
        # Legacy algorithms use id(space) for their cursor/feature caches.  A
        # fresh facade on each suggestion would silently reset those caches.
        existing = getattr(space, "_asset_read_only_view", None)
        if isinstance(existing, cls):
            return existing
        instance = super().__new__(cls)
        space._asset_read_only_view = instance
        return instance

    def __init__(self, space: ConfigSpace):
        object.__setattr__(self, "_SpaceView__space", space)

    def __setattr__(self, name, value):
        raise AttributeError("search algorithms receive a read-only space")

    def __deepcopy__(self, memo):
        # ConfigSpace caches this facade, creating a space/view reference cycle.
        # A copied policy must bind to its copied space, not retain a view onto
        # the original experiment. deepcopy's memo resolves either copy order.
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        copied_space = deepcopy(self.__space, memo)
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        copied = SpaceView(copied_space)
        memo[id(self)] = copied
        return copied

    @property
    def domains(self):
        return MappingProxyType(deepcopy(self.__space.domains))

    @property
    def parameter_specs(self):
        specifications = getattr(self.__space, "parameter_specs", None)
        if specifications is None:
            specifications = {name: ParameterSpec(values) for name, values in self.__space.domains.items()}
        return MappingProxyType({name: ParameterSpec(deepcopy(specification.values), specification.kind,
                                                     deepcopy(dict(specification.active_when)), deepcopy(specification.default),
                                                     description=specification.description, unit=specification.unit)
                                 for name, specification in specifications.items()})

    @property
    def name(self):
        return self.__space.name

    @property
    def cardinality(self):
        return self.__space.cardinality

    @property
    def fingerprint(self):
        return self.__space.fingerprint

    @property
    def identity(self):
        return deepcopy(self.__space.identity)

    @property
    def asset_identity(self):
        return deepcopy(getattr(self.__space, "asset_identity", {}))

    @property
    def resolution(self):
        return deepcopy(getattr(self.__space, "resolution", {}))

    @property
    def conditional(self):
        return getattr(self.__space, "conditional", False)

    @property
    def has_constraints(self):
        return getattr(self.__space, "has_constraints", bool(self.__space.constraints))

    def __len__(self):
        return len(self.__space)

    def config_at(self, index):
        return deepcopy(self.__space.config_at(index))

    def sample_configs(self, rng, count, exclude_ids=()):
        return deepcopy(self.__space.sample_configs(rng, count, exclude_ids=exclude_ids))

    def validate(self, config, hardware, operator, shape):
        return self.__space.validate(deepcopy(config), deepcopy(hardware), deepcopy(operator), deepcopy(shape))

    def iter_configs(self):
        for config in self.__space.iter_configs():
            yield deepcopy(config)

    def neighbors(self, config):
        return deepcopy(self.__space.neighbors(deepcopy(config)))

    def legal_configs(self, hardware, operator, shape):
        return [config for config in self.iter_configs() if self.validate(config, hardware, operator, shape).valid]


__all__ = ["build_effective_space", "check_algorithm_compatibility", "SpaceView"]
