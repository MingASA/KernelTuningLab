"""Bounded operations on ConfigSpace, without a second space or measurement store."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import operator
import random
import time
from typing import Mapping

from .models import KernelConfig, canonical_json


@dataclass(frozen=True)
class SpaceBuildBudget:
    max_nodes: int = 100_000
    max_wall_time_s: float = 10.0
    max_configs: int = 50_000
    max_bytes: int = 32 * 1024**2

    def __post_init__(self):
        for key in ('max_nodes', 'max_configs', 'max_bytes'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key + ' must be a positive integer')
        if isinstance(self.max_wall_time_s, bool) or not math.isfinite(self.max_wall_time_s) or self.max_wall_time_s < 0:
            raise ValueError('max_wall_time_s must be finite and nonnegative')


@dataclass(frozen=True)
class PartialValidation:
    status: str
    reasons: tuple[str, ...] = ()

    @property
    def valid(self):
        return self.status == 'valid'


@dataclass(frozen=True)
class SpaceBuildResult:
    configs: tuple[KernelConfig, ...]
    status: str
    diagnostics: Mapping = field(default_factory=dict)

    @property
    def complete(self):
        return self.status in ('complete', 'proven_empty')

    def to_dict(self):
        return {'status': self.status, 'complete': self.complete,
                'configs': [dict(c.params) for c in self.configs], 'diagnostics': dict(self.diagnostics)}


@dataclass(frozen=True)
class Relation:
    """An explicit relation between two parameters, never inferred from Python."""
    left: str
    op: str
    right: str

    def __post_init__(self):
        if self.op not in ('==', '!=', '<', '<=', '>', '>=', 'divides'):
            raise ValueError('unsupported structured relation')
        if self.left == self.right:
            raise ValueError('relation requires two distinct parameter names')

    @property
    def dependencies(self):
        return (self.left, self.right)

    def to_dict(self):
        return {'type': 'relation', 'left': self.left, 'op': self.op, 'right': self.right}

    def check_partial(self, values):
        if not all(k in values for k in self.dependencies):
            return PartialValidation('unknown')
        a, b = values[self.left], values[self.right]
        fn = {'==': operator.eq, '!=': operator.ne, '<': operator.lt, '<=': operator.le,
              '>': operator.gt, '>=': operator.ge, 'divides': lambda a, b: a != 0 and b % a == 0}[self.op]
        return PartialValidation('valid' if fn(a, b) else 'invalid',
                                 () if fn(a, b) else (f'{self.left} {self.op} {self.right} is not satisfied',))

    def __call__(self, config, hardware, operator, shape):
        result = self.check_partial(config.params)
        return '; '.join(result.reasons) if result.status == 'invalid' else None


@dataclass(frozen=True)
class Forbidden:
    """Forbid one explicit conjunction; alternatives are finite value lists."""
    values: Mapping

    @property
    def dependencies(self):
        return tuple(self.values)

    def to_dict(self):
        return {'type': 'forbidden', 'values': dict(self.values)}

    def check_partial(self, assignment):
        for name, value in self.values.items():
            alternatives = value if isinstance(value, (list, tuple)) else (value,)
            if name in assignment and canonical_json(assignment[name]) not in {canonical_json(v) for v in alternatives}:
                return PartialValidation('valid')
        if all(name in assignment for name in self.values):
            return PartialValidation('invalid', ('explicit forbidden combination',))
        return PartialValidation('unknown')

    def __call__(self, config, hardware, operator, shape):
        result = self.check_partial(config.params)
        return '; '.join(result.reasons) if result.status == 'invalid' else None


def _specs(space):
    return getattr(space, 'parameter_specs', {})


def _order(space):
    specs = _specs(space); result = []; visited = set()
    def add(name):
        if name in visited:
            return
        visited.add(name)
        for parent in getattr(specs.get(name), 'active_when', {}):
            add(parent)
        result.append(name)
    for name in space.domains:
        add(name)
    return result


def _active(name, values, specs):
    for parent, condition in getattr(specs.get(name), 'active_when', {}).items():
        if parent not in values:
            return None
        choices = condition if isinstance(condition, (tuple, list)) else (condition,)
        if _active(parent, values, specs) is False or canonical_json(values[parent]) not in {canonical_json(c) for c in choices}:
            return False
    return True


def normalize(space, config):
    values = dict(config.params if isinstance(config, KernelConfig) else config)
    if values.keys() - space.domains.keys():
        raise ValueError('normalization cannot invent undeclared parameters')
    specs = _specs(space)
    for name in _order(space):
        if _active(name, values, specs) is False:
            if name in values and canonical_json(values[name]) not in {canonical_json(v) for v in specs[name].values}:
                raise ValueError('inactive value outside declaration: ' + name)
            values[name] = specs[name].default
    return KernelConfig(values)


def validate_partial(space, values, hardware, operator, shape):
    values = dict(values.params if isinstance(values, KernelConfig) else values)
    if values.keys() - space.domains.keys():
        return PartialValidation('invalid', ('unknown partial parameter',))
    for name, value in values.items():
        if canonical_json(value) not in space._domain_membership[name][1]:
            return PartialValidation('invalid', (name + ' is outside its domain',))
        spec = _specs(space).get(name)
        if spec and _active(name, values, _specs(space)) is False and canonical_json(value) != canonical_json(spec.default):
            return PartialValidation('invalid', ('inactive parameter must retain its declared default: ' + name,))
    for rule in space.rules:
        check = rule.check_partial(values)
        if check.status == 'invalid':
            return check
    if len(values) == len(space.domains):
        result = space.validate(KernelConfig(values), hardware, operator, shape)
        if any(reason.startswith('constraint error:') for reason in result.reasons):
            return PartialValidation('unknown', result.reasons)
        return PartialValidation('valid' if result.valid else 'invalid', result.reasons)
    # Opaque callbacks are not parsed, and consistency is not a proof that a
    # partial assignment can be completed.
    return PartialValidation('unknown')


def canonical_configs(space):
    """Enumerate declarations with inactive defaults, without static filtering."""
    specs = _specs(space); order = _order(space)
    def walk(values, index):
        if index == len(order):
            yield KernelConfig(dict(values)); return
        name = order[index]
        domain = (specs[name].default,) if _active(name, values, specs) is False else space.domains[name]
        for value in domain:
            if canonical_json(value) not in space._domain_membership[name][1]:
                continue
            values[name] = value
            yield from walk(values, index + 1)
        values.pop(name, None)
    yield from walk({}, 0)


def construct(space, hardware, operator, shape, *, partial=None, budget=None, count=None, rng=None, exclude_ids=()):
    budget = budget or SpaceBuildBudget(); started = time.perf_counter(); reasons = Counter()
    nodes = checks = errors = used_bytes = 0; output = []; seen = set(exclude_ids); stop = None
    values = dict(normalize(space, partial or {}).params)
    specs = _specs(space); order = _order(space)
    partial_check = validate_partial(space, values, hardware, operator, shape)
    if partial_check.status == 'invalid':
        return SpaceBuildResult((), 'proven_empty', {'partial_reasons': list(partial_check.reasons), 'nodes': 0,
            'scope': 'No completion of the supplied partial assignment; not a claim about other regions',
            'wall_s': time.perf_counter()-started})
    class Stop(Exception):
        pass
    def tick():
        nonlocal nodes, stop
        if nodes >= budget.max_nodes or time.perf_counter()-started >= budget.max_wall_time_s:
            stop = 'budget_exhausted'; raise Stop()
        nodes += 1
    def accept(config):
        nonlocal checks, used_bytes, errors, stop
        if config.id in seen:
            return
        checks += 1; result = space.validate(config, hardware, operator, shape)
        if not result.valid:
            reasons.update(result.reasons)
            errors += int(any(r.startswith('constraint error:') for r in result.reasons)); return
        estimate = len(canonical_json(config.params).encode()) + 256
        if used_bytes + estimate > budget.max_bytes or len(output) >= budget.max_configs:
            stop = 'budget_exhausted'; raise Stop()
        seen.add(config.id); output.append(config); used_bytes += estimate
        if count is not None and len(output) >= count:
            stop = 'sampled'; raise Stop()
    def candidates(name):
        if _active(name, values, specs) is False:
            return (specs[name].default,)
        # Equality propagates a bound value directly; it does not rejection-sample
        # the full Cartesian product, even for disconnected diagonal regions.
        for rule in space.rules:
            if isinstance(rule, Relation) and rule.op == '==':
                partner = rule.right if rule.left == name else rule.left if rule.right == name else None
                if partner in values:
                    return (values[partner],)
        domain = space.domains[name]
        if rng is None or len(domain) <= 1:
            return domain
        start = rng.randrange(len(domain)); stride = rng.randrange(1, len(domain))
        while math.gcd(stride, len(domain)) != 1:
            stride = (stride + 1) % len(domain) or 1
        return (domain[(start+i*stride) % len(domain)] for i in range(len(domain)))
    def walk(index):
        if index == len(order):
            accept(KernelConfig(dict(values))); return
        name = order[index]
        if name in values:
            walk(index+1); return
        for value in candidates(name):
            tick()
            if canonical_json(value) not in space._domain_membership[name][1]:
                reasons['propagated value outside restricted domain'] += 1; continue
            values[name] = value
            bad = next((r.check_partial(values) for r in space.rules if r.check_partial(values).status == 'invalid'), None)
            if bad is None:
                walk(index+1)
            else:
                reasons.update(bad.reasons)
        values.pop(name, None)
    try:
        if space.native_configs is not None:
            configs = list(space.native_configs)
            if rng is not None:
                rng.shuffle(configs)
            for config in configs:
                tick()
                if all(canonical_json(config.get(k)) == canonical_json(v) for k, v in values.items()):
                    accept(config)
        else:
            tick(); walk(0)
    except Stop:
        pass
    status = stop or ('unsupported' if errors else 'complete' if output else 'exhausted' if exclude_ids else 'proven_empty')
    return SpaceBuildResult(tuple(output), status, {'nodes': nodes, 'final_checks': checks,
        'wall_s': time.perf_counter()-started, 'estimated_result_bytes': used_bytes,
        'constraint_errors': errors, 'rejection_reasons': dict(reasons),
        'distribution': 'uniform shuffled native list' if space.native_configs is not None and rng else 'dependency-aware randomized cyclic DFS; not uniform over feasible leaves' if rng else 'dependency-aware deterministic DFS',
        'declared_cardinality': space.cardinality,
        'notice': 'Result storage bytes are estimated, not an OS RSS limit; unknown callbacks run only on complete assignments.'})
