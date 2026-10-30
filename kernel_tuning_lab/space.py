"""Operator-specific finite configuration spaces with explicit legality results."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product, combinations
from math import prod
import random
from copy import deepcopy
from typing import Any, Callable, Iterator, Mapping

from .models import HardwareSpec, KernelConfig, OperatorSpec, ShapeSpec, canonical_json, stable_id


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: tuple[str, ...] = ()


Constraint = Callable[[KernelConfig, HardwareSpec, OperatorSpec, ShapeSpec], str | None]


class ConfigSpace:
    def __init__(self, domains: Mapping[str, tuple[Any, ...] | list[Any]], constraints: tuple[Constraint, ...] = (), name: str = "custom", identity: Any = None, *, allow_empty: bool = False, rules=(), native_configs=None, advisories=()):
        self.domains = {k: tuple(v) for k, v in domains.items()}
        self.rules = tuple(rules)
        self.constraints = tuple(constraints) + self.rules
        self.advisory_callbacks = tuple(advisories)
        self.native_configs = None if native_configs is None else tuple(deepcopy(c) if isinstance(c, KernelConfig) else KernelConfig(c) for c in native_configs)
        if self.native_configs is not None and len({c.id for c in self.native_configs}) != len(self.native_configs):
            raise ValueError('native candidate configurations must be distinct')
        self._native_ids = None if self.native_configs is None else frozenset(c.id for c in self.native_configs)
        self._feasible_cache = {}
        self.name = name
        canonical_json(identity)
        self.identity = identity
        self._fingerprint_cache = None
        if (not self.domains and not allow_empty) or any(not values for values in self.domains.values()):
            raise ValueError("configuration domains must be nonempty")
        self._domain_membership = {}
        for key, values in self.domains.items():
            membership = frozenset(canonical_json(v) for v in values)
            if len(membership) != len(values):
                raise ValueError("configuration domains must not contain duplicate values")
            self._domain_membership[key] = (values, membership)
        if any(set(rule.dependencies) - self.domains.keys() for rule in self.rules):
            raise ValueError('structured rule refers to undeclared parameters')
        if self.native_configs is not None and any(set(c.params) != set(self.domains) or any(canonical_json(c.get(k)) not in self._domain_membership[k][1] for k in self.domains) for c in self.native_configs):
            raise ValueError('native candidates must use declared parameter keys and values')

    def __len__(self) -> int:
        return self.cardinality

    @property
    def cardinality(self) -> int:
        return len(self.native_configs) if self.native_configs is not None else prod(len(v) for v in self.domains.values())

    @property
    def fingerprint(self) -> str:
        """Compact protocol identity; avoid copying huge domains into every record."""
        from .assets import callable_identity
        constraints = [callable_identity(constraint) for constraint in self.constraints]
        extra = {'native_config_ids': sorted(self._native_ids) if self._native_ids is not None else None,
                 'advisories': [callable_identity(c) for c in self.advisory_callbacks]}
        signature = (self.name, tuple((key, id(values)) for key, values in self.domains.items()),
                     canonical_json(constraints), canonical_json(self.identity), canonical_json(extra))
        if self._fingerprint_cache is None or self._fingerprint_cache[0] != signature:
            self._fingerprint_cache = (signature, stable_id({"name": self.name, "domains": self.domains,
                                                          "constraints": constraints, "identity": self.identity, "capabilities": extra}))
        return self._fingerprint_cache[1]

    def read_only(self):
        """Return the public algorithm facade without copying the search space."""
        from .asset_space import SpaceView
        return SpaceView(self)

    def config_at(self, index: int) -> KernelConfig:
        """Decode one Cartesian index in O(number of dimensions), without enumeration."""
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.cardinality:
            raise IndexError("configuration index outside the finite space")
        if self.native_configs is not None:
            return deepcopy(self.native_configs[index])
        values = {}
        for key, domain in reversed(tuple(self.domains.items())):
            index, digit = divmod(index, len(domain))
            values[key] = domain[digit]
        return KernelConfig({key: values[key] for key in self.domains})

    def sample_configs(self, rng: random.Random, count: int, exclude_ids=()) -> list[KernelConfig]:
        """Sample configurations without building the Cartesian product.

        Legality is deliberately separate. Rejection work is bounded; a nearly
        exhausted space may yield fewer than ``count`` configurations.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("sample count must be a nonnegative integer")
        if count == 0:
            return []
        excluded = set(exclude_ids)
        target = min(count, self.cardinality)
        attempts = min(self.cardinality, max(64, target * 8 + len(excluded)))
        # random.sample(range(...)) does not allocate the range itself.
        if self.cardinality <= 2**63 - 1:
            indices = rng.sample(range(self.cardinality), attempts)
        else:
            chosen = set()
            while len(chosen) < attempts:
                chosen.add(rng.randrange(self.cardinality))
            indices = sorted(chosen)
            rng.shuffle(indices)
        result = []
        for index in indices:
            config = self.config_at(index)
            if config.id not in excluded:
                result.append(config)
                excluded.add(config.id)
                if len(result) == target:
                    break
        return result

    def iter_configs(self) -> Iterator[KernelConfig]:
        if self.native_configs is not None:
            yield from (deepcopy(c) for c in self.native_configs)
            return
        if getattr(self, 'conditional', False):
            from .space_tools import canonical_configs
            yield from canonical_configs(self)
            return
        keys = tuple(self.domains)
        for values in product(*(self.domains[k] for k in keys)):
            yield KernelConfig(dict(zip(keys, values)))

    def validate(self, config: KernelConfig, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec) -> ValidationResult:
        reasons = []
        for key in config.params.keys() - self.domains.keys():
            reasons.append(f"unsupported parameter: {key}")
        for key, values in self.domains.items():
            if key not in config.params:
                reasons.append(f"missing parameter: {key}")
            else:
                cached = self._domain_membership.get(key)
                if cached is None or cached[0] is not values:
                    cached = (values, frozenset(canonical_json(v) for v in values))
                    self._domain_membership[key] = cached
                if canonical_json(config.params[key]) not in cached[1]:
                    reasons.append(f"{key}={config.params[key]!r} is outside its domain")
        if not reasons and self._native_ids is not None and config.id not in self._native_ids:
            reasons.append('configuration is not in the declared native candidate set')
        if not reasons and getattr(self, 'conditional', False):
            if self.normalize(config).id != config.id:
                reasons.append('inactive parameters must retain their declared defaults')
        if not reasons:
            for constraint in self.constraints:
                try:
                    reason = constraint(config, hardware, operator, shape)
                except Exception as exc:
                    reason = f"constraint error: {type(exc).__name__}: {exc}"
                if reason:
                    reasons.append(reason)
        return ValidationResult(not reasons, tuple(reasons))

    @classmethod
    def from_configs(cls, configs, *, name='native finite candidates', identity=None):
        configs = tuple(c if isinstance(c, KernelConfig) else KernelConfig(c) for c in configs)
        if not configs or any(set(c.params) != set(configs[0].params) for c in configs):
            raise ValueError('nonempty native candidates must have identical parameter keys')
        domains = {key: list({canonical_json(c.get(key)): c.get(key) for c in configs}.values()) for key in configs[0].params}
        return cls(domains, name=name, identity=identity, native_configs=configs, allow_empty=True)

    def normalize(self, config):
        from .space_tools import normalize
        return normalize(self, config)

    def validate_partial(self, values, hardware, operator, shape):
        from .space_tools import validate_partial
        return validate_partial(self, values, hardware, operator, shape)

    def build_feasible(self, hardware, operator, shape, *, budget=None):
        from .space_tools import construct
        result = construct(self, hardware, operator, shape, budget=budget)
        if result.complete:
            self._feasible_cache[(hardware.id, operator.id, shape.id, self.fingerprint,
                                  result.diagnostics.get('runtime_context_sha'))] = result
        return result

    def complete(self, values, hardware, operator, shape, *, count=1, rng=None, budget=None, exclude_ids=()):
        from .space_tools import construct
        if type(count) is not int or count < 1:
            raise ValueError('completion count must be positive')
        return construct(self, hardware, operator, shape, partial=values, count=count, rng=rng,
                         budget=budget, exclude_ids=exclude_ids)

    def sample_feasible(self, rng, count, hardware, operator, shape, *, budget=None, exclude_ids=()):
        from .space_tools import SpaceBuildResult
        runtime_identity = getattr(self, '_runtime_context_identity', None)
        try:runtime_sha = stable_id(runtime_identity()) if runtime_identity is not None else None
        except Exception as exc:
            return SpaceBuildResult((), 'unsupported', {'reason': str(exc), 'runtime_context_changed': True})
        cached = self._feasible_cache.get((hardware.id, operator.id, shape.id, self.fingerprint, runtime_sha))
        if cached is not None:
            support = getattr(self, '_construction_support', None)
            try:reason = support(hardware, operator, shape) if support is not None else None
            except Exception as exc:reason = str(exc)
            if reason:return SpaceBuildResult((), 'unsupported', {'reason': reason})
            excluded = set(exclude_ids)
            candidates = [c for c in cached.configs if c.id not in excluded]
            sample = tuple(rng.sample(candidates, min(count, len(candidates))))
            return SpaceBuildResult(sample, 'sampled' if len(sample) == count else 'exhausted',
                                    {'distribution': 'uniform without replacement over the complete static feasible set', 'complete_set_size': len(cached.configs)})
        return self.complete({}, hardware, operator, shape, count=count, rng=rng, budget=budget, exclude_ids=exclude_ids)

    def assess(self, config, hardware, operator, shape):
        result = []
        for callback in self.advisory_callbacks:
            try:
                value = callback(config, hardware, operator, shape)
                if value is not None: result.append(value)
            except Exception as exc:
                result.append({'kind': 'advisory_error', 'message': type(exc).__name__ + ': ' + str(exc), 'excludes_candidate': False})
        return result

    @property
    def capabilities(self):
        return {'conditional': getattr(self, 'conditional', False), 'native_candidates': self.native_configs is not None,
                'partial_rules': [r.to_dict() for r in self.rules], 'opaque_final_constraints': len(self.constraints)-len(self.rules),
                'normalization': 'declared inactive defaults only', 'sampling': 'uniform over complete cached set; otherwise dependency-aware nonuniform construction'}

    def legal_configs(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec) -> list[KernelConfig]:
        return [c for c in self.iter_configs() if self.validate(c, hardware, operator, shape).valid]

    def neighbors(self, config: KernelConfig, *, max_changed=1, min_changed=1, max_neighbors=4096) -> list[KernelConfig]:
        if type(max_changed) is not int or type(min_changed) is not int or not 1 <= min_changed <= max_changed or type(max_neighbors) is not int or max_neighbors < 1:
            raise ValueError('positive neighbor limits required')
        result = []; seen = {config.id}
        if self.native_configs is not None:
            return [deepcopy(c) for c in self.native_configs if c.id != config.id and
                    min_changed <= sum(canonical_json(c.get(k)) != canonical_json(config.get(k)) for k in self.domains) <= max_changed][:max_neighbors]
        for degree in range(min_changed, min(max_changed, len(self.domains)) + 1):
            for keys in combinations(self.domains, degree):
                changed_domains = [tuple(v for v in self.domains[k] if canonical_json(v) != canonical_json(config.get(k))) for k in keys]
                for values in product(*changed_domains):
                    candidate = self.normalize(KernelConfig({**config.params, **dict(zip(keys, values))}))
                    changed = sum(canonical_json(candidate.get(k)) != canonical_json(config.get(k)) for k in self.domains)
                    if candidate.id not in seen and min_changed <= changed <= max_changed:
                        result.append(candidate); seen.add(candidate.id)
                        if len(result) >= max_neighbors:
                            return result
        return result


def gemm_constraint(config: KernelConfig, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec) -> str | None:
    if operator.kind not in {"gemm", "batched_gemm"} or not {"M", "N", "K"} <= shape.dims.keys():
        return "GEMM requires gemm/batched_gemm and M/N/K dimensions"
    if operator.kind == "batched_gemm" and "Batch" not in shape.dims:
        return "batched GEMM requires an explicit Batch dimension"
    allowed_dims = {"M", "N", "K", "Batch"} if operator.kind == "batched_gemm" else {"M", "N", "K"}
    if shape.dims.keys() - allowed_dims:
        return "baseline GEMM accepts M/N/K and Batch only for batched GEMM"
    if operator.parameters:
        return "baseline GEMM has no transpose, fusion, broadcasting, or other operator parameters"
    if operator.dtype not in {"float16", "bfloat16"}:
        return "baseline GEMM supports float16 and bfloat16"
    if operator.dtype == "bfloat16" and hardware.architecture.startswith("sm_"):
        if int(hardware.architecture[3:]) < 80:
            return "bfloat16 requires NVIDIA compute capability >= 8.0"
    bm, bn, bk = (config.get(k) for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
    stages, warps = config.get("num_stages"), config.get("num_warps")
    if any(not isinstance(v, int) or v <= 0 or v & (v - 1) for v in (bm, bn, bk, warps)):
        return "tiles and warp count must be positive powers of two"
    if not isinstance(stages, int) or stages <= 0:
        return "pipeline stages must be positive"
    if warps * hardware.warp_size > hardware.max_threads_per_block:
        return "warp count exceeds hardware thread limit"
    estimate = (bm * bk + bk * bn) * 2 * stages
    if hardware.smem_per_block and estimate > hardware.smem_per_block:
        return f"conservative shared-memory estimate {estimate} exceeds {hardware.smem_per_block} bytes"
    return None


def gemm_space(small: bool = False) -> ConfigSpace:
    return ConfigSpace({
        "BLOCK_M": (16, 32, 64) if small else (16, 32, 64, 128),
        "BLOCK_N": (32, 64) if small else (32, 64, 128),
        "BLOCK_K": (32,) if small else (32, 64),
        "num_warps": (4, 8),
        "num_stages": (2, 3) if small else (2, 3, 4),
    }, constraints=(gemm_constraint,), name="triton_gemm_v1")


def reduction_constraint(config: KernelConfig, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec) -> str | None:
    if operator.kind not in {"softmax", "layernorm"} or set(shape.dims) != {"M", "N"}:
        return "row reduction requires softmax/layernorm and M/N dimensions"
    if operator.dtype not in {"float16", "bfloat16", "float32"}:
        return "row reductions support float16, bfloat16, and float32"
    block = config.get("BLOCK_SIZE")
    warps, stages = config.get("num_warps"), config.get("num_stages")
    if isinstance(block, bool) or not isinstance(block, int) or block <= 0 or block & (block - 1):
        return "BLOCK_SIZE must be a positive power of two"
    if block < shape.get("N"):
        return "BLOCK_SIZE must cover the complete row N"
    if block > 65536:
        return "v0.1 row reduction supports BLOCK_SIZE <= 65536"
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (warps, stages)) or warps not in (4, 8) or stages not in (1, 2):
        return "row reduction supports num_warps 4/8 and num_stages 1/2"
    if warps * hardware.warp_size > hardware.max_threads_per_block:
        return "warp count exceeds hardware thread limit"
    allowed_parameters = {"axis"} if operator.kind == "softmax" else {"axis", "eps", "affine", "elementwise_affine"}
    if operator.parameters.keys() - allowed_parameters:
        return "baseline reduction received unsupported operator parameters"
    if operator.kind == "layernorm" and (operator.parameters.get("affine", False) or operator.parameters.get("elementwise_affine", False)):
        return "v0.1 layernorm is non-affine"
    axis = operator.parameters.get("axis", -1)
    if isinstance(axis, bool) or not isinstance(axis, int) or axis not in (-1, 1):
        return "only row-wise last-axis reductions are supported"
    if operator.kind == "layernorm":
        import math
        eps = operator.parameters.get("eps", 1e-5)
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            return "layernorm eps must be finite and positive"
    return None
