"""Operator-specific finite configuration spaces with explicit legality results."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
import random
from typing import Any, Callable, Iterator, Mapping

from .models import HardwareSpec, KernelConfig, OperatorSpec, ShapeSpec, canonical_json, stable_id


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: tuple[str, ...] = ()


Constraint = Callable[[KernelConfig, HardwareSpec, OperatorSpec, ShapeSpec], str | None]


class ConfigSpace:
    def __init__(self, domains: Mapping[str, tuple[Any, ...] | list[Any]], constraints: tuple[Constraint, ...] = (), name: str = "custom", identity: Any = None, *, allow_empty: bool = False):
        self.domains = {k: tuple(v) for k, v in domains.items()}
        self.constraints = tuple(constraints)
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

    def __len__(self) -> int:
        return self.cardinality

    @property
    def cardinality(self) -> int:
        return prod(len(v) for v in self.domains.values())

    @property
    def fingerprint(self) -> str:
        """Compact protocol identity; avoid copying huge domains into every record."""
        from .assets import callable_identity
        constraints = [callable_identity(constraint) for constraint in self.constraints]
        signature = (self.name, tuple((key, id(values)) for key, values in self.domains.items()),
                     canonical_json(constraints), canonical_json(self.identity))
        if self._fingerprint_cache is None or self._fingerprint_cache[0] != signature:
            self._fingerprint_cache = (signature, stable_id({"name": self.name, "domains": self.domains,
                                                          "constraints": constraints, "identity": self.identity}))
        return self._fingerprint_cache[1]

    def read_only(self):
        """Return the public algorithm facade without copying the search space."""
        from .asset_space import SpaceView
        return SpaceView(self)

    def config_at(self, index: int) -> KernelConfig:
        """Decode one Cartesian index in O(number of dimensions), without enumeration."""
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.cardinality:
            raise IndexError("configuration index outside the finite space")
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
        if not reasons:
            for constraint in self.constraints:
                try:
                    reason = constraint(config, hardware, operator, shape)
                except Exception as exc:
                    reason = f"constraint error: {type(exc).__name__}: {exc}"
                if reason:
                    reasons.append(reason)
        return ValidationResult(not reasons, tuple(reasons))

    def legal_configs(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec) -> list[KernelConfig]:
        return [c for c in self.iter_configs() if self.validate(c, hardware, operator, shape).valid]

    def neighbors(self, config: KernelConfig) -> list[KernelConfig]:
        result = []
        for key, values in self.domains.items():
            for value in values:
                if canonical_json(value) != canonical_json(config.get(key)):
                    result.append(KernelConfig({**config.params, key: value}))
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
