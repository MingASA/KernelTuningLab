"""Shared, JSON-serializable contracts. All durations explicitly state their unit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
import statistics
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def stable_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()[:24]


class RecordStatus(str, Enum):
    SUCCESS = "success"
    INVALID_CONFIG = "invalid_config"
    COMPILE_ERROR = "compile_error"
    CORRECTNESS_ERROR = "correctness_error"
    RUNTIME_ERROR = "runtime_error"
    CORRECTNESS_UNKNOWN = "correctness_unknown"


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    architecture: str = "unknown"
    sm_count: int = 0
    smem_per_block: int = 0
    registers_per_sm: int = 0
    compute_tflops: float | None = None
    bandwidth_gbps: float | None = None
    l2_bytes: int | None = None
    memory_bytes: int | None = None
    warp_size: int = 32
    max_threads_per_block: int = 1024
    features: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or any(not isinstance(v, int) or isinstance(v, bool) or v < 0
                                for v in (self.sm_count, self.smem_per_block, self.registers_per_sm)):
            raise ValueError("hardware name is required and resource limits must be nonnegative integers")
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (self.warp_size, self.max_threads_per_block)):
            raise ValueError("thread and warp limits must be positive")
        for value in (self.compute_tflops, self.bandwidth_gbps, self.l2_bytes, self.memory_bytes):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("known hardware capacities must be finite and positive")

    @property
    def id(self) -> str:
        return stable_id(asdict(self))


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    kind: str = "gemm"
    implementation: str = "mock_gemm"
    dtype: str = "float16"
    parameters: dict[str, Any] = field(default_factory=dict)
    correctness_baseline: str = "torch.matmul"
    version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return stable_id(asdict(self))


@dataclass(frozen=True)
class ShapeSpec:
    dims: dict[str, int]

    def __post_init__(self) -> None:
        if not self.dims or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in self.dims.values()):
            raise ValueError("shape dimensions must be positive integers")

    @property
    def id(self) -> str:
        return stable_id(self.dims)

    def get(self, key: str, default: int = 1) -> int:
        return self.dims.get(key, default)


@dataclass(frozen=True)
class KernelConfig:
    params: dict[str, Any]

    def __post_init__(self) -> None:
        canonical_json(self.params)

    @property
    def id(self) -> str:
        return stable_id(self.params)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


@dataclass(frozen=True)
class MeasurementRecord:
    hardware: HardwareSpec
    operator: OperatorSpec
    shape: ShapeSpec
    config: KernelConfig
    status: RecordStatus
    compile_time_ms: float = 0.0
    runtime_samples_ms: tuple[float, ...] = ()
    wall_time_s: float = 0.0
    warmup: int = 0
    repetitions: int = 0
    measurement_mode: str = "unspecified"
    environment: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    provenance: str = "measured"
    split: str = "train"
    seed: int = 0
    experiment_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RecordStatus(self.status))
        object.__setattr__(self, "runtime_samples_ms", tuple(self.runtime_samples_ms))
        if self.provenance not in {"mock", "measured", "imported"}:
            raise ValueError("measurement provenance must be mock, measured, or imported; predictions are not measurements")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if self.warmup < 0 or self.repetitions < 0:
            raise ValueError("warmup and repetitions must be nonnegative")
        if any(not math.isfinite(v) or v < 0 for v in (self.compile_time_ms, self.wall_time_s)):
            raise ValueError("measurement costs must be finite and nonnegative")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in self.runtime_samples_ms):
            raise ValueError("runtime samples must be finite and positive")
        if self.status == RecordStatus.SUCCESS and not self.runtime_samples_ms:
            raise ValueError("successful measurements require runtime samples")
        if self.status != RecordStatus.SUCCESS and not self.failure_reason:
            raise ValueError("failed measurements require a reason")

    @property
    def successful(self) -> bool:
        return self.status == RecordStatus.SUCCESS

    @property
    def correctness_verified(self) -> bool:
        """Distinguish accepted submitter assertions from reference-checked success.

        Older version-1 records keep their original success semantics. Explicit
        unknown/declared evidence never becomes verified through a missing
        top-level convenience flag. Physical GPU verification remains a separate
        field in the correctness evidence (Mock and CPU are not GPU evidence).
        """
        if (not self.successful or self.metadata.get("correctness_verified") is False
                or self.metadata.get("correctness_origin") in {"submitter_claim", "submitter_declared", "unknown", "unverified"}):
            return False
        evidence = self.metadata.get("correctness", {})
        if not isinstance(evidence, dict):
            return False
        return (evidence.get("status") not in {"declared", "failed", "unknown", "unverified", "not_run"}
                and evidence.get("comparison_performed") is not False
                and evidence.get("correctness_origin") not in {"submitter_claim", "submitter_declared", "unknown", "unverified"})

    @property
    def median_ms(self) -> float | None:
        return statistics.median(self.runtime_samples_ms) if self.successful else None

    def quantile_ms(self, q: float) -> float | None:
        if not 0 <= q <= 1:
            raise ValueError("quantile must be in [0, 1]")
        if not self.runtime_samples_ms or not self.successful:
            return None
        values = sorted(self.runtime_samples_ms)
        index = (len(values) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    @property
    def case_id(self) -> str:
        return stable_id([self.hardware.id, self.operator.id, self.shape.id])

    @property
    def cache_key(self) -> str:
        return stable_id([self.hardware.id, self.operator.id, self.shape.id, self.config.id,
                          self.warmup, self.repetitions, self.measurement_mode, self.environment,
                          self.provenance, self.seed, self.experiment_id, self.metadata.get("protocol", {})])

    @property
    def id(self) -> str:
        return stable_id(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["schema_version"] = 1
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MeasurementRecord:
        raw = dict(data)
        if raw.pop("schema_version", 1) != 1:
            raise ValueError("unsupported measurement schema version")
        raw["hardware"] = HardwareSpec(**raw["hardware"])
        raw["operator"] = OperatorSpec(**raw["operator"])
        raw["shape"] = ShapeSpec(**raw["shape"])
        raw["config"] = KernelConfig(**raw["config"])
        return cls(**raw)


@dataclass(frozen=True)
class WorkloadDistribution:
    shapes: tuple[ShapeSpec, ...]
    weights: tuple[float, ...]
    name: str = "workload"

    def __post_init__(self) -> None:
        if not self.shapes or len(self.shapes) != len(self.weights):
            raise ValueError("workload needs a weight for each shape")
        if len({s.id for s in self.shapes}) != len(self.shapes):
            raise ValueError("workload shapes must be unique")
        if any(not math.isfinite(w) or w < 0 for w in self.weights) or sum(self.weights) <= 0:
            raise ValueError("weights must be finite, nonnegative, with positive total")

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        scale = max(self.weights)
        scaled = tuple(w / scale for w in self.weights)
        total = math.fsum(scaled)
        return tuple(w / total for w in scaled)


@dataclass(frozen=True)
class SearchBudget:
    max_evaluations: int = 32
    max_wall_time_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_evaluations, int) or isinstance(self.max_evaluations, bool) or self.max_evaluations < 0:
            raise ValueError("max_evaluations must be a nonnegative integer")
        if self.max_wall_time_s is not None and (isinstance(self.max_wall_time_s, bool)
                or not isinstance(self.max_wall_time_s, (int, float))
                or not math.isfinite(self.max_wall_time_s) or self.max_wall_time_s < 0):
            raise ValueError("max_wall_time_s must be finite and nonnegative")

    def exhausted(self, evaluations: int, elapsed_s: float) -> bool:
        return evaluations >= self.max_evaluations or (self.max_wall_time_s is not None and elapsed_s >= self.max_wall_time_s)

    def remaining(self, evaluations: int, elapsed_s: float) -> SearchBudget:
        wall = None if self.max_wall_time_s is None else max(0.0, self.max_wall_time_s - elapsed_s)
        return SearchBudget(max(0, self.max_evaluations - evaluations), wall)
