"""Portable experiment specifications and explicitly fictional demo hardware."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .models import HardwareSpec, OperatorSpec, SearchBudget, ShapeSpec, canonical_json
from .space import ConfigSpace, gemm_constraint, gemm_space, reduction_constraint


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    backend: str
    operator: OperatorSpec
    shapes: tuple[ShapeSpec, ...]
    domains: dict[str, tuple[Any, ...]]
    hardware: HardwareSpec | None = None
    budget: SearchBudget = field(default_factory=lambda: SearchBudget(32))
    seed: int = 0
    warmup: int = 2
    repetitions: int = 7
    split: str = "train"

    def __post_init__(self) -> None:
        if not self.name or not self.backend:
            raise ValueError("experiment name and backend are required")
        if not self.shapes or len({shape.id for shape in self.shapes}) != len(self.shapes):
            raise ValueError("experiment must contain nonempty, distinct shapes")
        if not isinstance(self.warmup, int) or self.warmup < 0 or not isinstance(self.repetitions, int) or self.repetitions < 1:
            raise ValueError("warmup must be nonnegative and repetitions positive integers")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.config_space()

    def config_space(self) -> ConfigSpace:
        constraints = ((gemm_constraint,) if self.operator.kind in {"gemm", "batched_gemm"} else
                       (reduction_constraint,) if self.operator.kind in {"softmax", "layernorm"} else ())
        return ConfigSpace(self.domains, constraints=constraints, name=self.operator.implementation)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Preserve parameter ordering across write/load: it defines Grid traversal.
        target.write_text(json.dumps({"schema_version": 1, **asdict(self)}, ensure_ascii=False,
                                     indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSpec:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.pop("schema_version", 1) != 1:
            raise ValueError("unsupported experiment schema version")
        raw["operator"] = OperatorSpec(**raw["operator"])
        raw["shapes"] = tuple(ShapeSpec(**s) for s in raw["shapes"])
        raw["domains"] = {k: tuple(v) for k, v in raw["domains"].items()}
        raw["budget"] = SearchBudget(**raw.get("budget", {}))
        if raw.get("hardware"):
            raw["hardware"] = HardwareSpec(**raw["hardware"])
        return cls(**raw)


def demo_hardware() -> tuple[HardwareSpec, ...]:
    return (
        HardwareSpec("MOCK-compute-rich", "mock_v1", sm_count=80, smem_per_block=65536,
                     registers_per_sm=65536, compute_tflops=120, bandwidth_gbps=800, l2_bytes=32 * 1024**2,
                     features=("synthetic",), metadata={"fictional": True}),
        HardwareSpec("MOCK-bandwidth-rich", "mock_v2", sm_count=40, smem_per_block=16384,
                     registers_per_sm=65536, compute_tflops=24, bandwidth_gbps=1600, l2_bytes=16 * 1024**2,
                     features=("synthetic",), metadata={"fictional": True}),
        HardwareSpec("MOCK-unseen-architecture", "mock_v3", sm_count=60, smem_per_block=49152,
                     registers_per_sm=65536, compute_tflops=60, bandwidth_gbps=1200, l2_bytes=24 * 1024**2,
                     features=("synthetic",), metadata={"fictional": True}),
    )


def demo_shapes() -> tuple[ShapeSpec, ...]:
    return tuple(ShapeSpec({"M": m, "N": n, "K": k}) for m, n, k in (
        (1, 256, 256), (16, 512, 256), (32, 256, 512), (64, 512, 512),
        (128, 256, 512), (256, 512, 512), (384, 512, 256), (512, 1024, 512),
    ))


def example_spec(backend: str = "mock", operator_name: str = "gemm", dtype: str = "float16") -> ExperimentSpec:
    from .operators import operator_space, operator_spec
    shapes = (ShapeSpec({"M": 64, "N": 128, "K": 96}), ShapeSpec({"M": 127, "N": 129, "K": 65}))
    if operator_name == "batched_gemm":
        shapes = tuple(ShapeSpec({**shape.dims, "Batch": i + 2}) for i, shape in enumerate(shapes))
    elif operator_name in {"softmax", "layernorm"}:
        shapes = (ShapeSpec({"M": 7, "N": 193}), ShapeSpec({"M": 33, "N": 1025}))
    actual_backend = backend if backend == "mock" or operator_name == "gemm" else f"triton_{operator_name}"
    domains = operator_space(operator_name, small=True, shapes=shapes).domains
    if operator_name in {"softmax", "layernorm"}:
        # A smoke run should verify both row lengths within eight attempts.
        domains = {**domains, "BLOCK_SIZE": (max(domains["BLOCK_SIZE"]),)}
    return ExperimentSpec(name=f"{backend}-{operator_name}-{dtype}-smoke-v1", backend=actual_backend,
                          operator=operator_spec(operator_name, backend, dtype),
                          shapes=shapes,
                          domains=domains,
                          hardware=demo_hardware()[0] if backend == "mock" else None,
                          budget=SearchBudget(8, 120.0))
