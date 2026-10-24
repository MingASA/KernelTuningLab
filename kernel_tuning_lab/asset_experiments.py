"""Version 2 experiment files select registered assets, never implementation glue."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from .models import SearchBudget, ShapeSpec


def _keys(value: Mapping, allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(sorted(unknown))}")


@dataclass(frozen=True)
class KernelSelection:
    asset: str
    space: dict[str, tuple[Any, ...]] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.asset, str) or not self.asset.strip():
            raise ValueError("kernel asset reference is required")
        if not isinstance(self.space, dict):
            raise ValueError("kernel space restrictions must be a mapping")
        for name, values in self.space.items():
            if not isinstance(name, str) or not name or not isinstance(values, (tuple, list)) or not values:
                raise ValueError("space restrictions require named, nonempty finite value lists")


@dataclass(frozen=True)
class AlgorithmSelection:
    asset: str
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.asset, str) or not self.asset.strip():
            raise ValueError("algorithm asset reference is required")
        if not isinstance(self.options, dict) or any(not isinstance(k, str) for k in self.options):
            raise ValueError("algorithm options must be a mapping with string keys")
        if "seed" in self.options:
            raise ValueError("set the experiment seed instead of overriding algorithm.options.seed")


@dataclass(frozen=True)
class ValidationBudget:
    """Whole-experiment acceptance budget; research budgets are per combination."""

    max_configs: int = 8
    max_cases: int = 2
    max_wall_time_s: float = 30.0
    seed: int = 17
    trust_claims: bool = True

    def __post_init__(self):
        for key in ("max_configs", "max_cases", "seed"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"validation.{key} must be a nonnegative integer")
        SearchBudget(0, self.max_wall_time_s)
        if self.max_wall_time_s is None:
            raise ValueError("validation requires a finite hard wall deadline")
        if not isinstance(self.trust_claims, bool):
            raise ValueError("validation.trust_claims must be boolean")


@dataclass(frozen=True)
class AssetExperiment:
    name: str
    operator: str
    kernels: tuple[KernelSelection, ...]
    algorithms: tuple[AlgorithmSelection, ...]
    shapes: tuple[ShapeSpec, ...]
    plugins: tuple[str, ...] = ()
    dtype: str = "float32"
    parameters: dict[str, Any] = field(default_factory=dict)
    hardware: str | None = None
    budget: SearchBudget = field(default_factory=lambda: SearchBudget(8, 30.0))
    validation: ValidationBudget = field(default_factory=ValidationBudget)
    preflight_seconds: float = 30.0
    seed: int = 0
    warmup: int = 2
    repetitions: int = 7
    split: str = "train"
    schema_version: int = 2
    workspace_memory_limit_bytes: int = 256 * 1024**2

    def __post_init__(self):
        if self.schema_version != 2:
            raise ValueError("asset experiments require schema_version=2; v1 remains supported by benchmark")
        if not isinstance(self.name, str) or not self.name.strip() or not isinstance(self.operator, str) or not self.operator:
            raise ValueError("experiment name and operator reference are required")
        if not self.kernels or not self.algorithms or not self.shapes:
            raise ValueError("select at least one kernel, algorithm and shape")
        if len({s.id for s in self.shapes}) != len(self.shapes):
            raise ValueError("experiment shapes must be distinct")
        if len({k.asset for k in self.kernels}) != len(self.kernels) or len({a.asset for a in self.algorithms}) != len(self.algorithms):
            raise ValueError("kernel/algorithm asset selections must be distinct")
        if any(not isinstance(p, str) or not p for p in self.plugins):
            raise ValueError("plugins must be explicit nonempty file/module references")
        if not isinstance(self.dtype, str) or not self.dtype or not isinstance(self.parameters, dict):
            raise ValueError("dtype and operator parameter mapping are required")
        for key, minimum in (("seed", 0), ("warmup", 0), ("repetitions", 1)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if (isinstance(self.workspace_memory_limit_bytes, bool) or not isinstance(self.workspace_memory_limit_bytes, int)
                or self.workspace_memory_limit_bytes <= 0):
            raise ValueError("workspace_memory_limit_bytes must be a positive integer")
        SearchBudget(0, self.preflight_seconds)
        if self.preflight_seconds is None:
            raise ValueError("preflight requires a finite hard wall deadline")
        if self.budget.max_wall_time_s is None:
            raise ValueError("asset searches require a finite wall deadline")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AssetExperiment:
        if not isinstance(value, dict):
            raise ValueError("experiment must be a JSON object")
        _keys(value, set(cls.__dataclass_fields__), "experiment")
        raw = dict(value)
        for field_name, selection_type, allowed in (
            ("kernels", KernelSelection, {"asset", "space"}),
            ("algorithms", AlgorithmSelection, {"asset", "options"}),
        ):
            values = raw.get(field_name, ())
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"{field_name} must be a list of asset references")
            selections = []
            for item in values:
                item = {"asset": item} if isinstance(item, str) else item
                if not isinstance(item, dict):
                    raise ValueError(f"{field_name} entries must be strings or objects")
                _keys(item, allowed, field_name)
                selections.append(selection_type(**item))
            raw[field_name] = tuple(selections)
        shapes = raw.get("shapes", ())
        if not isinstance(shapes, (list, tuple)):
            raise ValueError("shapes must be a list")
        if any(not isinstance(shape, dict) for shape in shapes):
            raise ValueError("each shape must be a dimension mapping")
        raw["shapes"] = tuple(ShapeSpec(**s) if set(s) == {"dims"} else ShapeSpec(s) for s in shapes)
        plugins = raw.get("plugins", ())
        if not isinstance(plugins, (list, tuple)):
            raise ValueError("plugins must be a list")
        raw["plugins"] = tuple(plugins)
        raw["budget"] = SearchBudget(**{"max_evaluations": 8, "max_wall_time_s": 30.0, **raw.get("budget", {})})
        raw["validation"] = ValidationBudget(**raw.get("validation", {}))
        return cls(**raw)

    @classmethod
    def load(cls, path: str | Path) -> AssetExperiment:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
