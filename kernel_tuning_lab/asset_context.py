"""Resolved, platform-owned context shared by acceptance and research execution."""

from dataclasses import dataclass
from typing import Any

from .models import HardwareSpec, OperatorSpec, ShapeSpec, stable_id
from .space import ConfigSpace


@dataclass
class AssetCase:
    kernel: Any
    operator_definition: Any
    hardware: HardwareSpec
    operator: OperatorSpec
    shape: ShapeSpec
    space: ConfigSpace
    adapter: Any
    validation_only: bool = False
    unavailable_reason: str | None = None

    @property
    def id(self) -> str:
        return stable_id([self.kernel.ref, self.hardware.id, self.operator.id, self.shape.id])

    def manifest(self) -> dict:
        return {"id": self.id, "kernel": self.kernel.ref, "operator": self.operator_definition.ref,
                "hardware": self.hardware.name, "architecture": self.hardware.architecture,
                "shape": dict(self.shape.dims), "dtype": self.operator.dtype,
                "validation_only": self.validation_only, "unavailable_reason": self.unavailable_reason,
                "space": {"name": self.space.name, "domains": self.space.domains,
                          "resolution": getattr(self.space, "resolution", {}),
                          "fingerprint": self.space.fingerprint, "cardinality_upper_bound": self.space.cardinality}}
