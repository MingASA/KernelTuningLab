"""Minimal extension contract for Triton, TileLang, and other kernel implementations."""

from __future__ import annotations

from typing import Any, Protocol

from ..models import HardwareSpec, KernelConfig, OperatorSpec, ShapeSpec
from ..space import ValidationResult


class BackendUnavailableError(RuntimeError):
    """A required runtime or physical device is absent."""


class KernelAdapter(Protocol):
    name: str
    provenance: str
    measurement_mode: str

    def environment(self) -> dict[str, Any]: ...
    def protocol_metadata(self) -> dict[str, Any]: ...
    def validate(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec, config: KernelConfig) -> ValidationResult: ...
    def prepare(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec, seed: int) -> Any: ...
    def compile(self, prepared: Any, config: KernelConfig) -> Any: ...
    def check_correctness(self, compiled: Any) -> tuple[bool, dict[str, Any]]: ...
    def warmup(self, compiled: Any, iterations: int) -> None: ...
    def measure(self, compiled: Any, repetitions: int) -> tuple[float, ...]: ...

