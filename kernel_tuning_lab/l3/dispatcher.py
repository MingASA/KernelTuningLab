"""Compile finite-workload selection results into constant-time table dispatch."""

from __future__ import annotations

from typing import Any

from ..models import HardwareSpec, KernelConfig, OperatorSpec, ShapeSpec
from ..space import ConfigSpace
from .matrix import MissingMeasurementError, PerformanceMatrix
from .solver import SelectionResult


class UnknownShapeError(KeyError):
    """The finite dispatcher has no observed assignment for an unseen shape."""


def _shape_key(shape: ShapeSpec) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(shape.dims.items()))


class ShapeDispatcher:
    """Exact shape lookup: no latency scan, model inference, or compilation at runtime."""

    def __init__(self, assignments: dict[tuple[tuple[str, int], ...], KernelConfig],
                 *, selected_configs: tuple[KernelConfig, ...], provenance: str):
        selected_ids = {config.id for config in selected_configs}
        if any(config.id not in selected_ids for config in assignments.values()):
            raise ValueError("dispatcher assignment is outside its selected variant set")
        self._assignments = dict(assignments)
        self.selected_configs = tuple(selected_configs)
        self.provenance = provenance

    @classmethod
    def from_result(
        cls, matrix: PerformanceMatrix, result: SelectionResult, *,
        hardware: HardwareSpec | None = None,
        operator: OperatorSpec | None = None,
        space: ConfigSpace | None = None,
    ) -> ShapeDispatcher:
        supplied = (hardware is not None, operator is not None, space is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("hardware, operator, and space must be provided together for legality checking")
        if len(result.selected_indices) > result.k or len(set(result.selected_indices)) != len(result.selected_indices):
            raise ValueError("selection violates the finite variant budget")
        if any(j < 0 or j >= len(matrix.configs) for j in result.selected_indices):
            raise ValueError("selection contains an unknown config column")
        if tuple(matrix.configs[j].id for j in result.selected_indices) != tuple(config.id for config in result.configs):
            raise ValueError("selection config identities do not match the performance matrix")
        if len(result.assignment_indices) != len(matrix.workload.shapes):
            raise ValueError("selection assignment does not cover the matrix shape dimension")
        if result.provenance != matrix.provenance:
            raise ValueError("selection provenance does not match the performance matrix")
        known_shapes = result.metadata.get("shape_ids")
        if known_shapes is not None and list(known_shapes) != [shape.id for shape in matrix.workload.shapes]:
            raise ValueError("selection shape identities do not match the performance matrix")
        if hardware is not None and matrix.metadata.get("hardware_id") not in {None, hardware.id}:
            raise ValueError("dispatcher hardware differs from the measured hardware")
        if operator is not None and matrix.metadata.get("operator_id") not in {None, operator.id}:
            raise ValueError("dispatcher operator differs from the measured operator")
        selected = set(result.selected_indices)
        assignments = {}
        for i, (shape, j) in enumerate(zip(matrix.workload.shapes, result.assignment_indices)):
            if j is None:
                continue
            if j not in selected or matrix.latencies_ms[i][j] is None:
                raise MissingMeasurementError("dispatcher assignment must be selected and have an observed latency")
            config = matrix.configs[j]
            if space is not None:
                validity = space.validate(config, hardware, operator, shape)
                if not validity.valid:
                    raise ValueError(f"illegal dispatch for shape {shape.dims}: {'; '.join(validity.reasons)}")
            assignments[_shape_key(shape)] = config
        return cls(assignments, selected_configs=result.configs, provenance=matrix.provenance)

    def select_config(self, shape: ShapeSpec) -> KernelConfig:
        try:
            return self._assignments[_shape_key(shape)]
        except KeyError:
            raise UnknownShapeError(f"no observed dispatch assignment for shape {shape.dims}") from None

    __call__ = select_config

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "exact_shape_table", "provenance": self.provenance,
                "selected_configs": [dict(config.params) for config in self.selected_configs],
                "assignments": [{"shape": dict(key), "config_id": config.id,
                                 "config": dict(config.params)}
                                for key, config in sorted(self._assignments.items())]}
