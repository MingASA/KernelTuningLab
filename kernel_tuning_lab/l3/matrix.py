"""Observed latency tables for a single hardware/operator experiment context."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Iterable

from ..models import KernelConfig, MeasurementRecord, WorkloadDistribution


class MissingMeasurementError(ValueError):
    """A requested execution has no observed, successful latency."""


@dataclass(frozen=True)
class PerformanceMatrix:
    """Rows are workload shapes; columns are candidate configurations.

    ``None`` is unknown, never an inferred latency or an infinite measurement.
    Hand-entered tables default to ``unspecified`` provenance. Predictions must
    live outside this class so that solvers cannot present them as observations.
    """

    workload: WorkloadDistribution
    configs: tuple[KernelConfig, ...]
    latencies_ms: tuple[tuple[float | None, ...], ...]
    provenance: str = "unspecified"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        configs = tuple(self.configs)
        rows = tuple(tuple(row) for row in self.latencies_ms)
        if not configs or len({c.id for c in configs}) != len(configs):
            raise ValueError("matrix configurations must be nonempty and unique")
        if len(rows) != len(self.workload.shapes) or any(len(row) != len(configs) for row in rows):
            raise ValueError("latency matrix dimensions must be shape count x config count")
        if self.provenance not in {"measured", "imported", "mock", "unspecified"}:
            raise ValueError("matrix provenance must be measured, imported, mock, or unspecified; predictions are not measurements")
        for row in rows:
            for latency in row:
                if latency is not None and (
                    isinstance(latency, bool)
                    or not isinstance(latency, (int, float))
                    or not math.isfinite(latency)
                    or latency <= 0
                ):
                    raise ValueError("latencies must be finite positive milliseconds, or None for missing")
        object.__setattr__(self, "configs", configs)
        object.__setattr__(self, "latencies_ms", tuple(
            tuple(None if value is None else float(value) for value in row) for row in rows
        ))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def complete(self) -> bool:
        return all(value is not None for row in self.latencies_ms for value in row)

    @property
    def coverage(self) -> float:
        """Fraction of observed cells; selection coverage instead uses shape weights."""
        count = sum(value is not None for row in self.latencies_ms for value in row)
        return count / (len(self.workload.shapes) * len(self.configs))

    @property
    def shape_coverage(self) -> float:
        return math.fsum(weight for weight, row in zip(self.workload.normalized_weights, self.latencies_ms)
                         if any(value is not None for value in row))

    @property
    def reference_latency_ms(self) -> float | None:
        """Unrestricted per-shape observed optimum, never a prediction of missing cells."""
        terms = []
        for weight, row in zip(self.workload.normalized_weights, self.latencies_ms):
            observed = [value for value in row if value is not None]
            if weight > 0 and not observed:
                return None
            if observed:
                terms.append(weight * min(observed))
        return math.fsum(terms)

    @classmethod
    def from_records(
        cls,
        workload: WorkloadDistribution,
        configs: Iterable[KernelConfig],
        records: Iterable[MeasurementRecord],
        *,
        hardware_id: str | None = None,
        operator_id: str | None = None,
        provenance: str | None = None,
    ) -> PerformanceMatrix:
        """Aggregate successful repetitions by median; deduplicate record IDs.

        Never mix hardware, operators, or synthetic/real sources implicitly.
        Failed attempts and explicitly unverified correctness leave cells missing.
        Filtering occurs before context checks, allowing a shared experiment store.
        """
        configs = tuple(configs)
        shape_indices = {shape.id: i for i, shape in enumerate(workload.shapes)}
        config_indices = {config.id: j for j, config in enumerate(configs)}
        relevant = []
        seen = set()
        for record in records:
            if record.shape.id not in shape_indices or record.config.id not in config_indices:
                continue
            if hardware_id is not None and record.hardware.id != hardware_id:
                continue
            if operator_id is not None and record.operator.id != operator_id:
                continue
            if provenance is not None and record.provenance != provenance:
                continue
            if record.id not in seen:
                seen.add(record.id)
                relevant.append(record)
        contexts = {(record.hardware.id, record.operator.id) for record in relevant}
        if len(contexts) > 1:
            raise ValueError("matrix records span hardware/operator contexts; supply hardware_id and operator_id")
        sources = {record.provenance for record in relevant}
        if len(sources) > 1:
            raise ValueError("matrix records mix provenance; select one source explicitly")
        samples: dict[tuple[int, int], list[float]] = {}
        excluded_unverified = 0
        for record in relevant:
            if record.metadata.get("correctness_verified") is False or (record.successful and not record.correctness_verified):
                excluded_unverified += 1
                continue
            if record.successful:
                key = (shape_indices[record.shape.id], config_indices[record.config.id])
                samples.setdefault(key, []).extend(record.runtime_samples_ms)
        rows = tuple(tuple(statistics.median(samples[i, j]) if (i, j) in samples else None
                           for j in range(len(configs))) for i in range(len(workload.shapes)))
        context = next(iter(contexts), (hardware_id, operator_id))
        source = next(iter(sources), provenance or "unspecified")
        return cls(workload, configs, rows, source, {
            "hardware_id": context[0], "operator_id": context[1],
            "aggregation": "median_of_deduplicated_runtime_samples",
            "record_count": len(relevant), "excluded_unverified_records": excluded_unverified,
            "splits": sorted({record.split for record in relevant}),
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": {"name": self.workload.name,
                         "shapes": [dict(shape.dims) for shape in self.workload.shapes],
                         "weights": list(self.workload.weights)},
            "configs": [dict(config.params) for config in self.configs],
            "latencies_ms": [list(row) for row in self.latencies_ms],
            "provenance": self.provenance, "metadata": dict(self.metadata),
            "coverage": self.coverage, "complete": self.complete,
        }
