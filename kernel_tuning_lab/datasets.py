"""Case-level splits; never partition configurations from one case across sets."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Iterable

from .models import MeasurementRecord, stable_id


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[MeasurementRecord, ...]
    validation: tuple[MeasurementRecord, ...]
    test: tuple[MeasurementRecord, ...]
    seed: int = 0
    holdout: str = "case"

    def __post_init__(self) -> None:
        if self.holdout not in {"case", "shape", "hardware"}:
            raise ValueError("holdout must be case, shape, or hardware")
        seen: dict[str, str] = {}
        for split in ("train", "validation", "test"):
            for record in getattr(self, split):
                if record.split != split:
                    raise ValueError(f"{split} contains a record labelled {record.split}")
                key = _group_key(record, self.holdout)
                if key in seen and seen[key] != split:
                    raise ValueError(f"{self.holdout} leaks between {seen[key]} and {split}")
                seen[key] = split

    def manifest(self) -> dict:
        return {
            "seed": self.seed,
            "holdout": self.holdout,
            "splits": {
                split: {
                    "records": len(getattr(self, split)),
                    "case_ids": sorted({r.case_id for r in getattr(self, split)}),
                    "hardware_ids": sorted({r.hardware.id for r in getattr(self, split)}),
                    "shape_ids": sorted({r.shape.id for r in getattr(self, split)}),
                }
                for split in ("train", "validation", "test")
            },
        }


def _group_key(record: MeasurementRecord, holdout: str) -> str:
    if holdout == "hardware":
        return record.hardware.id
    if holdout == "shape":
        return record.shape.id
    return record.case_id


def split_records(
    records: Iterable[MeasurementRecord],
    seed: int = 0,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    holdout: str = "case",
) -> DatasetSplits:
    """Deterministic grouping supports unseen-shape and unseen-GPU evaluation.

    Existing labels are intentionally replaced: this function creates a new split.
    Use ``validate_split_isolation`` to validate previously assigned labels.
    """
    if holdout not in {"case", "shape", "hardware"}:
        raise ValueError("holdout must be case, shape, or hardware")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (train_fraction, validation_fraction)):
        raise ValueError("split fractions must be finite and in [0, 1]")
    if train_fraction + validation_fraction > 1:
        raise ValueError("train and validation fractions must sum to at most one")
    rows = tuple(records)
    groups = sorted({_group_key(r, holdout) for r in rows})
    random.Random(seed).shuffle(groups)
    fractions = (train_fraction, validation_fraction, 1 - train_fraction - validation_fraction)
    counts = [int(len(groups) * fraction) for fraction in fractions]
    remainder = len(groups) - sum(counts)
    order = sorted(range(3), key=lambda i: (-(len(groups) * fractions[i] - counts[i]), i))
    for index in order[:remainder]:
        counts[index] += 1
    # Keep each requested set useful when there are enough independent groups.
    active = [i for i, fraction in enumerate(fractions) if fraction > 0]
    if len(groups) >= len(active):
        for index in active:
            if counts[index] == 0:
                donor = max(active, key=lambda i: counts[i])
                counts[donor] -= 1
                counts[index] += 1
    assignments: dict[str, str] = {}
    cursor = 0
    for name, count in zip(("train", "validation", "test"), counts):
        assignments.update((key, name) for key in groups[cursor:cursor + count])
        cursor += count
    partitions: dict[str, list[MeasurementRecord]] = {name: [] for name in ("train", "validation", "test")}
    for record in rows:
        split = assignments[_group_key(record, holdout)]
        partitions[split].append(replace(record, split=split))
    return DatasetSplits(*(tuple(partitions[name]) for name in ("train", "validation", "test")), seed=seed, holdout=holdout)


def validate_split_isolation(records: Iterable[MeasurementRecord], holdout: str = "case") -> DatasetSplits:
    rows = tuple(records)
    return DatasetSplits(*(tuple(r for r in rows if r.split == name)
                           for name in ("train", "validation", "test")), holdout=holdout)


def require_training_records(
    records: Iterable[MeasurementRecord],
    forbidden_case_ids: Iterable[str] = (),
    forbidden_hardware_ids: Iterable[str] = (),
    forbidden_shape_ids: Iterable[str] = (),
) -> tuple[MeasurementRecord, ...]:
    """Reject accidental fitting on validation/test or a held-out identity."""
    rows = tuple(records)
    cases, hardware, shapes = map(set, (forbidden_case_ids, forbidden_hardware_ids, forbidden_shape_ids))
    for record in rows:
        if record.split != "train":
            raise ValueError("fitting/transfer data must contain train records only")
        if record.case_id in cases or record.hardware.id in hardware or record.shape.id in shapes:
            raise ValueError("training data overlaps a held-out evaluation group")
    return rows


def case_id(hardware, operator, shape) -> str:
    return stable_id([hardware.id, operator.id, shape.id])
