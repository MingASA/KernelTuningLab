"""Compile, verify, and measure configurations with an append-only result journal."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import time
import traceback
from typing import Iterable

from .backends.base import KernelAdapter
from .models import HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec, RecordStatus, SearchBudget, ShapeSpec, canonical_json
from .space import ConfigSpace
from .storage import ExperimentStore


class BenchmarkRunner:
    def __init__(self, adapter: KernelAdapter, store: ExperimentStore | None = None, warmup: int = 2, repetitions: int = 7,
                 seed: int = 0, experiment_id: str = "default", failure_log_path: str | Path | None = None):
        if warmup < 0 or repetitions < 1:
            raise ValueError("warmup must be nonnegative and repetitions positive")
        self.adapter = adapter
        self.store = store
        self.warmup = warmup
        self.repetitions = repetitions
        self.seed = seed
        self.experiment_id = experiment_id
        self.failure_log_path = Path(failure_log_path) if failure_log_path else None
        self.last_cache_hit = False
        self.last_run_stats: dict[str, object] = {}

    def measure(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec, config: KernelConfig,
                space: ConfigSpace, split: str = "train", resume: bool = True) -> MeasurementRecord:
        started = time.perf_counter()
        self.last_cache_hit = False
        environment = self.adapter.environment()
        protocol = {**self.adapter.protocol_metadata(),
                    "config_space": {"name": space.name, "fingerprint": space.fingerprint,
                                     "cardinality": space.cardinality}}
        # Always revalidate before cache lookup: plugins may tighten constraints.
        validation_error = None
        try:
            validity = space.validate(config, hardware, operator, shape)
            backend_validity = self.adapter.validate(hardware, operator, shape, config)
            current_valid = validity.valid and backend_validity.valid
        except Exception as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
            current_valid = False
        initial = MeasurementRecord(hardware, operator, shape, config, RecordStatus.RUNTIME_ERROR,
                                    warmup=self.warmup, repetitions=self.repetitions,
                                    measurement_mode=self.adapter.measurement_mode, environment=environment,
                                    failure_reason="evaluation has not completed", provenance=self.adapter.provenance,
                                    split=split, seed=self.seed, experiment_id=self.experiment_id,
                                    metadata={"protocol": protocol})
        if resume and self.store:
            cached = self.store.get_cached(initial.cache_key)
            if cached and validation_error is None and ((cached.status != RecordStatus.INVALID_CONFIG) == current_valid):
                if cached.split != split:
                    raise ValueError("cached case belongs to a different dataset split")
                self.last_cache_hit = True
                return cached

        compile_ms = 0.0
        samples: tuple[float, ...] = ()
        metadata = {"protocol": protocol}
        status = RecordStatus.SUCCESS
        reason = None
        phase = "validation"
        try:
            if validation_error:
                raise RuntimeError(validation_error)
            reasons = (*validity.reasons, *backend_validity.reasons)
            if not validity.valid or not backend_validity.valid:
                status, reason = RecordStatus.INVALID_CONFIG, "; ".join(reasons) or "configuration rejected"
            else:
                phase = "preparation"
                prepare_start = time.perf_counter()
                prepared = self.adapter.prepare(hardware, operator, shape, self.seed)
                metadata["preparation_time_ms"] = (time.perf_counter() - prepare_start) * 1000
                phase = "compile"
                compile_start = time.perf_counter()
                try:
                    compiled = self.adapter.compile(prepared, config)
                finally:
                    compile_ms = (time.perf_counter() - compile_start) * 1000
                phase = "correctness"
                correct, details = self.adapter.check_correctness(compiled)
                metadata["correctness"] = details
                if details.get("status") in {"verified", "declared"}:
                    metadata["correctness_verified"] = correct and details["status"] == "verified"
                if not correct:
                    status, reason = RecordStatus.CORRECTNESS_ERROR, "kernel output does not match correctness baseline"
                else:
                    phase = "warmup"
                    self.adapter.warmup(compiled, self.warmup)
                    phase = "measurement"
                    samples = tuple(self.adapter.measure(compiled, self.repetitions))
                    if len(samples) != self.repetitions:
                        raise ValueError("backend returned a different number of samples than requested")
                    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in samples):
                        raise ValueError("backend runtime samples must be finite positive numbers")
        except Exception as exc:
            status = (RecordStatus.COMPILE_ERROR if phase == "compile" else
                      RecordStatus.CORRECTNESS_ERROR if phase == "correctness" else RecordStatus.RUNTIME_ERROR)
            reason = f"{phase}: {type(exc).__name__}: {exc}"
            metadata["traceback"] = traceback.format_exc(limit=8)
            samples = ()
        elapsed = time.perf_counter() - started
        record = replace(initial, status=status, compile_time_ms=compile_ms, runtime_samples_ms=samples,
                         wall_time_s=elapsed, failure_reason=reason, metadata=metadata)
        if self.store:
            self.store.add(record)
        if not record.successful and self.failure_log_path:
            self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.failure_log_path.open("a", encoding="utf-8") as stream:
                stream.write(canonical_json({"record_id": record.id, "status": record.status.value,
                                              "case_id": record.case_id, "config": config.params,
                                              "failure_reason": reason, "provenance": record.provenance}) + "\n")
        return record

    def run(self, hardware: HardwareSpec, operator: OperatorSpec, shapes: Iterable[ShapeSpec], space: ConfigSpace,
            budget: SearchBudget | None = None, split: str = "train", resume: bool = True) -> list[MeasurementRecord]:
        started = time.perf_counter()
        evaluations, reused = 0, 0
        records = []
        exhausted = False
        shape_values = tuple(shapes)
        # Round-robin over shapes so a small smoke budget exercises boundary cases.
        for config in space.iter_configs():
            for shape in shape_values:
                elapsed = time.perf_counter() - started
                if budget and budget.exhausted(evaluations, elapsed):
                    exhausted = True
                    break
                record = self.measure(hardware, operator, shape, config, space, split=split, resume=resume)
                records.append(record)
                reused += int(self.last_cache_hit)
                evaluations += int(not self.last_cache_hit)
            if exhausted:
                break
        elapsed = time.perf_counter() - started
        self.last_run_stats = {"evaluations": evaluations, "cache_hits": reused, "wall_time_s": elapsed,
                               "budget_exhausted": exhausted,
                               "time_budget_overrun": bool(budget and budget.max_wall_time_s is not None and elapsed > budget.max_wall_time_s),
                               "budget_semantics": "admission bound; running kernel compilation cannot be preempted"}
        return records
