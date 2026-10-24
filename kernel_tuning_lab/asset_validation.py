"""Bounded, round-robin physical correctness acceptance for resolved assets."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
import traceback
from typing import Callable

from .acceptance import probe_legal_configs
from .asset_context import AssetCase
from .asset_experiments import ValidationBudget
from .backends.base import BackendUnavailableError
from .models import MeasurementRecord, RecordStatus, canonical_json
from .runner import BenchmarkRunner


def _validation_seeds(case: AssetCase, budget: ValidationBudget) -> tuple[int, ...]:
    seeds = [budget.seed]
    declared = case.operator_definition.metadata.get("validation_seeds", ())
    if not isinstance(declared, (tuple, list)):
        raise ValueError("operator validation_seeds must be a finite list of nonnegative integers")
    for value in declared:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("operator validation_seeds must be nonnegative integers")
        if value not in seeds:
            seeds.append(value)
    return tuple(seeds)


def _sample_seed(case: AssetCase, budget: ValidationBudget, round_index: int) -> int:
    seeds = _validation_seeds(case, budget)
    offset = 0
    if case.validation_only and len(seeds) > 1:
        shapes = list(case.operator_definition.validation_shapes)
        index = next((i for i, shape in enumerate(shapes) if shape.id == case.shape.id), int(case.shape.id[:8], 16))
        offset = 1 + index % (len(seeds) - 1)
    return seeds[(round_index + offset) % len(seeds)]


def validate_cases(cases: list[AssetCase], budget: ValidationBudget, *, store,
                   split: str, experiment_id: str, output_dir: str | Path,
                   progress: Callable[[dict], None] | None = None) -> dict:
    """Validate samples under one whole-experiment attempt allowance.

    Parent supervision supplies the hard process deadline.  This function also
    checks admission time/counts before probing and before every new attempt.
    A passed report means its listed samples passed, never the entire domain.
    """
    started = time.perf_counter()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "validation_report.json"
    records_path = directory / "validation_records.jsonl"
    failure_path = directory / "validation_failures.jsonl"
    records_path.write_text("", encoding="utf-8")
    failure_path.unlink(missing_ok=True)
    contexts = [{"case_id": case.id, "kernel": case.kernel.ref, "operator": case.operator_definition.ref,
                 "shape": dict(case.shape.dims), "dtype": case.operator.dtype,
                 "validation_only": case.validation_only,
                 "status": "deferred" if case.unavailable_reason else "pending",
                 "reason": case.unavailable_reason, "samples": 0, "record_ids": [],
                 "selected_seeds": [], "attempted_config_seed_pairs": []} for case in cases]
    report = {"schema_version": 1, "status": "partial", "reason": "validation_in_progress",
              "budget": {**asdict(budget), "used_configs": 0, "remaining_configs": budget.max_configs},
              "checks": {"attempted": 0, "verified": 0, "declared": 0, "failed": 0},
              "contexts": contexts, "contexts_total": len(contexts), "contexts_checked": 0,
              "samples": [], "coverage": {"scope": "only the listed sampled configurations, shapes and input seeds",
                                             "whole_space_verified": False},
              "artifacts": {"report": str(report_path), "records": str(records_path), "failures": str(failure_path)},
              "notice": "Submitter declarations and CPU checks are distinguished from direct GPU reference comparisons."}
    used = 0
    fatal = False
    budget_stopped = False

    def elapsed():
        return time.perf_counter() - started

    def expired():
        return used >= budget.max_configs or elapsed() >= budget.max_wall_time_s

    def save():
        report["wall_time_s"] = elapsed()
        report["budget"].update({"used_configs": used, "remaining_configs": max(0, budget.max_configs - used),
                                 "used_wall_time_s": report["wall_time_s"]})
        report["contexts_checked"] = sum(context["samples"] > 0 for context in contexts)
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(canonical_json(report) + "\n", encoding="utf-8")
        temporary.replace(report_path)

    def emit(value):
        if progress is not None:
            progress({"phase": "validation", **value})

    # Immediately revoke any previous success marker before calling plugin code.
    save()
    if not cases:
        report.update(status="failed", reason="no validation contexts were resolved")
        save()
        return report
    if len({case.id for case in cases}) != len(cases):
        report.update(status="failed", reason="duplicate validation contexts must be resolved before acceptance")
        save()
        return report

    pools: dict[int, list] = {}
    for index, case in enumerate(cases):
        context = contexts[index]
        if case.unavailable_reason:
            continue
        if expired():
            budget_stopped = True
            break
        emit({"case_id": case.id, "kernel": case.kernel.ref, "shape": dict(case.shape.dims), "operation": "probe_legal_configs"})
        try:
            probe = probe_legal_configs(case.space, case.hardware, case.operator, case.shape,
                                        seed=budget.seed + int(case.id[:8], 16),
                                        count=budget.max_configs, max_inspected=512)
            pool = probe.pop("configs")
            context["space_probe"] = probe
            seeds = _validation_seeds(case, budget)
            # First cover distinct configurations, then spend spare allowance
            # on new input seeds even when a fixed Kernel has only one config.
            samples, seen = [], set()
            for offset in range(len(seeds)):
                for config_index, config in enumerate(pool):
                    seed = _sample_seed(case, budget, config_index + offset)
                    identity = (config.id, seed)
                    if identity not in seen:
                        seen.add(identity)
                        samples.append((config, seed))
                    if len(samples) >= budget.max_configs:
                        break
                if len(samples) >= budget.max_configs:
                    break
            pools[index] = samples
            context["planned_input_seeds"] = sorted({seed for _, seed in samples})
            if not pool:
                if probe.get("exhaustive"):
                    context.update(status="failed", reason="the effective space contains no legal configuration")
                    fatal = True
                else:
                    context.update(status="partial", reason="no legal configuration found within the bounded space probe")
            save()
            if fatal:
                break
        except BackendUnavailableError as exc:
            context.update(status="deferred", reason=str(exc))
            save()
        except Exception as exc:
            context.update(status="failed", reason=f"space probe: {type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=8))
            fatal = True
            save()
            break

    round_index = 0
    while not fatal and any(round_index < len(pool) for pool in pools.values()):
        for index, case in enumerate(cases):
            pool = pools.get(index, ())
            if round_index >= len(pool):
                continue
            if expired():
                budget_stopped = True
                break
            context = contexts[index]
            config, seed = pool[round_index]
            attempt = {"case_id": case.id, "kernel": case.kernel.ref, "operator": case.operator_definition.ref,
                       "shape": dict(case.shape.dims), "dtype": case.operator.dtype,
                       "config": dict(config.params), "seed": seed, "validation_only": case.validation_only}
            emit({**attempt, "operation": "evaluate"})
            used += 1
            if seed not in context["selected_seeds"]:
                context["selected_seeds"].append(seed)
            context["attempted_config_seed_pairs"].append({"config_id": config.id, "seed": seed})
            report["checks"]["attempted"] += 1
            save()
            runner = BenchmarkRunner(case.adapter, store, warmup=0, repetitions=2, seed=seed,
                                     experiment_id=f"{experiment_id}:validation", failure_log_path=failure_path)
            attempt_started = time.perf_counter()
            try:
                pending = MeasurementRecord(case.hardware, case.operator, case.shape, config, RecordStatus.RUNTIME_ERROR,
                                            warmup=0, repetitions=2, seed=seed, split=split,
                                            experiment_id=f"{experiment_id}:validation", provenance="measured",
                                            measurement_mode=case.adapter.measurement_mode,
                                            environment=case.adapter.environment(),
                                            failure_reason="acceptance evaluation has not completed",
                                            metadata={"pending_acceptance": True, "correctness_verified": False})
                emit({**attempt, "operation": "evaluate", "attempt_record": pending.to_dict()})
                record = runner.measure(case.hardware, case.operator, case.shape, config, case.space,
                                        split=split, resume=False)
            except BackendUnavailableError as exc:
                context.update(status="deferred", reason=str(exc))
                pools[index] = []
                attempt.update(status="deferred", reason=str(exc))
                report["samples"].append(attempt)
                emit({**attempt, "operation": "deferred"})
                save()
                continue
            except Exception as exc:
                # Persist failures that occurred before the legacy runner could
                # construct a result, including environment/protocol failures.
                record = MeasurementRecord(case.hardware, case.operator, case.shape, config, RecordStatus.RUNTIME_ERROR,
                                           wall_time_s=time.perf_counter() - attempt_started, warmup=0, repetitions=2,
                                           measurement_mode=getattr(case.adapter, "measurement_mode", "asset_acceptance"),
                                           environment={"runtime": case.kernel.runtime, "kernel_ref": case.kernel.ref},
                                           failure_reason=f"acceptance: {type(exc).__name__}: {exc}", provenance="measured",
                                           split=split, seed=seed, experiment_id=f"{experiment_id}:validation",
                                           metadata={"phase": "acceptance", "traceback": traceback.format_exc(limit=8)})
                if store is not None:
                    try:
                        store.add(record)
                    except Exception as persistence_error:
                        attempt["store_error"] = f"{type(persistence_error).__name__}: {persistence_error}"
                with failure_path.open("a", encoding="utf-8") as stream:
                    stream.write(canonical_json({"record_id": record.id, "failure_reason": record.failure_reason,
                                                 "case_id": record.case_id, "status": record.status.value}) + "\n")
            with records_path.open("a", encoding="utf-8") as stream:
                stream.write(canonical_json(record.to_dict()) + "\n")
            correctness = record.metadata.get("correctness", {})
            evidence = correctness.get("status")
            accepted = record.successful and record.provenance == "measured" and evidence in {"verified", "declared"}
            attempt.update({"status": "passed" if accepted else "failed", "record_status": record.status.value,
                            "record_id": record.id, "record_ids": [record.id], "correctness": correctness,
                            "cache_hit": runner.last_cache_hit,
                            "reason": record.failure_reason if not record.successful else None})
            if record.successful and not accepted:
                attempt["reason"] = "successful timing lacked measured, versioned correctness evidence"
            report["samples"].append(attempt)
            context["samples"] += 1
            context["record_ids"].append(record.id)
            if accepted:
                report["checks"][evidence] += 1
                context.update(status="passed", reason=None)
            else:
                report["checks"]["failed"] += 1
                context.update(status="failed", reason=attempt["reason"] or "kernel acceptance failed")
                fatal = True
            emit({**attempt, "operation": "evaluated", "attempt_record": record.to_dict()})
            save()
            if fatal:
                break
        if budget_stopped:
            break
        round_index += 1

    uncovered = [context for context in contexts if context["status"] in {"pending", "partial"}]
    deferred = [context for context in contexts if context["status"] == "deferred"]
    for context in uncovered:
        context["status"] = "partial"
        context["reason"] = context["reason"] or "no physical sample admitted within the validation allowance"
    if fatal:
        report.update(status="failed", reason="a validation context or sampled execution failed")
    elif uncovered:
        report.update(status="partial", reason="some available contexts lack a physical sample within the bounded validation allowance")
    elif deferred:
        report.update(status="deferred", reason="physical validation requires the unavailable runtime or device")
    else:
        report.update(status="passed", reason="all listed sampled executions passed or were explicitly accepted by scoped submitter declarations")
    report["contexts_passed"] = sum(context["status"] == "passed" for context in contexts)
    report["contexts_deferred"] = len(deferred)
    report["budget_exhausted"] = budget_stopped or expired()
    report["time_budget_overrun"] = elapsed() > budget.max_wall_time_s
    report["coverage"].update({"contexts_checked": report["contexts_checked"], "contexts_total": len(contexts),
                               "configuration_attempts": used, "platform_verified_samples": report["checks"]["verified"],
                               "submitter_declared_samples": report["checks"]["declared"]})
    save()
    return report


__all__ = ["validate_cases"]
