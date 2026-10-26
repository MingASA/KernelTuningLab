"""Platform-owned, budgeted search over independently registered assets."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import time
from typing import Any

from .acceptance import create_searcher, request_candidates
from .datasets import training_record_manifest
from .models import MeasurementRecord, RecordStatus, SearchBudget, canonical_json, stable_id
from .runner import BenchmarkRunner
from .search import SearchHistory


def _correctness(record: MeasurementRecord | None) -> tuple[bool, str | None]:
    if record is None or not record.successful:
        return False, None
    details = record.metadata.get("correctness", {})
    origin = record.metadata.get("correctness_origin", details.get("correctness_origin", "legacy_unspecified"))
    verified = record.correctness_verified
    # Retain the distinction when using a legacy runner which has not promoted
    # the adapter's correctness evidence to top-level metadata yet.
    if details.get("comparison_performed") is False or origin in {"submitter_claim", "submitter_declared"}:
        verified = False
    return verified, origin


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def search_case(case, algorithm_definition, options, budget: SearchBudget, *, store, seed,
                warmup, repetitions, split, experiment_id, output_dir, resume=True, progress=None) -> dict:
    """Search one Kernel/input pair without handing budgets to the implementation.

    Candidate requests, contract failures and new runner attempts are separate
    counters. Every revealed candidate, including a failure or cache hit,
    consumes one research observation. New physical attempts and cache reuse
    are separately counted so algorithm comparisons have the same information
    budget even when an earlier trial populated the measurement cache.
    A surrounding parent process supplies the hard phase deadline.
    """
    if not isinstance(budget, SearchBudget):
        raise TypeError("search budget must be SearchBudget")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    options = deepcopy(dict(options or {}))
    identity = {"case_id": case.id, "kernel_fingerprint": case.kernel.fingerprint,
                "operator_fingerprint": case.operator_definition.fingerprint,
                "space_fingerprint": case.space.fingerprint,
                "algorithm_ref": algorithm_definition.ref, "algorithm_fingerprint": algorithm_definition.fingerprint,
                "options": options, "seed": seed, "warmup": warmup, "repetitions": repetitions,
                "split": split, "experiment_id": experiment_id,
                "budget": {"max_evaluations": budget.max_evaluations, "max_wall_time_s": budget.max_wall_time_s}}
    trial_id = stable_id(identity)
    records = []
    trace = []
    attempted = set()
    failures = []
    attempts = evaluations = cache_hits = contract_failures = unrecorded_failures = 0
    status = "passed"
    stop_reason = "evaluation_budget"
    trace_path = output / "observations.jsonl"
    trace_path.write_text("", encoding="utf-8")
    _save_json(output / "trial.json", {**identity, "trial_id": trial_id, "status": "running",
        "kernel_ref": case.kernel.ref, "operator_ref": case.operator_definition.ref,
        "shape": dict(case.shape.dims), "dtype": case.operator.dtype, "trace_schema_version": 1})

    def observed(record, cached, admitted_at):
        verified = _correctness(record)[0]
        previous = trace[-1]["best_verified_latency_ms"] if trace else None
        best = min(previous, record.median_ms) if verified and previous is not None else (
            record.median_ms if verified else previous)
        row = {"observation": len(records), "record_id": record.id, "config_id": record.config.id,
               "status": record.status.value, "cache_hit": cached, "correctness_verified": verified,
               "latency_ms": record.median_ms if verified else None, "best_verified_latency_ms": best,
               "elapsed_wall_time_s": time.perf_counter() - started,
               "lookup_and_measurement_wall_time_s": time.perf_counter() - admitted_at,
               "source_compile_time_ms": record.compile_time_ms, "source_measurement_wall_time_s": record.wall_time_s,
               "physical_compile_time_ms": 0.0 if cached else record.compile_time_ms,
               "physical_measurement_wall_time_s": 0.0 if cached else record.wall_time_s}
        trace.append(row)
        # Linear append-only checkpoint; final trial.json remains the manifest.
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(row) + "\n")

    def notify(value):
        if progress is not None:
            progress({"trial_id": trial_id, "kernel": case.kernel.ref,
                      "algorithm": algorithm_definition.ref, "shape": dict(case.shape.dims), **value})

    def note_failure(phase, exc):
        event = {"trial_id": trial_id, "kernel": case.kernel.ref, "algorithm": algorithm_definition.ref,
                 "shape": dict(case.shape.dims), "phase": phase,
                 "error_type": type(exc).__name__, "error": str(exc),
                 "attempts": attempts, "evaluations": evaluations, "cache_hits": cache_hits}
        failures.append(event)
        with (output / "trial_failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(event) + "\n")

    def finish():
        successes = [record for record in records if record.successful]
        best = min(successes, key=lambda record: (record.median_ms, record.config.id), default=None)
        verified_rows = [record for record in successes if _correctness(record)[0]]
        best_verified = min(verified_rows, key=lambda record: (record.median_ms, record.config.id), default=None)
        verified, origin = _correctness(best)
        result = {"trial_id": trial_id, "case_id": case.id, "status": status, "stop_reason": stop_reason,
                  "kernel_ref": case.kernel.ref, "kernel_fingerprint": identity["kernel_fingerprint"],
                  "operator_ref": case.operator_definition.ref,
                  "algorithm_ref": algorithm_definition.ref, "algorithm_fingerprint": identity["algorithm_fingerprint"],
                  "algorithm_options": options, "options": options, "seed": seed,
                  "training_data": identity.get("training_data"),
                  "shape": dict(case.shape.dims), "dtype": case.operator.dtype,
                  "record_ids": [record.id for record in records], "evaluations": evaluations,
                  "observations": len(records),
                  "physical_evaluations": evaluations, "cache_hits": cache_hits, "attempts": attempts,
                  "algorithm_contract_failures": contract_failures,
                  "failed_attempts": sum(not record.successful for record in records) + contract_failures + unrecorded_failures,
                  "failed_measurements": sum(not record.successful for record in records),
                  "statuses": dict(Counter(record.status.value for record in records)),
                  "best_config": dict(best.config.params) if best is not None else None,
                  "best_latency_ms": best.median_ms if best is not None else None,
                  "best_record_id": best.id if best is not None else None,
                  "recommendation_verified": verified, "correctness_origin": origin,
                  "best_verified_config": dict(best_verified.config.params) if best_verified is not None else None,
                  "best_verified_latency_ms": best_verified.median_ms if best_verified is not None else None,
                  "best_verified_record_id": best_verified.id if best_verified is not None else None,
                  "wall_time_s": time.perf_counter() - started, "budget": identity["budget"],
                  "space_cardinality": case.space.cardinality,
                  "space_fingerprint": identity["space_fingerprint"], "failures": failures,
                  "trace_schema_version": 1, "observation_trace": trace,
                  "physical_compile_time_ms": sum(row["physical_compile_time_ms"] for row in trace),
                  "physical_measurement_wall_time_s": sum(row["physical_measurement_wall_time_s"] for row in trace),
                  "source_compile_time_ms": sum(row["source_compile_time_ms"] for row in trace),
                  "source_measurement_wall_time_s": sum(row["source_measurement_wall_time_s"] for row in trace),
                  "budget_semantics": "max_evaluations bounds revealed candidates, including failures and cache hits; physical evaluations are reported separately; parent enforces hard wall deadline"}
        _save_json(output / "trial.json", result)
        return result

    # Zero allowance never creates the implementation or asks it for proposals.
    if budget.exhausted(0, time.perf_counter() - started):
        stop_reason = "wall_time_budget" if budget.max_wall_time_s is not None and time.perf_counter() - started >= budget.max_wall_time_s else "evaluation_budget"
        return finish()
    if case.unavailable_reason:
        status, stop_reason = "deferred", "backend_unavailable"
        failures.append({"phase": "runtime", "error": case.unavailable_reason})
        return finish()
    if case.validation_only:
        status, stop_reason = "skipped", "validation_only_case"
        return finish()

    try:
        notify({"event": "algorithm_factory"})
        searcher = create_searcher(algorithm_definition, options, seed)
        # Built-in Transfer and opt-in plugins expose their actual historical
        # records. Bind content after loading; a database path can change in
        # place while algorithm code/options remain identical.
        training = getattr(searcher, "training_records", None)
        if training is not None:
            identity["training_data"] = training_record_manifest(training)
            trial_id = stable_id(identity)
            _save_json(output / "trial.json", {**identity, "trial_id": trial_id, "status": "running",
                "kernel_ref": case.kernel.ref, "operator_ref": case.operator_definition.ref,
                "shape": dict(case.shape.dims), "dtype": case.operator.dtype, "trace_schema_version": 1})
            notify({"event": "algorithm_ready", "training_data_sha256": identity["training_data"]["record_ids_sha256"]})
    except Exception as exc:
        status, stop_reason = "failed", "algorithm_contract_error"
        contract_failures += 1
        note_failure("algorithm_factory", exc)
        return finish()
    runner = BenchmarkRunner(case.adapter, store, warmup=warmup, repetitions=repetitions, seed=seed,
                             experiment_id=experiment_id, failure_log_path=output / "failures.jsonl")

    while not budget.exhausted(evaluations, time.perf_counter() - started):
        if len(records) >= budget.max_evaluations:
            stop_reason = "observation_budget"
            break
        attempts += 1
        try:
            notify({"event": "algorithm_suggest", "attempts": attempts,
                    "evaluations": evaluations, "cache_hits": cache_hits})
            proposals = request_candidates(
                searcher, case.hardware, case.operator, case.shape, case.space,
                SearchHistory(tuple(deepcopy(records)), frozenset(attempted)),
                budget.remaining(len(records), time.perf_counter() - started), batch_size=1)
        except Exception as exc:
            status, stop_reason = "failed", "algorithm_contract_error"
            contract_failures += 1
            note_failure("algorithm_suggest", exc)
            break
        if not proposals:
            stop_reason = "searcher_exhausted"
            break
        config = proposals[0]
        # Suggestion time is charged before admitting compilation or execution.
        if budget.exhausted(evaluations, time.perf_counter() - started):
            stop_reason = "wall_time_budget"
            break
        attempted.add(config.id)
        initial = None
        attempt_started = time.perf_counter()
        try:
            # This marker survives even an environment/protocol probe hanging.
            notify({"event": "measurement_preparation", "config": dict(config.params),
                    "attempts": attempts, "evaluations": evaluations, "cache_hits": cache_hits})
            environment = case.adapter.environment()
            protocol = {**case.adapter.protocol_metadata(),
                        "config_space": {"name": case.space.name, "fingerprint": case.space.fingerprint,
                                         "cardinality": case.space.cardinality}}
            initial = MeasurementRecord(
                case.hardware, case.operator, case.shape, deepcopy(config), RecordStatus.RUNTIME_ERROR,
                warmup=warmup, repetitions=repetitions, measurement_mode=case.adapter.measurement_mode,
                environment=environment, failure_reason="research evaluation has not completed",
                provenance=case.adapter.provenance, split=split, seed=seed, experiment_id=experiment_id,
                metadata={"protocol": protocol, "correctness_verified": False, "pending_trial_id": trial_id})
            notify({"event": "measurement", "config": dict(config.params), "attempt_record": initial.to_dict(),
                    "attempts": attempts, "evaluations": evaluations, "cache_hits": cache_hits})
            if budget.exhausted(evaluations, time.perf_counter() - started):
                stop_reason = "wall_time_budget"
                break
            evaluations += 1
            record = runner.measure(case.hardware, case.operator, case.shape, deepcopy(config), case.space,
                                    split=split, resume=resume)
            if runner.last_cache_hit:
                evaluations -= 1
                cache_hits += 1
            if record.case_id != initial.case_id or record.config.id != config.id or record.split != split:
                raise ValueError("runner returned a record for a different case/configuration/split")
            records.append(deepcopy(record))
            observed(record, runner.last_cache_hit, attempt_started)
            notify({"event": "measurement_completed", "config": dict(config.params),
                    "record_id": record.id, "record_status": record.status.value,
                    "cache_hit": runner.last_cache_hit, "attempts": attempts,
                    "evaluations": evaluations, "cache_hits": cache_hits})
        except Exception as exc:
            status, stop_reason = "failed", "runner_error"
            note_failure("measurement", exc)
            # Runner failures before its own journal append still leave evidence
            # for a real admitted candidate; no fake runtime samples are created.
            archived = False
            if initial is not None:
                failed = replace(initial, failure_reason=f"research runner: {type(exc).__name__}: {exc}",
                                 wall_time_s=time.perf_counter() - attempt_started)
                try:
                    if store is not None:
                        store.add(failed)
                    records.append(failed)
                    observed(failed, False, attempt_started)
                    archived = True
                except Exception as archive_error:
                    note_failure("failure_archive", archive_error)
            if not archived:
                unrecorded_failures += 1
            break

    elapsed = time.perf_counter() - started
    if status == "passed" and budget.max_wall_time_s is not None and elapsed >= budget.max_wall_time_s:
        stop_reason = "wall_time_budget"
    if status == "passed" and records and not any(record.successful for record in records):
        status = "failed"
        failures.append({"phase": "research", "error": "no configuration produced a successful measurement"})
    if status == "passed" and not records and stop_reason == "searcher_exhausted":
        status = "failed"
        failures.append({"phase": "research", "error": "algorithm exhausted without producing any observation"})
    return finish()


__all__ = ["search_case"]
