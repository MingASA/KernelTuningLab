"""Configuration-driven asset execution with supervised phase deadlines."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .asset_experiments import AssetExperiment
from .models import HardwareSpec, MeasurementRecord, OperatorSpec, RecordStatus, ShapeSpec, stable_id
from .storage import ExperimentStore
from .run_lifecycle import (OutputDirectoryBusyError, OutputLease, WorkerInterrupted,
                            archive_previous_attempt, supervised_worker)


class AssetNotApplicableError(ValueError):
    """A declared dtype/architecture/algorithm capability does not match."""


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path: Path, default=None):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value
    except (OSError, ValueError):
        return default


def _plugin_reference(reference: str, directory: Path) -> str:
    path = Path(reference).expanduser()
    candidate = directory / path
    if path.suffix == ".py" or path.is_absolute() or candidate.is_file():
        return str((path if path.is_absolute() else candidate).resolve())
    return reference


def _archives(output: Path, metadata: dict | None = None) -> None:
    """Retain the previous attempt, then revoke current approval markers."""
    output.mkdir(parents=True, exist_ok=True)
    previous = archive_previous_attempt(output)
    metadata = {**(metadata or {}), "previous_attempt": str(previous) if previous else None}
    _write(output / "run_metadata.json", metadata)
    _write(output / "run_summary.json", {"status": "running", "run_id": metadata.get("run_id"),
                                        "physical_gpu_validated": False})
    _write(output / "trials.json", [])


def _error(exc: Exception) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)}


def _report(output: Path, summary: dict, trials: list) -> None:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# Asset experiment", "", f"Status: **{summary['status']}**", "",
             "Evidence applies to the recorded runtime, sampled inputs and configurations. "
             "CPU timing is not GPU timing. Submitter assertions remain separate from platform reference checks.", "",
             f"Research evaluations: {summary.get('evaluations', 0)}; validation evaluations: "
             f"{summary.get('validation_evaluations', 0)}; cache hits: {summary.get('cache_hits', 0)}.", "",
             "| Kernel | Algorithm | Shape | Best ms | Evidence | Status / stop |",
             "| --- | --- | --- | ---: | --- | --- |"]
    for trial in trials:
        latency = trial.get("best_latency_ms")
        lines.append("| " + " | ".join(map(cell, (
            trial.get("kernel_ref"), trial.get("algorithm_ref"), trial.get("shape"),
            "N/A" if latency is None else f"{latency:.6g}", trial.get("correctness_origin") or "none",
            f"{trial.get('status')} / {trial.get('stop_reason')}"))) + " |")
    lines += ["", "[Preflight](preflight.json) · [Acceptance](acceptance.json) · "
              "[Asset lock](assets.lock.json) · [Trials](trials.json) · [Measurements](measurements.jsonl) · [Worker log](worker.log)", ""]
    if summary.get("error"):
        lines += ["## Failure", "", cell(summary["error"]), ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _save_failure(output: Path, status: str, error: dict, *, phase: str, exit_code: int) -> dict:
    summary = {"status": status, "phase": phase, "error": error,
               "run_id": _read(output / "run_metadata.json", {}).get("run_id"),
               "exit_code": exit_code, "physical_gpu_validated": False,
               "output": str(output.resolve()), "worker_log": str((output / "worker.log").resolve())}
    for name in ("preflight.json", "acceptance.json"):
        previous = _read(output / name, {})
        if not isinstance(previous, dict):
            previous = {}
        if name == "acceptance.json" and phase == "validation":
            previous = _read(output / "validation_report.json", previous)
        if previous.get("status") not in {"passed", "deferred"}:
            previous.update({"status": status, "phase": phase, "error": error})
            _write(output / name, previous)
    if not (output / "assets.lock.json").exists():
        _write(output / "assets.lock.json", {"status": "incomplete", "assets": [], "error": error})
    _write(output / "run_summary.json", summary)
    _report(output, summary, _read(output / "trials.json", []))
    return summary


def _interrupt_measurement(output: Path, progress: dict, reason: str, *, status: str = "timeout") -> None:
    _write(output / "interrupted_attempt.json", {"status": status, "progress": progress, "reason": reason})
    raw = progress.get("attempt_record")
    if not isinstance(raw, dict):
        return
    try:
        row = MeasurementRecord.from_dict(raw)
        if row.status != RecordStatus.RUNTIME_ERROR or not (
            row.metadata.get("pending_trial_id") or row.metadata.get("pending_acceptance")
        ):
            return
        row = replace(row, status=RecordStatus.RUNTIME_ERROR, runtime_samples_ms=(), failure_reason=reason,
                      wall_time_s=max(0.0, time.monotonic() - progress.get("attempt_started_monotonic", time.monotonic())),
                      metadata={**row.metadata, "terminated_by_parent": True, "termination_kind": status,
                                "termination_run_id": _read(output / "run_metadata.json", {}).get("run_id"),
                                "timeout_phase": progress.get("phase") if status == "timeout" else None})
        with ExperimentStore(output / "measurements.sqlite") as store:
            store.add(row)
            store.export_jsonl(output / "measurements.jsonl")
    except Exception as exc:
        _write(output / "interrupted_record_error.json", _error(exc))


def execute_asset_experiment(spec_path: str | Path, output_dir: str | Path, *, check_only: bool = False,
                             plugins: tuple[str, ...] = (), inherited_fds: tuple[int, ...] = (),worker_environment=None) -> dict:
    """Run untrusted-in-behavior extensions in a disposable supervised process.

    Local Python plugins are trusted code, not a security sandbox. The parent
    enforces deadlines and retains diagnostics if import/JIT/launch/suggest hangs.
    Killing a worker does not promise that an unresponsive GPU driver is reset.
    """
    output, source = Path(output_dir).resolve(), Path(spec_path).resolve()
    try:
        with OutputLease(output) as lease:
            return _execute_owned(source, output, check_only=check_only, plugins=plugins, lease=lease,
                                  inherited_fds=inherited_fds,worker_environment=worker_environment)
    except OutputDirectoryBusyError as exc:
        # A rejected concurrent caller must not overwrite the owner's progress,
        # worker log, approval marker, or archived experiment.
        return {"status": "failed", "exit_code": 2, "phase": "output_ownership", "error": _error(exc),
                "physical_gpu_validated": False, "output": str(output)}


def _execute_owned(source: Path, output: Path, *, check_only: bool, plugins: tuple[str, ...], lease,
                   inherited_fds: tuple[int, ...] = (),worker_environment=None) -> dict:
    # Parse before moving the old metadata: the user may resume using the
    # archived experiment.json from this very output directory.
    try:
        spec = AssetExperiment.load(source)
    except Exception as exc:
        _archives(output, lease.metadata)
        return _save_failure(output, "failed", _error(exc), phase="configuration", exit_code=2)
    references = tuple(dict.fromkeys([
        *(_plugin_reference(plugin, source.parent) for plugin in spec.plugins),
        *(_plugin_reference(plugin, Path.cwd()) for plugin in plugins),
    ]))
    _archives(output, lease.metadata)
    spec.save(output / "source_experiment.json")
    spec = replace(spec, plugins=references)
    snapshot = output / "experiment.json"
    spec.save(snapshot)
    _write(output / "experiment_origin.json", {"source": str(source), "plugin_base_directory": str(source.parent)})
    command = [sys.executable, "-m", "kernel_tuning_lab.asset_runner", "--worker", "--spec", str(snapshot),
               "--origin", str(source), "--output", str(output)]
    if check_only:
        command.append("--check-only")
    started = time.monotonic()
    combinations = len(spec.kernels) * len(spec.algorithms) * len(spec.shapes)
    total_limit = 2 * spec.preflight_seconds + spec.validation.max_wall_time_s
    if not check_only:
        total_limit += combinations * spec.budget.max_wall_time_s
    # Startup/atomic result writes have a small bounded allowance outside phases.
    total_limit += 5.0
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        try:
            process_options = lease.subprocess_options
            if os.name == "posix":
                process_options["pass_fds"] = tuple(dict.fromkeys((*process_options.get("pass_fds", ()), *inherited_fds)))
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix",
                                       env=worker_environment,**process_options)
        except OSError as exc:
            return _save_failure(output, "failed", _error(exc), phase="worker_startup", exit_code=2)
        timeout = None
        try:
            with supervised_worker(process):
                while process.poll() is None:
                    now = time.monotonic()
                    progress = _read(output / "phase.json", {})
                    phase = progress.get("phase", "preflight")
                    phase_start = progress.get("started_monotonic", started)
                    limit = (spec.validation.max_wall_time_s if phase == "validation" else
                             spec.budget.max_wall_time_s if phase == "search" else spec.preflight_seconds)
                    if now - phase_start >= limit or now - started >= total_limit:
                        timeout = progress, phase, limit
                        break
                    try:
                        process.wait(timeout=0.025)
                    except subprocess.TimeoutExpired:
                        pass
        except (WorkerInterrupted, KeyboardInterrupt) as exc:
            progress = _read(output / "phase.json", {})
            signum = getattr(exc, "signum", signal.SIGINT)
            reason = f"experiment interrupted by signal {signum}"
            _interrupt_measurement(output, progress, reason, status="interrupted")
            return _save_failure(output, "interrupted", {"type": "Interrupted", "message": reason},
                                 phase=progress.get("phase", "worker"), exit_code=128 + signum)
        if timeout is not None:
            progress, phase, limit = timeout
            reason = f"{phase} exceeded hard wall deadline ({limit:g}s)"
            _interrupt_measurement(output, progress, reason)
            return _save_failure(output, "timeout", {"type": "HardTimeout", "message": reason,
                                                      "limit_s": limit}, phase=phase, exit_code=124)
    result = _read(output / "run_summary.json", {})
    if not isinstance(result, dict) or result.get("status") == "running" or not result:
        progress = _read(output / "phase.json", {})
        reason = f"worker exited {process.returncode} before completing result archival"
        _interrupt_measurement(output, progress, reason, status="worker_exit")
        return _save_failure(output, "failed", {"type": "WorkerExit", "message": f"worker exited {process.returncode}"},
                             phase=progress.get("phase", "worker"), exit_code=2)
    if result.get("exit_code") != process.returncode:
        progress = _read(output / "phase.json", {})
        reason = f"worker exit {process.returncode} disagrees with archived exit {result.get('exit_code')}"
        _interrupt_measurement(output, progress, reason, status="worker_exit")
        return _save_failure(output, "failed", {"type": "WorkerExit", "message":
                             reason}, phase=progress.get("phase", "worker"), exit_code=2)
    return result


def _prepare_cases(spec, catalog, output, mark):
    from .acceptance import check_algorithm, check_operator_request, inspect_kernel_entrypoint, inspect_signature, probe_legal_configs
    from .asset_context import AssetCase
    from .asset_space import build_effective_space, check_algorithm_compatibility, constraint_arguments
    from .assets import SpaceContext
    from .backends import BackendUnavailableError
    from .hardware import get_hardware_profile, validate_hardware_profile
    from .kernel_runtime import AssetKernelAdapter

    operator_definition = catalog.resolve("operator", spec.operator)
    kernels = [(selection, catalog.resolve("kernel", selection.asset)) for selection in spec.kernels]
    algorithms = [(selection, catalog.resolve("algorithm", selection.asset)) for selection in spec.algorithms]
    definitions = [operator_definition] + [item for _, item in kernels] + [item for _, item in algorithms]
    lock = {"schema_version": 1, "experiment_schema": 2, "assets": [
        {**definition.manifest(), "ref": definition.ref, "fingerprint": definition.fingerprint}
        for definition in definitions], "effective_spaces": [],
        "requested_space_scope": "experiment_restrictions", "extra_space_scope": "kernel_defaults"}
    # Claims deliberately do not change implementation identity, but remain locked evidence.
    lock["correctness_claims"] = [{"kernel": definition.ref, "claims": [
        claim.to_dict() if hasattr(claim, "to_dict") else dict(claim) for claim in definition.claims]}
        for _, definition in kernels]
    _write(output / "assets.lock.json", lock)
    inspect_signature(operator_definition.generate_inputs, object(), spec.seed)
    inspect_signature(operator_definition.reference, object())
    if operator_definition.validate:
        inspect_signature(operator_definition.validate, object())
    for selection, algorithm in algorithms:
        inspect_signature(algorithm.factory, seed=spec.seed, **selection.options)
    checks, cases = [], []
    requested_ids = {shape.id for shape in spec.shapes}
    factory = operator_definition.validation_case_factory
    validation_shapes = operator_definition.validation_shapes
    if factory is not None:
        parameters = deepcopy({**operator_definition.parameters, **spec.parameters})
        requested_shapes = deepcopy(tuple(spec.shapes))
        inspect_signature(factory, parameters, requested_shapes)
        generated = factory(parameters, requested_shapes)
        if not isinstance(generated, tuple):
            raise TypeError("validation_case_factory must return a tuple of ShapeSpec or dimension mappings")
        normalized, generated_ids = [], set()
        for index, raw in enumerate(generated):
            if not isinstance(raw, (ShapeSpec, Mapping)):
                raise TypeError(f"validation_case_factory case {index} must be a ShapeSpec or dimension mapping")
            shape = ShapeSpec(dict(raw.dims if isinstance(raw, ShapeSpec) else raw))
            if set(shape.dims) != set(operator_definition.shape_dims):
                raise ValueError(f"validation_case_factory case {index} must match shape_dims {operator_definition.shape_dims}")
            if shape.id in generated_ids:
                raise ValueError(f"validation_case_factory returned duplicate case dimensions at index {index}")
            normalized.append(shape)
            generated_ids.add(shape.id)
        validation_shapes = tuple(normalized)
    extra = tuple(shape for shape in validation_shapes if shape.id not in requested_ids)[:spec.validation.max_cases]
    lock["validation_cases"] = {"source": "validation_case_factory" if factory is not None else "validation_shapes",
                                "generated": [dict(shape.dims) for shape in validation_shapes],
                                "extra": [dict(shape.dims) for shape in extra]}
    _write(output / "assets.lock.json", lock)
    for selection, kernel in kernels:
        mark({"kernel": kernel.ref, "step": "resolve_and_probe"})
        try:
            bound_operator = catalog.resolve("operator", kernel.operator)
            if bound_operator.ref != operator_definition.ref or bound_operator.fingerprint != operator_definition.fingerprint:
                raise ValueError(f"kernel {kernel.ref} implements {bound_operator.ref}, experiment selects {operator_definition.ref}")
            if spec.dtype not in operator_definition.dtypes:
                raise AssetNotApplicableError(f"operator {operator_definition.ref} does not support dtype={spec.dtype}; choose from {operator_definition.dtypes}")
            if kernel.dtypes and spec.dtype not in kernel.dtypes:
                raise AssetNotApplicableError(f"kernel {kernel.ref} does not support dtype={spec.dtype}; choose from {kernel.dtypes}")
            adapter = AssetKernelAdapter(kernel, operator_definition, accept_claims=spec.validation.trust_claims,
                                         force_correctness=not spec.validation.trust_claims,
                                         workspace_memory_limit_bytes=spec.workspace_memory_limit_bytes,
                                         timing_mode=spec.timing_mode, cuda_graph_batch_size=spec.cuda_graph_batch_size)
            unavailable = None
            try:
                hardware = adapter.probe_hardware()
            except BackendUnavailableError as exc:
                unavailable = str(exc)
                hardware = (get_hardware_profile(spec.hardware) if spec.hardware else
                            HardwareSpec("CUDA device pending physical probe", "unknown", metadata={"probe_pending": True}))
            if spec.hardware and not unavailable:
                expected = get_hardware_profile(spec.hardware)
                validation = validate_hardware_profile(expected, hardware)
                if not validation.valid:
                    raise ValueError("hardware profile mismatch: " + "; ".join(validation.reasons))
                hardware = replace(hardware, compute_tflops=expected.compute_tflops, bandwidth_gbps=expected.bandwidth_gbps,
                                   metadata={**hardware.metadata, "requested_profile": asdict(expected),
                                             "theoretical_values_source": "vendor profile; not measured"})
            if (kernel.architectures and hardware.architecture != "unknown"
                    and not any(fnmatchcase(hardware.architecture, pattern) for pattern in kernel.architectures)):
                raise AssetNotApplicableError(f"kernel {kernel.ref} does not support target architecture={hardware.architecture}")
            runtime_environment = adapter.environment() if not unavailable else {}
            checks.append({"kernel": kernel.ref, "check": "physical_runtime", "status": "deferred" if unavailable else "passed",
                           "reason": unavailable, "hardware": asdict(hardware),
                           "environment": runtime_environment if not unavailable else None})
        except Exception as exc:
            checks.append({"kernel": kernel.ref, "check": "kernel_binding", "status": "failed", "error": _error(exc)})
            continue
        for shape in (*spec.shapes, *extra):
            validation_only = shape.id not in requested_ids
            space_scope = "kernel_defaults" if validation_only else "experiment_restrictions"
            try:
                parameters = check_operator_request(operator_definition, spec.dtype, spec.parameters, shape)
                operator = OperatorSpec(operator_definition.id, kind=operator_definition.id, implementation=kernel.ref,
                                        dtype=spec.dtype, parameters=parameters, correctness_baseline=operator_definition.ref,
                                        version=operator_definition.version,
                                        metadata={"operator_asset_fingerprint": operator_definition.fingerprint,
                                                  "kernel_asset_fingerprint": kernel.fingerprint})
                restrictions = None if validation_only else selection.space
                space_context = SpaceContext(hardware, operator, shape, environment=runtime_environment,
                                             workspace_memory_limit_bytes=spec.workspace_memory_limit_bytes)
                space = build_effective_space(kernel, hardware, operator, shape, restrictions, context=space_context)
                space.resolution["space_scope"] = space_scope
                inspect_kernel_entrypoint(kernel, space)
                if kernel.supports:
                    inspect_signature(kernel.supports, hardware, operator, shape)
                for constraint in kernel.space.constraints:
                    inspect_signature(constraint, *constraint_arguments(constraint, space.config_at(0), space_context))
                probe = probe_legal_configs(space, hardware, operator, shape, seed=spec.validation.seed,
                                            count=max(1, min(8, spec.validation.max_configs)))
                if not probe["configs"] and not unavailable:
                    raise ValueError("no legal configuration found within bounded space probe; inspect constraints/restrictions")
                if probe["configs"] and not unavailable:
                    validation = adapter.validate(hardware, operator, shape, probe["configs"][0])
                    if not validation.valid:
                        raise ValueError("operator/kernel input contract: " + "; ".join(validation.reasons))
                case = AssetCase(kernel, operator_definition, hardware, operator, shape, space, adapter,
                                 validation_only=validation_only, unavailable_reason=unavailable)
                cases.append(case)
                lock["effective_spaces"].append(case.manifest())
                checks.append({"kernel": kernel.ref, "shape": dict(shape.dims), "case_id": case.id,
                               "check": "kernel_space", "status": "deferred" if unavailable else "passed",
                               "space_scope": space_scope,
                               "space_probe": {key: value for key, value in probe.items() if key != "configs"}})
                if case.validation_only:
                    continue
                for algorithm_selection, algorithm in algorithms:
                    mark({"kernel": kernel.ref, "shape": dict(shape.dims), "algorithm": algorithm.ref, "step": "algorithm_acceptance"})
                    try:
                        match = check_algorithm_compatibility(algorithm, space)
                        if not match.valid:
                            raise AssetNotApplicableError("algorithm/space mismatch: " + "; ".join(match.reasons))
                        if probe["configs"]:
                            evidence = check_algorithm(algorithm, algorithm_selection.options, spec.seed,
                                                       hardware, operator, shape, space)
                            status = "passed"
                        else:
                            evidence, status = {"reason": "physical resources unavailable; resolved-space synthetic checks deferred"}, "deferred"
                        checks.append({"kernel": kernel.ref, "shape": dict(shape.dims), "algorithm": algorithm.ref,
                                       "check": "algorithm_acceptance", "status": status, "evidence": evidence})
                    except Exception as exc:
                        checks.append({"kernel": kernel.ref, "shape": dict(shape.dims), "algorithm": algorithm.ref,
                                       "check": "algorithm_acceptance", "status": "failed", "error": _error(exc)})
            except Exception as exc:
                checks.append({"kernel": kernel.ref, "shape": dict(shape.dims), "check": "kernel_space",
                               "status": "failed", "space_scope": space_scope, "error": _error(exc)})
    _write(output / "assets.lock.json", lock)
    status = "failed" if any(check["status"] == "failed" for check in checks) else (
        "deferred" if any(check["status"] == "deferred" for check in checks) else "passed")
    preflight = {"status": status, "checks": checks, "cases": [case.manifest() for case in cases],
                 "validation_cases": lock["validation_cases"],
                 "requested_space_scope": lock["requested_space_scope"], "extra_space_scope": lock["extra_space_scope"],
                 "notice": "Static and synthetic checks do not establish physical numerical correctness."}
    _write(output / "preflight.json", preflight)
    return cases, algorithms, preflight


def _worker(spec_path: Path, output: Path, check_only: bool, plugins: tuple[str, ...], origin: Path | None = None) -> dict:
    # All plugin imports, discovery and acceptance run after the parent's timer starts.
    phase = {"phase": "preflight", "started_monotonic": time.monotonic()}
    _write(output / "phase.json", phase)

    def mark(details):
        if details.get("event") in {"suggest", "algorithm_suggest", "algorithm_factory", "measurement_preparation", "measurement_completed", "completed"}:
            phase.pop("attempt_record", None)
            phase.pop("attempt_started_monotonic", None)
        phase.update({key: value for key, value in details.items() if key not in {"phase", "started_monotonic"}})
        if "attempt_record" in details:
            phase["attempt_started_monotonic"] = time.monotonic()
        _write(output / "phase.json", phase)

    def begin(name, **details):
        phase.clear()
        phase.update({"phase": name, "started_monotonic": time.monotonic(), **details})
        _write(output / "phase.json", phase)

    from .asset_validation import validate_cases
    from .asset_search import search_case
    from .catalog import AssetCatalog, register_builtin_search_algorithms
    from .operator_library import register_builtin_assets
    from .plugins import discover_entry_points, load_plugin

    spec = AssetExperiment.load(spec_path)
    if origin is None:
        spec.save(output / "source_experiment.json")
    references = tuple(dict.fromkeys(_plugin_reference(plugin, spec_path.parent) for plugin in (*spec.plugins, *plugins)))
    spec = replace(spec, plugins=references)
    spec.save(output / "experiment.json")
    if origin is None:
        _write(output / "experiment_origin.json", {"source": str(spec_path), "plugin_base_directory": str(spec_path.parent)})
    catalog = AssetCatalog()
    register_builtin_search_algorithms(catalog)
    register_builtin_assets(catalog)
    discovered = discover_entry_points(catalog)
    for plugin in spec.plugins:
        mark({"step": "plugin_loading", "plugin": plugin})
        load_plugin(_plugin_reference(plugin, spec_path.parent), catalog)
    cases, algorithms, preflight = _prepare_cases(spec, catalog, output, mark)
    trials = []
    with ExperimentStore(output / "measurements.sqlite") as store:
        adapters = {id(case.adapter): case.adapter for case in cases if not case.unavailable_reason}
        counterexamples = []
        for record in store.iter_records(status=RecordStatus.CORRECTNESS_ERROR, provenance="measured"):
            for adapter in adapters.values():
                revoked = adapter.remember_counterexample(record)
                if revoked:
                    counterexamples.append({"record_id": record.id, "kernel": adapter.kernel.ref, "revoked_claims": revoked})
        _write(output / "counterexamples.json", counterexamples)
        if preflight["status"] == "failed":
            acceptance = {"status": "failed", "reason": "preflight rejected the experiment", "samples": [],
                          "preflight_evidence": [check for check in preflight["checks"] if check["status"] == "failed"]}
        else:
            if spec.validation.max_configs and spec.validation.max_wall_time_s > 0:
                begin("validation")
            acceptance = validate_cases(cases, spec.validation, store=store, split=spec.split,
                                        experiment_id=spec.name, output_dir=output, progress=mark)
        _write(output / "acceptance.json", acceptance)
        if not check_only and preflight["status"] != "failed" and acceptance["status"] != "failed":
            for case in cases:
                if case.validation_only or case.unavailable_reason:
                    continue
                for selection, algorithm in algorithms:
                    trial_directory = output / "trials" / stable_id([case.id, algorithm.ref, algorithm.fingerprint,
                                                                    selection.options, spec.seed])
                    if spec.budget.exhausted(0, 0):
                        # No research call or factory is made for zero budgets.
                        trial = search_case(case, algorithm, selection.options, spec.budget, store=store,
                                            seed=spec.seed, warmup=spec.warmup, repetitions=spec.repetitions,
                                            split=spec.split, experiment_id=spec.name, output_dir=trial_directory, progress=None)
                    else:
                        begin("search", case_id=case.id, kernel=case.kernel.ref, algorithm=algorithm.ref,
                              shape=dict(case.shape.dims))
                        trial = search_case(case, algorithm, selection.options, spec.budget, store=store,
                                            seed=spec.seed, warmup=spec.warmup, repetitions=spec.repetitions,
                                            split=spec.split, experiment_id=spec.name, output_dir=trial_directory, progress=mark)
                    trial["artifact_directory"] = str(trial_directory)
                    trials.append(trial)
                    _write(output / "trials.json", trials)
                    if trial.get("status") == "failed":
                        break
        begin("archiving")
        store.export_jsonl(output / "measurements.jsonl")
        counts = store.summary()
    failed = preflight["status"] == "failed" or acceptance["status"] == "failed" or any(t.get("status") == "failed" for t in trials)
    deferred = any(case.unavailable_reason for case in cases) or acceptance["status"] in {"deferred", "partial"}
    status = "failed" if failed else "deferred" if deferred else "passed"
    cuda_cases = {case.id for case in cases if case.kernel.runtime == "cuda"}
    gpu_verified_cases = {sample.get("case_id") for sample in acceptance.get("samples", ())
                          if sample.get("status") == "passed" and sample.get("correctness", {}).get("verified_on_gpu") is True}
    summary = {"status": status, "exit_code": 2 if failed else 3 if deferred else 0,
               "run_id": _read(output / "run_metadata.json", {}).get("run_id"),
               "experiment": spec.name, "check_only": check_only, "preflight_status": preflight["status"],
               "acceptance_status": acceptance["status"], "trials": len(trials),
               "evaluations": sum(t.get("evaluations", 0) for t in trials),
               "observations": sum(t.get("observations", 0) for t in trials),
               "cache_hits": sum(t.get("cache_hits", 0) for t in trials),
               "validation_evaluations": acceptance.get("budget", {}).get("used_configs", 0),
               "physical_gpu_validated": not failed and not deferred and bool(cuda_cases) and cuda_cases <= gpu_verified_cases,
               "correctness_evidence": acceptance.get("checks", {}),
               "installed_entry_points": list(discovered), "store": counts, "store_scope": "cumulative_journal",
               "output": str(output),
               "notice": "Acceptance is limited to recorded samples/claims. CPU evidence is not GPU validation; submitter claims are not platform verification."}
    _write(output / "run_summary.json", summary)
    _report(output, summary, trials)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Supervised asset experiment worker")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--plugin", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        result = _worker(args.spec, args.output, args.check_only, tuple(args.plugin), origin=args.origin)
    except Exception as exc:
        traceback.print_exc()
        progress = _read(args.output / "phase.json", {})
        result = _save_failure(args.output, "failed", _error(exc), phase=progress.get("phase", "loading"), exit_code=2)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
