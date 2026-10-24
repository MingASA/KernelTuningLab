"""Read Kernel Tuner cache/results JSON without importing Kernel Tuner.

An upstream timing does not establish correctness. By default imported timing
records have CORRECTNESS_UNKNOWN status, so they cannot become replay or L3
oracles. See docs/PUBLIC_DATA.md for the supported formats and trust boundary.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..models import HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec, RecordStatus, ShapeSpec


class KernelTunerImportError(ValueError):
    """The source cannot be mapped without guessing measurement semantics."""


def bundled_sample_path() -> Path:
    """Return the bundled, attributed real upstream cache excerpt."""
    return Path(__file__).with_name("data") / "benchmark_hub_convolution_A100.sample.json"


def _positive(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _duration(value: Any) -> float | None:
    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _shape(problem_size: Any) -> ShapeSpec:
    if isinstance(problem_size, int) and not isinstance(problem_size, bool):
        dims = {"X": problem_size}
    elif isinstance(problem_size, (list, tuple)):
        names = ("X", "Y", "Z")
        dims = {(names[i] if i < 3 else f"D{i}"): value for i, value in enumerate(problem_size)}
    elif isinstance(problem_size, Mapping):
        dims = dict(problem_size)
    else:
        raise KernelTunerImportError("problem_size is absent or symbolic; provide an explicit ShapeSpec")
    try:
        return ShapeSpec(dims)
    except (TypeError, ValueError) as exc:
        raise KernelTunerImportError("problem_size must contain positive integer dimensions; provide ShapeSpec") from exc


def load_kernel_tuner_json(
    path: str | Path,
    *,
    hardware: HardwareSpec | None = None,
    operator: OperatorSpec | None = None,
    shape: ShapeSpec | None = None,
    correctness: str = "unknown",
    correctness_evidence: str | None = None,
    source_url: str | None = None,
    experiment_id: str = "kernel_tuner_import",
    split: str = "train",
    tune_params_keys: Sequence[str] | None = None,
) -> list[MeasurementRecord]:
    """Import cache JSON, JSON.gz, or a ``results``/``env`` envelope.

    ``correctness='verified'`` requires caller-provided evidence applying to all
    successful source entries. No correctness claim is inferred from a timing,
    a GPU name, or ``verification_time``. Domain-specific identity can be given
    with hardware/operator/shape overrides; otherwise unknown facts stay unknown.
    A bare result list additionally requires explicit ``tune_params_keys``.
    """
    if correctness not in {"unknown", "verified"}:
        raise KernelTunerImportError("correctness must be 'unknown' or 'verified'")
    if correctness == "verified" and (not correctness_evidence or not correctness_evidence.strip()):
        raise KernelTunerImportError("verified correctness requires nonempty correctness_evidence")
    source_path = Path(path)
    raw_bytes = source_path.read_bytes()
    try:
        decoded = gzip.decompress(raw_bytes) if raw_bytes[:2] == b"\x1f\x8b" else raw_bytes
        payload = json.loads(decoded)
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KernelTunerImportError(f"invalid Kernel Tuner JSON: {exc}") from exc
    if isinstance(payload, list):
        header: dict[str, Any] = {}
        entries = [(str(i), row) for i, row in enumerate(payload)]
        source_format = "results_list"
    elif isinstance(payload, dict):
        header = payload
        if isinstance(payload.get("cache"), dict):
            entries = list(payload["cache"].items())
            source_format = "cache"
        elif isinstance(payload.get("results"), list):
            entries = [(str(i), row) for i, row in enumerate(payload["results"])]
            source_format = "results_envelope"
        else:
            raise KernelTunerImportError("expected a cache mapping or results list; T4/store_results schemas are unsupported")
    else:
        raise KernelTunerImportError("top-level JSON must be a cache object or results list")

    environment = header.get("environment", header.get("env", {}))
    if not isinstance(environment, dict):
        raise KernelTunerImportError("env/environment must be a JSON object")
    domains = header.get("tune_params", {})
    keys = list(tune_params_keys or header.get("tune_params_keys") or (domains.keys() if isinstance(domains, dict) else ()))
    if not keys or len(set(keys)) != len(keys) or any(not isinstance(key, str) for key in keys):
        raise KernelTunerImportError("explicit, unique tune_params_keys or tune_params are required; metric keys are not configuration parameters")
    source_manifest = header.get("_kernel_tuning_lab_source", {})
    if not isinstance(source_manifest, dict):
        source_manifest = {}
    source_url = source_url or source_manifest.get("source_url")
    device_name = header.get("device_name", environment.get("device_name"))
    kernel_name = header.get("kernel_name", environment.get("kernel_name"))
    if hardware is None:
        hardware = HardwareSpec(str(device_name or "unknown imported device"), metadata={"hardware_details_known": False})
    if operator is None:
        operator = OperatorSpec(
            str(kernel_name or "unknown imported kernel"), kind="unknown", implementation="kernel_tuner",
            dtype="unknown", correctness_baseline="unspecified",
        )
    if shape is None:
        shape = _shape(header.get("problem_size", environment.get("problem_size")))

    error_status = {
        "InvalidConfig": RecordStatus.INVALID_CONFIG,
        "CompilationFailedConfig": RecordStatus.COMPILE_ERROR,
        "RuntimeFailedConfig": RecordStatus.RUNTIME_ERROR,
        "VerificationFailedConfig": RecordStatus.CORRECTNESS_ERROR,
        "CorrectnessFailedConfig": RecordStatus.CORRECTNESS_ERROR,
    }
    source_digest = hashlib.sha256(raw_bytes).hexdigest()
    records: list[MeasurementRecord] = []
    for entry_key, row in entries:
        if not isinstance(row, dict):
            raise KernelTunerImportError(f"entry {entry_key}: result must be an object")
        missing = [key for key in keys if key not in row]
        if missing:
            raise KernelTunerImportError(f"entry {entry_key}: missing tuning parameters {missing}")
        samples_raw = row.get("times")
        upstream_time = row.get("time")
        samples: tuple[float, ...] = ()
        aggregate_only = False
        if isinstance(samples_raw, list) and samples_raw:
            if not all(_positive(value) for value in samples_raw):
                raise KernelTunerImportError(f"entry {entry_key}: runtime samples must be finite positive milliseconds")
            samples = tuple(float(value) for value in samples_raw)
        elif _positive(upstream_time):
            samples = (float(upstream_time),)
            aggregate_only = True
        if isinstance(upstream_time, str) and upstream_time in error_status:
            status = error_status[upstream_time]
            reason = str(row.get("error", row.get("failure_reason", upstream_time)))
            samples = ()
        elif samples:
            status = RecordStatus.SUCCESS if correctness == "verified" else RecordStatus.CORRECTNESS_UNKNOWN
            reason = None if correctness == "verified" else "upstream correctness verification unspecified; timing retained for inspection only"
        else:
            raise KernelTunerImportError(f"entry {entry_key}: missing valid timing or recognized upstream failure marker {upstream_time!r}")
        cost_keys = ("compile_time", "verification_time", "benchmark_time", "framework_time", "strategy_time")
        costs = {key: _duration(row.get(key)) for key in cost_keys}
        cost_known = all(value is not None for value in costs.values())
        metadata: dict[str, Any] = {
            "source_adapter": "kernel_tuner_json", "source_format": source_format,
            "source_url": source_url, "source_file_sha256": source_digest,
            "source_entry": entry_key, "source_manifest": source_manifest,
            "upstream_device_name": device_name, "upstream_kernel_name": kernel_name,
            "upstream_problem_size": header.get("problem_size"),
            "correctness_verified": correctness == "verified" and status == RecordStatus.SUCCESS,
            "correctness_evidence": correctness_evidence,
            "runtime_statistic": "upstream_aggregate_only" if aggregate_only else "individual_samples",
            "upstream_time_ms": upstream_time,
            "measurement_cost_known": cost_known,
            "compile_time_available": costs["compile_time"] is not None,
            "source_cost_components_ms": costs,
            "warmup_known": False,
            "upstream_metrics": {key: value for key, value in row.items() if key not in {*keys, "times"}},
        }
        if domains:
            metadata["upstream_parameter_domains"] = domains
        records.append(MeasurementRecord(
            hardware, operator, shape, KernelConfig({key: row[key] for key in keys}), status,
            compile_time_ms=costs["compile_time"] or 0.0,
            runtime_samples_ms=samples,
            wall_time_s=sum(value for value in costs.values() if value is not None) / 1000 if cost_known else 0.0,
            warmup=0, repetitions=len(samples) if not aggregate_only else 0,
            measurement_mode="imported_kernel_tuner_samples" if not aggregate_only else "imported_kernel_tuner_aggregate",
            environment=environment, failure_reason=reason, provenance="imported", split=split,
            experiment_id=experiment_id, metadata=metadata,
        ))
    return records
