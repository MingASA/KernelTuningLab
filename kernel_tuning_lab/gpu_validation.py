"""Aggregate recorded physical smoke runs into a device-specific validation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import MeasurementRecord


SMOKE_CASES = tuple((operator, dtype) for operator in ("gemm", "batched_gemm", "softmax", "layernorm")
                   for dtype in ("float16", "bfloat16")) + (("softmax", "float32"), ("layernorm", "float32"))
_PHYSICAL_FIELDS = ("name", "architecture", "sm_count", "smem_per_block", "registers_per_sm", "memory_bytes")
_RUNTIME_FIELDS = ("python", "torch", "triton", "cuda_runtime", "nvidia_driver")


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def summarize_gpu_smoke(output_dir: str | Path) -> dict:
    root = Path(output_dir)
    errors, stages = [], []
    preflight, environment = {}, {}
    source_hashes: dict[str, set[str]] = {}

    def collect_source(filename: str, fingerprint: str | None, problems: list[str]) -> None:
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            problems.append(f"missing or invalid source fingerprint for {filename}")
        else:
            source_hashes.setdefault(filename, set()).add(fingerprint)

    try:
        preflight = _read_object(root / "preflight.json")
        if preflight.get("runtime_requirements_met") is not True:
            errors.append("preflight did not confirm runtime requirements")
        hardware = preflight["hardware"]
        if hardware.get("metadata", {}).get("specification_only") is True:
            errors.append("preflight contains a datasheet profile instead of a physical probe")
        environment = preflight["environment"]
        collect_source("_triton_kernel.py", environment.get("kernel_source_sha256"), errors)
        collect_source("triton_gemm.py", environment.get("adapter_source_sha256"), errors)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        hardware = {}
        errors.append(f"invalid or missing preflight: {exc}")
    total_records, cache_hits = 0, 0
    kernel_hashes: dict[str, set[str]] = {}
    for operator, dtype in SMOKE_CASES:
        directory = root / f"{operator}-{dtype}"
        problems, count, observed_shapes = [], 0, set()
        try:
            run = _read_object(directory / "run_summary.json")
            if run.get("provenance") != "measured" or run.get("all_successful") is not True:
                problems.append("run must contain only successful physical measurements")
            if not run.get("requested_shapes") or run.get("successful_shapes") != run["requested_shapes"]:
                problems.append("not every requested shape succeeded")
            with (directory / "run_records.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    record = MeasurementRecord.from_dict(row)
                    count += 1
                    observed_shapes.add(record.shape.id)
                    if row.get("provenance") != "measured" or row.get("status") != "success":
                        problems.append("journal contains synthetic, imported, unknown or failed records")
                    if row.get("operator", {}).get("kind") != operator or row.get("operator", {}).get("dtype") != dtype:
                        problems.append("journal operator/dtype differs from the requested stage")
                    if row.get("operator", {}).get("implementation") != f"triton_{operator}":
                        problems.append("journal does not contain the expected Triton implementation")
                    if row.get("metadata", {}).get("correctness", {}).get("verified_on_gpu") is not True:
                        problems.append("journal does not record physical numerical verification")
                    physical = row.get("hardware", {})
                    if physical.get("metadata", {}).get("specification_only") is True:
                        problems.append("journal substitutes a datasheet profile for physical probe data")
                    if any(physical.get(key) != hardware.get(key) for key in _PHYSICAL_FIELDS):
                        problems.append("journal device differs from the preflight device")
                    recorded_environment = row.get("environment", {})
                    if any(recorded_environment.get(key) != environment.get(key) for key in _RUNTIME_FIELDS):
                        problems.append("journal runtime differs from the preflight runtime")
                    implementation = row.get("operator", {}).get("implementation", "")
                    fingerprint = recorded_environment.get("kernel_source_sha256")
                    if operator in {"gemm", "batched_gemm"}:
                        kernel_file, adapter_file = "_triton_kernel.py", "triton_gemm.py"
                    else:
                        kernel_file, adapter_file = "_triton_reductions.py", "triton_reductions.py"
                        collect_source("triton_gemm.py", recorded_environment.get("base_adapter_source_sha256"), problems)
                    collect_source(kernel_file, fingerprint, problems)
                    collect_source(adapter_file, recorded_environment.get("adapter_source_sha256"), problems)
                    if isinstance(fingerprint, str):
                        kernel_hashes.setdefault(implementation, set()).add(fingerprint)
            if not count or count != run.get("returned_records"):
                problems.append("journal record count does not match the stage summary")
            if len(observed_shapes) != run.get("requested_shapes") or observed_shapes != set(run.get("requested_shape_ids", ())):
                problems.append("journal shapes do not match the requested stage shapes")
            cache_hits += int(run.get("stats", {}).get("cache_hits", 0))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            problems.append(f"invalid or missing stage artifacts: {exc}")
        total_records += count
        stages.append({"operator": operator, "dtype": dtype, "records": count,
                       "status": "failed" if problems else "passed", "errors": sorted(set(problems))})
    for filename, hashes in source_hashes.items():
        if len(hashes) != 1:
            errors.append(f"source changed across preflight/smoke stages for {filename}")
    passed = not errors and all(stage["status"] == "passed" for stage in stages)
    result = {"status": "passed" if passed else "failed", "physical_gpu_validated": passed,
              "hardware": hardware, "requested_profile": preflight.get("requested_profile"),
              "stages": stages, "total_records": total_records, "cache_hits": cache_hits,
              "kernel_source_hashes": {key: sorted(value) for key, value in kernel_hashes.items()},
              "source_hashes": {key: sorted(value) for key, value in source_hashes.items()},
              "errors": errors, "target_hopper_validation_required": True,
              "notice": "Device-specific correctness result; rerun on the target Hopper server before production use."}
    root.mkdir(parents=True, exist_ok=True)
    (root / "validation_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize ten physical GPU smoke stages")
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    result = summarize_gpu_smoke(args.output_dir)
    print(json.dumps({"status": result["status"], "total_records": result["total_records"],
                      "passed_stages": sum(stage["status"] == "passed" for stage in result["stages"]),
                      "summary": str(Path(args.output_dir) / "validation_summary.json")}, indent=2))
    return 0 if result["physical_gpu_validated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
