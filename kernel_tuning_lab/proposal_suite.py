"""Generate and execute the proposal's operator matrix through public asset APIs."""

from __future__ import annotations

import json
from pathlib import Path
import re

from .asset_experiments import AlgorithmSelection, AssetExperiment, KernelSelection, ValidationBudget
from .models import SearchBudget, ShapeSpec, stable_id


def _save(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def _slug(value):
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")[:72]


def proposal_operators(catalog=None):
    from .catalog import get_catalog
    catalog = catalog or get_catalog()
    return tuple(op for op in catalog.list("operator") if op.metadata.get("proposal_family"))


def build_suite(output_dir: str | Path, *, runtime: str = "python", scale: str = "smoke",
                hardware: str | None = None, operators=(), algorithms=("random@1.0.0", "grid@1.0.0"),
                dtypes=(), max_evaluations: int = 4, max_seconds: float = 120.0,
                validation_evaluations: int | None = None, workspace_memory_limit_bytes: int = 256 * 1024**2,
                variants: bool = False, plugins=(), catalog=None) -> dict:
    """Write reproducible cases; discover the operator set from provider metadata."""
    from .catalog import get_catalog
    if runtime not in {"python", "cuda"} or scale not in {"smoke", "production"}:
        raise ValueError("suite runtime must be python/cuda and scale smoke/production")
    if runtime == "python" and scale == "production":
        raise ValueError("production workloads target CUDA; use --runtime cuda --write-only on this CPU host")
    if hardware and runtime != "cuda":
        raise ValueError("a named GPU profile is only applicable to the CUDA suite")
    if validation_evaluations is not None and (type(validation_evaluations) is not int or validation_evaluations < 0):
        raise ValueError("validation-evaluations must be a nonnegative integer")
    catalog = catalog or get_catalog()
    selected = tuple(catalog.resolve("operator", ref) for ref in operators) if operators else proposal_operators(catalog)
    if not selected:
        raise ValueError("no proposal operators were discovered")
    algorithm_refs = tuple(catalog.resolve("algorithm", ref).ref for ref in algorithms)
    if not algorithm_refs:
        raise ValueError("suite requires at least one algorithm")
    requested_dtypes = tuple(dtypes) or (("float32",) if runtime == "python" else ("float16", "bfloat16"))
    if len(set(requested_dtypes)) != len(requested_dtypes):
        raise ValueError("suite dtypes must be distinct")
    output = Path(output_dir).resolve()
    jobs = []
    references = tuple(str(Path(plugin).resolve()) if Path(plugin).is_file() else plugin for plugin in plugins)
    for operator in selected:
        if scale == "production":
            shapes = tuple(ShapeSpec(dict(values)) for values in operator.metadata.get("production_shapes", ()))
        else:
            explicit = operator.metadata.get("smoke_shapes")
            shapes = tuple(ShapeSpec(dict(values)) for values in explicit) if explicit else operator.validation_shapes
        if not shapes:
            raise ValueError(f"operator {operator.ref} declares no {scale} shapes")
        semantic_cases = [{"name": "default", "parameters": {}}]
        if variants:
            semantic_cases += list(operator.metadata.get("validation_parameter_cases", ()))
        available = tuple(kernel for kernel in catalog.list("kernel")
                          if kernel.operator in {operator.id, operator.ref} and kernel.runtime == runtime)
        for dtype in requested_dtypes:
            kernels = tuple(kernel for kernel in available if not kernel.dtypes or dtype in kernel.dtypes)
            if dtype not in operator.dtypes or not kernels:
                raise ValueError(f"operator {operator.ref} has no {runtime} implementation supporting {dtype}")
            excluded = [{"kernel": kernel.ref, "reason": f"dtype {dtype} unsupported"}
                        for kernel in available if kernel not in kernels]
            for semantic in semantic_cases:
                if set(semantic) != {"name", "parameters"}:
                    raise ValueError("validation parameter cases require exactly name and parameters")
                name = f"{operator.id}-{runtime}-{dtype}-{scale}-{semantic['name']}"
                token = _slug(name) + "-" + stable_id([operator.ref, dtype, runtime, scale, semantic])[:8]
                limit = validation_evaluations if validation_evaluations is not None else max(
                    8, len(kernels) * (len(shapes) + min(2, len(operator.validation_shapes))))
                spec = AssetExperiment(
                    name=name, operator=operator.ref, kernels=tuple(KernelSelection(kernel.ref) for kernel in kernels),
                    algorithms=tuple(AlgorithmSelection(ref) for ref in algorithm_refs), shapes=shapes,
                    plugins=references, dtype=dtype, parameters=dict(semantic["parameters"]), hardware=hardware,
                    budget=SearchBudget(max_evaluations, max_seconds), preflight_seconds=60,
                    validation=ValidationBudget(max_configs=limit, max_cases=2, max_wall_time_s=max_seconds,
                                                seed=17, trust_claims=False),
                    workspace_memory_limit_bytes=workspace_memory_limit_bytes, seed=7, warmup=1, repetitions=5)
                filename = output / "specs" / f"{token}.json"
                spec.save(filename)
                jobs.append({"name": name, "operator": operator.ref, "family": operator.metadata.get("proposal_family"),
                             "dtype": dtype, "kernels": [kernel.ref for kernel in kernels],
                             "excluded_kernels": excluded, "shapes": [dict(shape.dims) for shape in shapes],
                             "parameters": dict(semantic["parameters"]), "spec": str(filename),
                             "output": str(output / "runs" / token), "status": "generated"})
    manifest = {"schema_version": 1, "status": "generated", "runtime": runtime, "scale": scale,
                "hardware": hardware, "operators": [operator.ref for operator in selected], "cases": jobs,
                "physical_gpu_validated": False, "workspace_memory_limit_bytes": workspace_memory_limit_bytes,
                "notice": "Generated configurations are not measurements; no performance or correctness claim follows."}
    _save(output / "suite_manifest.json", manifest)
    return manifest


def run_suite(output_dir: str | Path, *, write_only: bool = False, check_only: bool = False,
              strict: bool = True, **options) -> dict:
    from .asset_runner import execute_asset_experiment
    output = Path(output_dir).resolve()
    # Revoke any old approval before generation, including generation failures.
    _save(output / "suite_summary.json", {"status": "running", "physical_gpu_validated": False})
    try:
        manifest = build_suite(output, **options)
    except Exception as exc:
        summary = {"status": "failed", "exit_code": 2, "physical_gpu_validated": False,
                   "error": {"type": type(exc).__name__, "message": str(exc)}}
        _save(output / "suite_summary.json", summary)
        return summary
    cases = []
    for job in manifest["cases"]:
        if write_only:
            cases.append(job)
            continue
        result = execute_asset_experiment(job["spec"], job["output"], check_only=check_only)
        failed_configurations = 0
        try:
            trials = json.loads((Path(job["output"]) / "trials.json").read_text(encoding="utf-8"))
            failed_configurations = sum(trial.get("failed_measurements", 0) for trial in trials)
        except (OSError, ValueError, TypeError):
            pass
        status = "failed" if strict and failed_configurations else result["status"]
        cases.append({**job, "status": status, "run": result,
                      "strict_failed_configurations": failed_configurations if strict else 0})
        _save(output / "suite_progress.json", {"status": "running", "cases": cases})
    failed = any(case["status"] in {"failed", "timeout"} for case in cases)
    pending = any(case["status"] == "deferred" for case in cases)
    status = "generated" if write_only else "failed" if failed else "deferred" if pending else "passed"
    summary = {"status": status, "exit_code": 2 if failed else 3 if pending else 0, "runtime": manifest["runtime"],
               "scale": manifest["scale"], "hardware": manifest["hardware"], "strict": strict,
               "operators": manifest["operators"], "cases": cases,
               "physical_gpu_validated": not write_only and manifest["runtime"] == "cuda" and status == "passed"
                   and all(case["run"].get("physical_gpu_validated") is True for case in cases),
               "observations": sum(case.get("run", {}).get("observations", 0) for case in cases),
               "evaluations": sum(case.get("run", {}).get("evaluations", 0) for case in cases),
               "validation_evaluations": sum(case.get("run", {}).get("validation_evaluations", 0) for case in cases),
               "manifest": str(output / "suite_manifest.json"),
               "notice": "CPU results verify the Python contracts, not GPU kernels. CUDA compilation and performance require the physical device."}
    _save(output / "suite_summary.json", summary)
    lines = ["# Proposal operator suite", "", f"Status: **{status}**; runtime: **{manifest['runtime']}**; scale: **{manifest['scale']}**", "",
             "| Operator / case | Dtype | Status | Run report |", "| --- | --- | --- | --- |"]
    for case in cases:
        relative = (Path(case["spec"]).relative_to(output) if write_only else
                    Path(case["output"]).relative_to(output) / "report.md")
        label = "spec" if write_only else "report"
        lines.append(f"| {case['name']} | {case['dtype']} | {case['status']} | [{label}]({relative.as_posix()}) |")
    lines += ["", summary["notice"], ""]
    (output / "suite_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def cli(args):
    return run_suite(args.output, runtime=args.runtime, scale=args.scale, hardware=args.hardware,
                     operators=tuple(args.operators or ()), algorithms=tuple(args.algorithms), dtypes=tuple(args.dtypes or ()),
                     max_evaluations=args.max_evaluations, max_seconds=args.max_seconds,
                     validation_evaluations=args.validation_evaluations,
                     workspace_memory_limit_bytes=args.workspace_limit_mib * 1024**2,
                     variants=args.variants, plugins=tuple(args.plugin), write_only=args.write_only,
                     check_only=args.check_only, strict=not args.allow_config_failures)
