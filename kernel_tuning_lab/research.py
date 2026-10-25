"""Serial, configuration-driven comparisons over supervised v2 experiments.

Each algorithm runs independently; compatible trials use the same existing v2
journal. A restart replays suggestions against that cache, rather than trusting
stale completion flags or bypassing current source/environment validation.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import time

from .asset_experiments import AssetExperiment, _json_constant, _json_pairs, _keys
from .asset_runner import _plugin_reference, _write, execute_asset_experiment
from .models import canonical_json, stable_id
from .run_lifecycle import OutputDirectoryBusyError, OutputLease, archive_previous_attempt


DEFAULT_FIELDS = {"hardware", "warmup", "repetitions", "split", "timing_mode", "cuda_graph_batch_size",
                  "workspace_memory_limit_bytes", "validation", "preflight_seconds"}
CASE_FIELDS = DEFAULT_FIELDS | {"name", "operator", "kernels", "shapes", "parameters", "dtypes"}
ROOT_FIELDS = {"schema_version", "kind", "name", "plugins", "algorithms", "dtypes", "seeds", "budgets",
               "defaults", "cases", "max_jobs"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_json_pairs,
                      parse_constant=_json_constant)


def _distinct(values, label):
    if not isinstance(values, list) or not values or len({canonical_json(x) for x in values}) != len(values):
        raise ValueError(f"{label} must be a nonempty list of distinct values")
    return values


def expand_research(value, *, directory=Path("."), plugins=()):
    """Validate using the v2 schema; expansion imports no plugins or GPU SDKs."""
    _keys(value, ROOT_FIELDS, "research experiment")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1 or value.get("kind") != "research":
        raise ValueError("research requires schema_version=1 and kind=research")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("research name is required")
    references = value.get("plugins", [])
    if not isinstance(references, list) or any(not isinstance(p, str) or not p for p in references):
        raise ValueError("plugins must be a list of module/file references")
    references = tuple(dict.fromkeys(_plugin_reference(p, Path(directory)) for p in [*references, *plugins]))
    algorithms = _distinct(value.get("algorithms"), "algorithms")
    dtypes = _distinct(value.get("dtypes", ["float16"]), "dtypes")
    seeds = _distinct(value.get("seeds", [0]), "seeds")
    if any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must contain nonnegative integers")
    budgets = _distinct(value.get("budgets"), "budgets")
    for budget in budgets:
        _keys(budget, {"max_evaluations", "max_wall_time_s"}, "research budget")
        if set(budget) != {"max_evaluations", "max_wall_time_s"}:
            raise ValueError("each research budget requires max_evaluations and finite max_wall_time_s")
    defaults = value.get("defaults", {})
    _keys(defaults, DEFAULT_FIELDS, "research defaults")
    cases = _distinct(value.get("cases"), "cases")
    names = set()
    limit = value.get("max_jobs", 1024)
    if type(limit) is not int or limit < 1:
        raise ValueError("max_jobs must be a positive integer")
    jobs = []
    for case in cases:
        _keys(case, CASE_FIELDS, "research case")
        case_name = case.get("name")
        if not isinstance(case_name, str) or not case_name.strip() or case_name in names:
            raise ValueError("case names must be nonempty and unique")
        names.add(case_name)
        case_dtypes = _distinct(case.get("dtypes", dtypes), "case dtypes")
        kernels = _distinct(case.get("kernels"), "case kernels")
        # Validate one full v2 template first, including all shapes and selections.
        common = {"split": "validation", "validation": {"trust_claims": False}, **defaults,
                  **{key: val for key, val in case.items() if key not in {"name", "dtypes"}}}
        common["validation"] = {"trust_claims": False, **defaults.get("validation", {}), **case.get("validation", {})}
        template = AssetExperiment.from_dict({**common, "schema_version": 2, "name": name,
            "plugins": list(references), "algorithms": algorithms, "dtype": case_dtypes[0],
            "seed": seeds[0], "budget": budgets[0]})
        count = len(kernels) * len(case_dtypes) * len(seeds) * len(budgets) * len(algorithms)
        if len(jobs) + count > limit:
            raise ValueError(f"research expansion exceeds max_jobs={limit}; inspect dimensions before increasing it")
        for kernel in template.kernels:
            for dtype in case_dtypes:
                if not isinstance(dtype, str) or not dtype:
                    raise ValueError("dtypes must contain nonempty strings")
                # Stable journal ownership even when algorithms, Shapes or
                # protocols are added. The existing record cache key enforces
                # actual source/hardware/protocol/space/seed compatibility.
                group_id = stable_id([name, case_name, template.operator, kernel.asset, dtype])
                for seed in seeds:
                    for budget in budgets:
                        for algorithm in template.algorithms:
                            spec = AssetExperiment.from_dict({**template.to_dict(), "name": name + "-" + group_id,
                                "kernels": [asdict(kernel)], "algorithms": [asdict(algorithm)],
                                "dtype": dtype, "seed": seed, "budget": budget})
                            job_id = stable_id(spec.to_dict())
                            jobs.append({"id": job_id, "group_id": group_id, "case": case_name,
                                "operator": spec.operator, "kernel": kernel.asset, "algorithm": algorithm.asset,
                                "dtype": dtype, "seed": seed, "budget": asdict(spec.budget),
                                "shapes": [dict(s.dims) for s in spec.shapes], "status": "not_run",
                                "spec": f"specs/{job_id}.json", "output": f"runs/{group_id}",
                                "experiment": spec.to_dict()})
    planned = {"jobs": len(jobs), "groups": len({j["group_id"] for j in jobs}),
        "trials": sum(len(j["shapes"]) for j in jobs),
        "max_observations": sum(len(j["shapes"]) * j["budget"]["max_evaluations"] for j in jobs),
        "max_validation_evaluations": sum(j["experiment"]["validation"]["max_configs"] for j in jobs),
        "serial_wall_allowance_s": sum(j["experiment"]["preflight_seconds"] +
            j["experiment"]["validation"]["max_wall_time_s"] + len(j["shapes"]) * j["budget"]["max_wall_time_s"] + 5 for j in jobs)}
    return {"schema_version": 1, "name": name, "status": "generated", "serial": True,
            "planned": planned, "jobs": jobs,
            "resume_semantics": "revalidate each v2 job; replay suggestions using completed compatible cache; acceptance is separately charged"}


def snapshot_directory(output, job):
    group = Path(output) / job["output"]
    run_id = job.get("run_id")
    if not run_id:
        return None
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("invalid v2 run identity in research manifest")
    current = group / "run_metadata.json"
    if current.exists() and read_json(current).get("run_id") == run_id:
        return group
    archived = group / "attempts" / run_id
    if not archived.is_dir():
        raise ValueError(f"missing archived v2 attempt for research job {job['id']}")
    return archived


def _job_status(output, result):
    if not result.get("run_id"):
        return result["status"]
    preflight = Path(output) / "preflight.json"
    failures = [c for c in read_json(preflight).get("checks", []) if c.get("status") == "failed"] if preflight.exists() else []
    if failures and all(c.get("error", {}).get("type") == "AssetNotApplicableError" for c in failures):
        return "not_applicable"
    return result["status"]


def _archive(output):
    previous = archive_previous_attempt(output)
    if previous:
        for name in ("research_manifest.json", "summary.json", "figures", "observations.csv", "trials.csv", "failures.jsonl"):
            path = output / name
            if path.exists():
                path.replace(previous / name)
        report = previous / "report.md"
        if report.exists():
            report.write_text(report.read_text(encoding="utf-8").replace("](runs/", "](../../runs/"), encoding="utf-8")
        _write(previous / "archive.json", {"journal_root": "../..",
            "notice": "Group journals remain at the retained research root; manifest job paths are relative to that root. Figures/CSVs here are this archived attempt."})


def _report(output):
    from .analysis.research import generate_research_report
    try:
        return generate_research_report(output)
    except Exception as exc:
        result = {"status": "failed", "phase": "report", "exit_code": 2, "physical_gpu_validated": False,
                  "error": {"type": type(exc).__name__, "message": str(exc)}}
        _write(output / "run_summary.json", result)
        _write(output / "summary.json", result)
        (output / "report.md").write_text("# Research report failed\n\nSee [diagnostic](summary.json). "
            "The v2 journals and attempt snapshots were retained. No successful report is asserted.\n", encoding="utf-8")
        return result


def run_research(spec_path, output_dir, *, write_only=False, report_only=False, plugins=()):
    output = Path(output_dir).resolve()
    try:
        with OutputLease(output) as lease:
            if report_only:
                _write(output / "run_summary.json", {"status": "reporting", "physical_gpu_validated": False})
                return _report(output)
            if spec_path is None:
                raise ValueError("research requires --spec (unless --report-only)")
            source = Path(spec_path).resolve()
            try:
                raw = read_json(source)
                # Freeze/resolve before moving a spec supplied from this output itself.
                references = tuple(_plugin_reference(p, Path.cwd()) for p in plugins)
                manifest = expand_research(raw, directory=source.parent, plugins=references)
            except (ValueError, TypeError, OSError, KeyError):
                _archive(output)
                _write(output / "run_summary.json", {"status": "failed", "phase": "configuration", "physical_gpu_validated": False})
                raise
            _archive(output)
            _write(output / "run_metadata.json", lease.metadata)
            _write(output / "source_experiment.json", raw)
            frozen = {**raw, "plugins": manifest["jobs"][0]["experiment"]["plugins"]}
            _write(output / "experiment.json", frozen)
            manifest.update(status="generated" if write_only else "running", run_id=lease.metadata["run_id"])
            try:
                git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
                manifest["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                manifest["git_commit"] = None
            _write(output / "run_summary.json", {"status": manifest["status"], "physical_gpu_validated": False})
            for job in manifest["jobs"]:
                AssetExperiment.from_dict(job["experiment"]).save(output / job["spec"])
            _write(output / "research_manifest.json", manifest)
            if not write_only:
                for job in manifest["jobs"]:
                    job["status"] = "running"
                    _write(output / "research_manifest.json", manifest)
                    started = time.perf_counter()
                    try:
                        result = execute_asset_experiment(output / job["spec"], output / job["output"],
                                                          inherited_fds=(lease.stream.fileno(),))
                    except KeyboardInterrupt:
                        result = {"status": "interrupted", "exit_code": 130,
                                  "error": {"type": "Interrupted", "message": "research interrupted between v2 phases"}}
                    except Exception as exc:
                        result = {"status": "failed", "exit_code": 2,
                                  "error": {"type": type(exc).__name__, "message": str(exc)}}
                    job.update(result=result, status=_job_status(output / job["output"], result),
                               run_id=result.get("run_id"), wall_time_s=time.perf_counter() - started)
                    _write(output / "research_manifest.json", manifest)
                    if result["status"] == "interrupted":
                        break
                manifest["status"] = "finished"
                _write(output / "research_manifest.json", manifest)
            return _report(output)
    except OutputDirectoryBusyError as exc:
        return {"status": "failed", "exit_code": 2, "error": str(exc), "physical_gpu_validated": False}
