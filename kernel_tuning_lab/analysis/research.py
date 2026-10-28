"""Verified online-search reports from exact current v2 attempt identities.

No oracle replay, predictions, source remeasurement, or plugin imports occur here.
Historical observations enter only when the current trial actually revealed them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

from ..models import MeasurementRecord, canonical_json, stable_id
from ..measurement_context import source_context_id
from ..research import read_json, snapshot_directory
from ..storage import ExperimentStore
from ..asset_runner import _write
from .report import _bar_chart, _line_chart, _md, _table


def _jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical_json(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items() if key in columns})


def _trial(output, job, raw, store, directory):
    trace = raw.get("observation_trace")
    if trace is None:
        trace = _jsonl(directory / "observations.jsonl")
    ids = raw.get("record_ids", [row["record_id"] for row in trace])
    if trace and ids != [row["record_id"] for row in trace]:
        raise ValueError(f"trial observation trace disagrees with record manifest: {raw['trial_id']}")
    records = []
    for rid in ids:
        record = store.get_record(rid)
        if record is None:
            raise ValueError(f"trial references missing record {rid}; restore its v2 journal")
        # AssetCase.id includes the Kernel ref; MeasurementRecord.case_id is
        # the hardware/operator/shape identity and deliberately differs.
        asset_case_id = stable_id([raw["kernel_ref"], record.hardware.id, record.operator.id, record.shape.id])
        if record.id != rid or asset_case_id != raw["case_id"] or record.operator.implementation != raw["kernel_ref"]:
            raise ValueError("record content/case/kernel identity disagrees with trial")
        if record.seed != job["seed"] or record.shape.dims != raw["shape"] or record.operator.dtype != job["dtype"]:
            raise ValueError("trial references a different seed, shape or dtype")
        records.append(record)
    contexts = {source_context_id(r) for r in records}
    cases = {r.case_id for r in records}
    comparable = len(contexts) == len(cases) == 1
    context = next(iter(contexts)) if comparable else None
    first = records[0] if records else None
    training = raw.get("training_data")
    series_id = stable_id([job["algorithm"], raw.get("algorithm_ref"), raw.get("algorithm_fingerprint"), raw.get("options", {}),
                           training.get("record_ids_sha256") if training else None])
    info = {"job_id": job["id"], "case": job["case"], "trial_id": raw["trial_id"],
        "operator": first.operator.kind if first else job["operator"].split("@")[0],
        "operator_parameters": dict(first.operator.parameters) if first else job["experiment"]["parameters"],
        "kernel": raw.get("kernel_ref", job["kernel"]), "algorithm": raw.get("algorithm_ref", job["algorithm"]),
        "algorithm_fingerprint": raw.get("algorithm_fingerprint"), "algorithm_options": raw.get("options", {}),
        "training_data": training,
        "algorithm_selector": job["experiment"]["algorithms"][0],
        "series_id": series_id, "shape": raw["shape"], "dtype": job["dtype"], "seed": job["seed"],
        "budget": job["budget"], "status": raw.get("status", "interrupted"),
        "stop_reason": raw.get("stop_reason", "incomplete_trial"),
        "hardware": first.hardware.name if first else None,
        "hardware_id": first.hardware.id if first else None,
        "measurement_mode": first.measurement_mode if first else None,
        "source_context_id": context,
        "comparison_id": stable_id([job["case"], first.case_id, context, first.split]) if first and comparable else None,
        "trace_available": len(trace) == len(records), "record_ids": ids,
        "observations": len(records), "cache_hits": sum(row["cache_hit"] for row in trace) if trace else raw.get("cache_hits", 0),
        "evaluations": sum(not row["cache_hit"] for row in trace) if trace else raw.get("evaluations", 0),
        "wall_time_s": raw.get("wall_time_s"), "space_cardinality": raw.get("space_cardinality"),
        "failed_measurements": sum(not r.successful for r in records),
        "declared_or_unverified_successes": sum(r.successful and not r.correctness_verified for r in records),
        "statuses": dict(Counter(r.status.value for r in records)),
        "physical_compile_time_ms": sum(row["physical_compile_time_ms"] for row in trace) if trace else None,
        "physical_measurement_wall_time_s": sum(row["physical_measurement_wall_time_s"] for row in trace) if trace else None,
        "source_compile_time_ms": sum(r.compile_time_ms for r in records),
        "source_measurement_wall_time_s": sum(r.wall_time_s for r in records),
        "algorithm_costs_ms": raw.get("algorithm_costs_ms"),
        "algorithm_diagnostics": raw.get("algorithm_diagnostics", {}),
        "algorithm_runtime": raw.get("algorithm_runtime"),
        "observations_trace": [], "errors": raw.get("failures", [])}
    if raw.get("status") == "running":
        info["status"] = "interrupted"
    if raw.get("status") == "passed" and not records and job["budget"]["max_evaluations"] > 0:
        info["status"] = "partial"
    if records and not comparable:
        info.update(status="failed", stop_reason="incompatible_measurement_protocols")
        info["errors"].append({"phase": "report", "error": "trial mixes source/timing/hardware contexts; no ranking produced"})
    best = None
    for index, record in enumerate(records):
        verified = bool(record.correctness_verified and comparable)
        if verified and (best is None or record.median_ms < best.median_ms):
            best = record
        event = trace[index] if trace else {}
        info["observations_trace"].append({**event, "observation": index + 1, "record_id": record.id,
            "config": dict(record.config.params), "config_id": record.config.id, "status": record.status.value,
            "correctness_verified": verified, "failure_reason": record.failure_reason,
            "latency_ms": record.median_ms if verified else None,
            "p10_ms": record.quantile_ms(0.1) if verified else None,
            "p90_ms": record.quantile_ms(0.9) if verified else None,
            "best_verified_latency_ms": best.median_ms if best else None,
            "verified_on_gpu": record.metadata.get("correctness", {}).get("verified_on_gpu") is True})
    info.update(best_verified_latency_ms=best.median_ms if best else None,
                best_verified_config=dict(best.config.params) if best else None,
                best_verified_record_id=best.id if best else None)
    return info


def generate_research_report(output_dir):
    """Regenerate only the current suite attempt, with no dependency on CUDA."""
    output = Path(output_dir)
    manifest = read_json(output / "research_manifest.json")
    trials, jobs, failures, validation_records, incomplete = [], [], [], [], []
    for job in manifest["jobs"]:
        item = {key: value for key, value in job.items() if key != "experiment"}
        snapshot = snapshot_directory(output, job)
        raw_trials = []
        if snapshot is not None:
            item["snapshot"] = snapshot.relative_to(output).as_posix()
            preflight = snapshot / "preflight.json"
            item["preflight_failures"] = [c for c in read_json(preflight).get("checks", [])
                                           if c.get("status") == "failed"] if preflight.exists() else []
            path = snapshot / "trials.json"
            raw_trials = read_json(path) if path.exists() else []
            known = {t["trial_id"] for t in raw_trials}
            locations = {}
            for path in (snapshot / "trials").glob("*/trial.json"):
                raw = read_json(path)
                locations[raw["trial_id"]] = path.parent
                if raw["trial_id"] not in known:
                    raw_trials.append(raw)
                    known.add(raw["trial_id"])
            database = output / job["output"] / "measurements.sqlite"
            if raw_trials and not database.is_file():
                raise ValueError(f"missing measurement journal for job {job['id']}")
            if raw_trials:
                with ExperimentStore(database, read_only=True) as store:
                    for raw in raw_trials:
                        directory = locations.get(raw["trial_id"], snapshot / "trials")
                        trial = _trial(output, job, raw, store, directory)
                        trials.append(trial)
                        for row in trial["observations_trace"]:
                            if not row["correctness_verified"] and row["status"] != "success":
                                failures.append({"kind": "measurement", "job_id": job["id"], "trial_id": trial["trial_id"], **row})
                        failures.extend({"kind": "algorithm_or_runner", "job_id": job["id"], **error} for error in trial["errors"])
            acceptance = snapshot / "validation_report.json"
            item["validation_wall_time_s"] = read_json(acceptance).get("wall_time_s", 0.0) if acceptance.exists() else 0.0
            validation_rows = _jsonl(snapshot / "validation_records.jsonl")
            item["validation_evaluations"] = len(validation_rows)
            for raw in validation_rows:
                record = MeasurementRecord.from_dict(raw)
                validation_records.append(record)
                if not record.successful:
                    failures.append({"kind": "validation", "job_id": job["id"], "record_id": record.id,
                                     "status": record.status.value, "reason": record.failure_reason,
                                     "compile_time_ms": record.compile_time_ms, "wall_time_s": record.wall_time_s})
            interrupted = snapshot / "interrupted_attempt.json"
            if interrupted.exists():
                failures.append({"kind": "interrupted_attempt", "job_id": job["id"],
                                 "evidence": interrupted.relative_to(output).as_posix(), **read_json(interrupted)})
                if database.is_file():
                    with ExperimentStore(database, read_only=True) as store:
                        for record in store.iter_records(status="runtime_error"):
                            if record.metadata.get("termination_run_id") == job["run_id"]:
                                incomplete.append({"job_id": job["id"], "record_id": record.id,
                                    "wall_time_s": record.wall_time_s, "phase": record.metadata.get("timeout_phase"),
                                    "stage": "research" if record.metadata.get("pending_trial_id") else "validation",
                                    "compile_cost_known": False, "reason": record.failure_reason})
        shape_ids = {canonical_json(t["shape"]) for t in raw_trials}
        for shape in job["shapes"]:
            if canonical_json(shape) not in shape_ids:
                trials.append({"job_id": job["id"], "case": job["case"], "operator": job["operator"].split("@")[0],
                    "kernel": job["kernel"], "algorithm": job["algorithm"], "dtype": job["dtype"],
                    "shape": shape, "seed": job["seed"], "budget": job["budget"],
                    "status": job["status"] if job["status"] in {"not_applicable", "deferred"} else "not_run",
                    "stop_reason": "no research trial executed; see job preflight/validation/status",
                    "observations": 0, "evaluations": 0, "cache_hits": 0, "failed_measurements": 0,
                    "best_verified_latency_ms": None, "observations_trace": []})
        if job["status"] != "passed" and manifest["status"] != "generated":
            failures.append({"kind": "job", "job_id": job["id"], "status": job["status"],
                             "result": job.get("result"), "snapshot": item.get("snapshot"),
                             "preflight_failures": item.get("preflight_failures", [])})
        jobs.append(item)
    groups = defaultdict(list)
    for trial in trials:
        if trial.get("comparison_id"):
            groups[trial["comparison_id"]].append(trial)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    comparisons = []
    for identity, members in sorted(groups.items()):
        first = members[0]
        scope = f"{first['hardware']} / {first['measurement_mode']} / {canonical_json(first['shape'])}"
        title = f"{first['case'][:32]} / {first['kernel']} / {first['dtype']}"
        comparison = {"id": identity, "case": first["case"], "operator": first["operator"],
            "operator_parameters": first["operator_parameters"], "kernel": first["kernel"],
            "dtype": first["dtype"], "shape": first["shape"], "hardware": first["hardware"],
            "source_context_id": first["source_context_id"], "measurement_mode": first["measurement_mode"],
            "figures": [], "aggregates": []}
        curves, aggregates = defaultdict(list), defaultdict(list)
        for trial in members:
            curves[trial["seed"], canonical_json(trial["budget"])].append(trial)
            aggregates[canonical_json(trial["budget"]), trial["series_id"]].append(trial)
        for (seed, budget), rows in sorted(curves.items()):
            token = stable_id([identity, seed, budget])
            labels = {t["series_id"]: t["algorithm"] + " [" + t["algorithm_fingerprint"][:6] + "]" + (" [partial]" if t["status"] != "passed" else "") +
                      (" " + canonical_json(t["algorithm_options"]) if t["algorithm_options"] else "") +
                      (" [train:" + t["training_data"]["record_ids_sha256"][:6] + "]" if t["training_data"] else "") for t in rows}
            for xkey, suffix, xlabel in (("observation", "search", "Revealed observations (failures/cache included)"),
                                        ("elapsed_wall_time_s", "wall", "Actual trial elapsed time (s; shared caches)")):
                filename = f"{suffix}-{token}.svg"
                series = {t["series_id"]: [(e.get(xkey), e["best_verified_latency_ms"] * 1000)
                    for e in t["observations_trace"] if e.get(xkey) is not None and e["best_verified_latency_ms"] is not None]
                    for t in rows}
                _line_chart(figures / filename, f"Search: {title}", scope + f" / seed={seed}; budget={budget}",
                            series, xlabel, "Best verified latency (us)", series_labels=labels, render_png=False)
                comparison["figures"].append("figures/" + filename)
        for (budget, series), rows in sorted(aggregates.items()):
            eligible = [t for t in rows if t["status"] == "passed" and t["best_verified_latency_ms"] is not None]
            values = sorted(t["best_verified_latency_ms"] for t in eligible)
            comparison["aggregates"].append({"algorithm": rows[0]["algorithm"], "series_id": series,
                "algorithm_options": rows[0]["algorithm_options"], "algorithm_fingerprint": rows[0]["algorithm_fingerprint"],
                "training_data": rows[0]["training_data"],
                "algorithm_selector": rows[0]["algorithm_selector"],
                "budget": json.loads(budget), "successful_seeds": [t["seed"] for t in eligible],
                "attempted_seeds": [t["seed"] for t in rows], "n": len(values),
                "median_best_verified_ms": statistics.median(values) if values else None,
                "min_best_verified_ms": min(values) if values else None,
                "max_best_verified_ms": max(values) if values else None})
        # Keep failed/deferred/unstarted selectors in the requested denominator,
        # even when they produced no record from which to infer a protocol.
        first_job = next(j for j in manifest["jobs"] if j["id"] == first["job_id"])
        roster = defaultdict(list)
        for job in manifest["jobs"]:
            if job["group_id"] == first_job["group_id"]:
                selector = job["experiment"]["algorithms"][0]
                roster[canonical_json(job["budget"]), canonical_json(selector)].append(job)
        for (budget, selector), expected_jobs in sorted(roster.items()):
            matching = [a for a in comparison["aggregates"] if canonical_json(a["budget"]) == budget and
                        canonical_json(a["algorithm_selector"]) == selector]
            if not matching:
                selected = json.loads(selector)
                missing = {"algorithm": selected["asset"], "algorithm_selector": selected,
                    "algorithm_options": selected["options"], "algorithm_fingerprint": None,
                    "training_data": None,
                    "series_id": stable_id([selector, identity]), "budget": json.loads(budget),
                    "successful_seeds": [], "attempted_seeds": [], "n": 0,
                    "median_best_verified_ms": None, "min_best_verified_ms": None, "max_best_verified_ms": None}
                comparison["aggregates"].append(missing)
                matching = [missing]
            for aggregate in matching:
                aggregate["requested_seeds"] = sorted({j["seed"] for j in expected_jobs})
                aggregate["job_seed_statuses"] = {str(j["seed"]): j["status"] for j in expected_jobs}
        for budget in sorted({canonical_json(a["budget"]) for a in comparison["aggregates"]}):
            rows = [a for a in comparison["aggregates"] if canonical_json(a["budget"]) == budget and a["n"]]
            filename = f"performance-{stable_id([identity, budget])}.svg"
            _bar_chart(figures / filename, f"Observed best: {title}", scope + f" / budget={budget}; valid seed counts shown",
                [a["algorithm"] + " [" + a["algorithm_fingerprint"][:6] + "]" +
                 (" " + canonical_json(a["algorithm_options"]) if a["algorithm_options"] else "") +
                 f" (n={a['n']}/{len(a['requested_seeds'])})" for a in rows], [a["median_best_verified_ms"]*1000 for a in rows],
                xlabel="Median best verified latency across successful seeds (us)", render_png=False)
            comparison["figures"].append("figures/" + filename)
        comparisons.append(comparison)
    statuses = Counter(j["status"] for j in jobs)
    measured_failures = sum(t["failed_measurements"] for t in trials)
    trial_errors = any(t["status"] == "failed" for t in trials)
    status = ("generated" if manifest["status"] == "generated" else
              "interrupted" if statuses["interrupted"] else
              "failed" if statuses["failed"] or statuses["timeout"] or measured_failures or trial_errors else
              "partial" if any(s != "passed" for s in statuses) or manifest["status"] != "finished" or
                  any(t["status"] in {"partial", "not_run", "interrupted"} for t in trials) else "passed")
    interrupted_code = next((j.get("result", {}).get("exit_code", 130) for j in jobs if j["status"] == "interrupted"), 130)
    summary = {"schema_version": 1, "name": manifest["name"], "status": status,
        "exit_code": interrupted_code if status == "interrupted" else 2 if status == "failed" else 3 if status == "partial" else 0,
        "run_id": manifest["run_id"], "git_commit": manifest.get("git_commit"), "planned": manifest["planned"],
        "job_statuses": dict(statuses), "jobs": jobs, "trials": trials, "trial_count": len(trials),
        "executed_trial_count": sum("trial_id" in t for t in trials),
        "trial_statuses": dict(Counter(t["status"] for t in trials)),
        "observations": sum(t["observations"] for t in trials),
        "research_evaluations": sum(t["evaluations"] for t in trials) + sum(a["stage"] == "research" for a in incomplete), "cache_hits": sum(t["cache_hits"] for t in trials),
        "validation_evaluations": sum(j.get("validation_evaluations", 0) for j in jobs) + sum(a["stage"] == "validation" for a in incomplete),
        "validation_wall_time_s": sum(j.get("validation_wall_time_s", 0.0) for j in jobs),
        "job_wall_time_s": sum(j.get("wall_time_s", 0.0) for j in jobs),
        "failed_measurement_observations": measured_failures, "failure_events": len(failures),
        "known_physical_compile_time_ms": sum(t.get("physical_compile_time_ms") or 0 for t in trials),
        "unknown_compile_cost_attempts": len(incomplete) + sum(t["observations"] > 0 and not t.get("trace_available", False) for t in trials),
        "physical_compile_time_ms": None if incomplete or any(t["observations"] and not t.get("trace_available", False) for t in trials)
            else sum(t.get("physical_compile_time_ms") or 0 for t in trials),
        "physical_measurement_wall_time_s": sum(t.get("physical_measurement_wall_time_s") or 0 for t in trials) + sum(a["wall_time_s"] for a in incomplete if a["stage"] == "research"),
        "algorithm_costs_ms": {key: sum((t.get("algorithm_costs_ms") or {}).get(key, 0) for t in trials)
            for key in ("factory_ms", "proposal_ms", "feedback_ms")},
        "algorithm_cost_trace_complete": all(t.get("algorithm_costs_ms") is not None for t in trials if "trial_id" in t),
        "incomplete_attempts": incomplete,
        "cost_trace_complete": not incomplete and not statuses["timeout"] and not statuses["interrupted"] and all(t.get("trace_available", t["observations"] == 0) for t in trials),
        "physical_gpu_validated": status == "passed" and bool(jobs) and
            all(j.get("result", {}).get("physical_gpu_validated") is True for j in jobs) and
            any(t["observations"] for t in trials) and
            all(row["correctness_verified"] and row["verified_on_gpu"] for t in trials for row in t["observations_trace"]),
        "gpu_verified_jobs": sum(j.get("result", {}).get("physical_gpu_validated") is True for j in jobs),
        "comparison_groups": comparisons,
        "notice": "Only revealed, reference-verified observations rank configurations. Each protocol/hardware/operator/shape/dtype context is separate. Cached source cost is not new cost; compile is a subset of measurement wall time. No global optimum or independent cold-start cost claim."}
    observations = [{"job_id": t["job_id"], "trial_id": t.get("trial_id"), "case": t["case"],
        "algorithm": t["algorithm"], "kernel": t["kernel"], "dtype": t["dtype"], "seed": t["seed"],
        "shape": t["shape"], "measurement_mode": t.get("measurement_mode"),
        "source_context_id": t.get("source_context_id"), **row} for t in trials for row in t["observations_trace"]]
    _csv(output / "trials.csv", trials, ["job_id", "trial_id", "case", "operator", "kernel", "algorithm", "dtype", "shape", "seed",
        "operator_parameters", "budget", "status", "stop_reason", "observations", "evaluations", "cache_hits", "best_verified_latency_ms",
        "measurement_mode", "source_context_id", "wall_time_s", "physical_compile_time_ms", "physical_measurement_wall_time_s",
        "algorithm_costs_ms", "algorithm_diagnostics", "algorithm_runtime"])
    _csv(output / "observations.csv", observations, ["job_id", "trial_id", "case", "algorithm", "kernel", "dtype", "shape", "seed",
        "source_context_id", "measurement_mode", "observation", "record_id", "config", "status", "cache_hit",
        "correctness_verified", "latency_ms", "p10_ms", "p90_ms", "best_verified_latency_ms", "elapsed_wall_time_s",
        "lookup_and_measurement_wall_time_s",
        "physical_compile_time_ms", "physical_measurement_wall_time_s", "source_compile_time_ms", "source_measurement_wall_time_s", "failure_reason"])
    (output / "failures.jsonl").write_text("".join(canonical_json(f) + "\n" for f in failures), encoding="utf-8")
    _write(output / "trials.json", trials)
    _write(output / "summary.json", summary)
    _write(output / "run_summary.json", {key: val for key, val in summary.items() if key not in {"jobs", "trials", "comparison_groups"}})
    _bar_chart(figures / "cost.svg", "Current research attempt costs", "Actual observed costs; compilation is included in measurement wall time",
        ["All v2 job wall time", "Validation wall time", "New research measurement wall time", "New research compile time (subset)",
         "Algorithm factory", "Candidate generation", "Optimizer feedback"],
        [summary["job_wall_time_s"], summary["validation_wall_time_s"], summary["physical_measurement_wall_time_s"],
         summary["known_physical_compile_time_ms"]/1000, *[v/1000 for v in summary["algorithm_costs_ms"].values()]],
        xlabel="Known seconds (components overlap; incomplete costs flagged)", render_png=False)
    counts = Counter(f.get("status", f["kind"]) for f in failures)
    _bar_chart(figures / "failures.svg", "Failure and applicability events", "Job/validation/measurement events are separate; cached observations may repeat failures",
        list(counts), list(counts.values()), xlabel="Event count", render_png=False)
    lines = ["# Research comparison", "", f"Status: **{status}**; physical GPU gate: **{summary['physical_gpu_validated']}**.", "",
        summary["notice"], "", f"Planned jobs/trials: {manifest['planned']['jobs']} / {manifest['planned']['trials']}. "
        f"Observations/new attempts/cache hits: {summary['observations']} / {summary['research_evaluations']} / {summary['cache_hits']}. "
        f"Validation attempts: {summary['validation_evaluations']}.", "",
        "CPU measurements verify CPU execution only. Failures, unsupported declarations and missing GPU are not successful GPU validation.", "",
        _table(("Case", "Kernel", "Algorithm", "Dtype", "Seed", "Budget (observations / seconds)", "Job status", "Evidence"),
            ((j["case"], j["kernel"], j["algorithm"], j["dtype"], j["seed"],
              f"{j['budget']['max_evaluations']} / {j['budget']['max_wall_time_s']}", j["status"],
              f"[attempt]({j['snapshot']}/report.md)" if j.get("snapshot") else "not executed") for j in jobs)), "",
        "## Costs and failures", "", "![Costs](figures/cost.svg)", "", "![Failures](figures/failures.svg)", "",
        "Shared measurement/JIT caches make execution order affect physical tuning cost. Equal observation allowances do not imply equal cold-start cost. "
        "Failed attempts consume budgets; terminated attempts may have incomplete compile costs (see v2 interrupted evidence). "
        "Source measurement cost describes the original observation; cache hits add lookup/wall overhead, not another compilation.", ""]
    lines += ["Algorithm factory / proposal / feedback costs (ms): `" + canonical_json(summary["algorithm_costs_ms"]) + "`. "
              "External model/process calls and retries are included in proposal cost; they are not added twice. "
              f"Algorithm cost trace complete: **{summary['algorithm_cost_trace_complete']}**. Older records lacking this trace remain unknown.", ""]
    rejected = [j for j in jobs if j["status"] not in {"passed", "not_run"}]
    if rejected:
        lines += [_table(("Case", "Algorithm", "Status", "Diagnostic"),
            ((j["case"], j["algorithm"], j["status"],
              "; ".join(c.get("error", {}).get("message", "") for c in j.get("preflight_failures", [])) or
              canonical_json(j.get("result", {}).get("error", {}))) for j in rejected)), ""]
    for group in comparisons:
        lines += [f"## {_md(group['case'])} / {_md(group['kernel'])} / {_md(group['dtype'])} / {_md(canonical_json(group['shape']))}", "",
            f"{_md(group['hardware'])}; `{group['measurement_mode']}`; source context `{group['source_context_id']}`.", "",
            f"Operator parameters: `{_md(canonical_json(group['operator_parameters']))}`.", "",
            _table(("Algorithm / options", "Budget (observations / seconds)", "Valid / requested seeds", "Median best ms", "Min / max ms"),
                ((a["algorithm"] + (" " + canonical_json(a["algorithm_options"]) if a["algorithm_options"] else ""),
                  f"{a['budget']['max_evaluations']} / {a['budget']['max_wall_time_s']}", f"{a['n']} / {len(a['requested_seeds'])}",
                  a["median_best_verified_ms"], f"{a['min_best_verified_ms']} / {a['max_best_verified_ms']}") for a in group["aggregates"])), ""]
        lines += [f"![Observed comparison]({path})\n" for path in group["figures"]]
    lines += ["", "[Summary](summary.json) · [Trials CSV](trials.csv) · [Observation CSV](observations.csv) · "
              "[Failure events](failures.jsonl) · [Frozen experiment](experiment.json) · [Job manifest](research_manifest.json)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    # V2 moves the last live snapshot into attempts/ on the next invocation.
    # Repair only presentation links in older suite reports, preserving their
    # observations, manifests and the original v2 journals.
    for previous in (output / "attempts").glob("*/research_manifest.json"):
        report = previous.parent / "report.md"
        if not report.exists():
            continue
        text = report.read_text(encoding="utf-8")
        for job in read_json(previous)["jobs"]:
            snapshot = snapshot_directory(output, job)
            if snapshot is not None:
                old = f"](../../{job['output']}/report.md)"
                new = f"](../../{snapshot.relative_to(output).as_posix()}/report.md)"
                text = text.replace(old, new)
        report.write_text(text, encoding="utf-8")
    return {key: val for key, val in summary.items() if key not in {"jobs", "trials", "comparison_groups"}} | {"report": str(output / "report.md")}
