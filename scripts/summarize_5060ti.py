"""Historical 5060 Ti evidence summary; new studies use ``ktlab research``.

The generic workflow/report has no dependency on this original acceptance tree.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from kernel_tuning_lab.l3 import PerformanceMatrix, ShapeDispatcher, greedy, greedy_local_swap, oracle
from kernel_tuning_lab.models import MeasurementRecord, WorkloadDistribution
from kernel_tuning_lab.storage import ExperimentStore


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [MeasurementRecord.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line]


def main(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    journals = {p: records(p) for p in root.rglob("measurements.jsonl")}
    assert all(r.provenance == "measured" for rows in journals.values() for r in rows)
    unique = {r.id: r for rows in journals.values() for r in rows if r.provenance == "measured"}
    assert all(r.correctness_verified and r.metadata.get("correctness", {}).get("verified_on_gpu") is True
               for r in unique.values() if r.successful)
    summary = {"notice": "Physical observations only. Sample coverage is not whole-space validation. Graph and full-flow latencies remain separate.",
               "records": len(unique), "statuses": dict(Counter(r.status.value for r in unique.values())),
               "measurement_modes": dict(Counter(r.measurement_mode for r in unique.values())),
               "environments": list({json.dumps(r.environment, sort_keys=True): r.environment for r in unique.values()}.values()),
               "suites": {}, "coverage": [], "l2": [], "l3": None, "representative": []}
    ledger = root / "experiment-ledger.json"
    if ledger.exists():
        summary["experiment_ledger"] = read(ledger)
    smoke = read(root / "baseline" / "core" / "validation_summary.json")
    summary["core_smoke"] = {key: smoke[key] for key in ("status", "physical_gpu_validated", "hardware", "stages")}
    summary["runtime_snapshot"] = read(root / "baseline" / "environment" / "final-runtime.json")
    cache_check = root / "l2-cache-check" / "summary.json"
    if cache_check.exists():
        cache = read(cache_check)
        summary["cache_check"] = {key: cache[key] for key in (
            "status", "observations", "evaluations", "cache_hits", "validation_evaluations")}
        summary["cache_check"].update(read(root / "l2-cache-check" / "sequence-check.json"))
    for p in root.glob("*/suite_summary.json"):
        s = read(p)
        summary["suites"][p.parent.name] = {k: s.get(k) for k in (
            "status", "physical_gpu_validated", "observations", "evaluations", "validation_evaluations", "strict")}
        summary["suites"][p.parent.name]["jobs"] = len(s.get("cases", ()))
    groups = defaultdict(list)
    for r in unique.values():
        groups[r.operator.kind, r.operator.dtype].append(r)
    for (operator, dtype), rows in sorted(groups.items()):
        summary["coverage"].append({"operator": operator, "dtype": dtype,
            "journal_records": len(rows), "statuses": dict(Counter(r.status.value for r in rows)),
            "shapes": list({r.shape.id: r.shape.dims for r in rows}.values()),
            "configs": len({r.config.id for r in rows}), "input_seeds": sorted({r.seed for r in rows}),
            "implementations": sorted({r.operator.implementation for r in rows}),
            "config_shape_parameter_cells": len({(r.config.id, r.shape.id, json.dumps(r.operator.parameters, sort_keys=True), r.operator.implementation) for r in rows}),
            "observed_parameters": list({json.dumps(r.operator.parameters, sort_keys=True): r.operator.parameters for r in rows}.values()),
            "observed_knobs": {k: sorted({json.dumps(r.config.params[k]) for r in rows if k in r.config.params})
                               for k in ("SPLIT_K", "SCHEDULE", "USE_TMA", "GROUP_M", "USE_EXP2", "PRE_LOAD_V")
                               if any(k in r.config.params for r in rows)},
            "whole_space_verified": False})
    for p in sorted((root / "l2").glob("runs/*/trials.json")):
        for trial in read(p):
            rows = [unique[rid] for rid in trial["record_ids"]]
            best = None
            curve = []
            for r in rows:
                if r.correctness_verified:
                    best = r.median_ms if best is None else min(best, r.median_ms)
                curve.append(best)
            summary["l2"].append({"experiment": p.parent.name, **{k: trial[k] for k in (
                "algorithm_ref", "seed", "shape", "observations", "evaluations", "cache_hits", "wall_time_s",
                "best_verified_latency_ms", "best_verified_config", "space_cardinality", "stop_reason", "budget")},
                "config_ids": [r.config.id for r in rows], "best_so_far_ms": curve,
                "failed_measurements": sum(not r.successful for r in rows),
                "source_compile_ms": sum(r.compile_time_ms for r in rows),
                "source_wall_time_s": sum(r.wall_time_s for r in rows)})
    for p in sorted((root / "representative").glob("runs/*/trials.json")):
        for trial in read(p):
            rows = [unique[rid] for rid in trial["record_ids"]]
            best = min((r for r in rows if r.correctness_verified), key=lambda r: r.median_ms, default=None)
            summary["representative"].append({"experiment": p.parent.name,
                **{k: trial[k] for k in ("kernel_ref", "shape", "dtype", "seed", "observations", "evaluations", "cache_hits", "wall_time_s")},
                "samples": [{"config": r.config.params, "median_us": r.median_ms*1000 if r.successful else None,
                    "p10_us": r.quantile_ms(0.1)*1000 if r.successful else None,
                    "p90_us": r.quantile_ms(0.9)*1000 if r.successful else None,
                    "cv": statistics.stdev(r.runtime_samples_ms)/statistics.mean(r.runtime_samples_ms) if r.successful else None,
                    "compile_ms": r.compile_time_ms, "status": r.status.value,
                    "workspace_bytes": r.metadata.get("correctness", {}).get("execution", {}).get("workspace_bytes"),
                    "steps": r.metadata.get("correctness", {}).get("execution", {}).get("steps")} for r in rows],
                "best_verified_config": best.config.params if best else None,
                "best_verified_latency_us": best.median_ms*1000 if best else None})
    l3_rows = [r for p, rows in journals.items() if p.is_relative_to(root / "l3") for r in rows
               if r.seed in (7, 19, 31) and r.correctness_verified]
    if l3_rows:
        shapes = tuple(sorted({r.shape.id: r.shape for r in l3_rows}.values(), key=lambda s: s.get("M")))
        configs = tuple(sorted({r.config.id: r.config for r in l3_rows}.values(), key=lambda c: c.id))
        assert len(shapes) == 3 and len(configs) == 12
        workload = WorkloadDistribution(shapes, (0.4, 0.3, 0.3), "5060ti-limited-observed-gemm")
        matrix = PerformanceMatrix.from_records(workload, configs, l3_rows)
        solutions = [solver(matrix, k).to_dict() for k in (1, 2, 4) for solver in (oracle, greedy, greedy_local_swap)]
        result = oracle(matrix, 2)
        from kernel_tuning_lab.space import gemm_space
        dispatcher = ShapeDispatcher.from_result(matrix, result, hardware=l3_rows[0].hardware,
                                                  operator=l3_rows[0].operator, space=gemm_space())
        assert all(dispatcher(s).id in {c.id for c in result.configs} for s in shapes)
        summary["l3"] = {"matrix": matrix.to_dict(), "solutions": solutions,
                          "notice": "Complete only within these 3 shapes x 12 declared candidates; no global-space optimum or unseen-shape generalization claim."}
        # Exercise real GPU records through lossless SQLite/JSONL exchange.
        exchange = root / "l3" / "roundtrip"
        exchange.mkdir(exist_ok=True)
        with ExperimentStore(exchange / "measurements.sqlite") as store:
            store.add_many(l3_rows)
            store.export_jsonl(exchange / "measurements.jsonl")
        returned = records(exchange / "measurements.jsonl")
        assert {r.id for r in returned} == {r.id for r in l3_rows}
        with ExperimentStore(exchange / "imported.sqlite") as imported:
            imported.import_jsonl(exchange / "measurements.jsonl")
            imported.export_jsonl(exchange / "imported.jsonl")
        assert {r.id for r in records(exchange / "imported.jsonl")} == {r.id for r in l3_rows}
        summary["l3"]["roundtrip_identical_record_ids"] = True
    timing = root / "timing-control-graph-1024" / "timing.json"
    if not timing.is_file():
        timing = root / "timing-control-graph" / "timing.json"
    if timing.is_file():
        summary["timing_graph_batch_size"] = read(timing).get("cuda_graph_batch_size", 128)
        summary["timing"] = []
        for c in read(timing)["cases"]:
            stats = {}
            for key in ("v1_event_ms", "independent_event_ms", "v2_full_flow_ms", "v2_graph_ms", "triton_graph_ms", "torch_mm_graph_ms"):
                rounds = [statistics.median(r[key]) for r in c["rounds"]]
                values = sorted(x for r in c["rounds"] for x in r[key])
                stats[key] = {"median_us": statistics.median(values)*1000,
                    "p10_us": values[int(0.1*(len(values)-1))]*1000,
                    "p90_us": values[int(0.9*(len(values)-1))]*1000,
                    "round_medians_us": [v*1000 for v in rounds],
                    "round_median_cv": statistics.stdev(rounds)/statistics.mean(rounds)}
            summary["timing"].append({"shape": c["shape"], "statistics": stats,
                "telemetry_before": c["telemetry_before"], "telemetry_after": c["telemetry_after"]})
    # Environment dictionaries contain version/device/source hashes only, no local paths.
    (output / "5060ti-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(summary, output.parent / "figures")
    print(json.dumps({"records": summary["records"], "statuses": summary["statuses"],
                      "l2_trials": len(summary["l2"]), "l3": summary["l3"] is not None}, indent=2))


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    if summary.get("timing"):
        fig, ax = plt.subplots(figsize=(9, 4))
        keys = ("v1_event_ms", "v2_full_flow_ms", "v2_graph_ms", "triton_graph_ms")
        labels = ("v1 single Event", "v2 full call", "v2 CUDA Graph", "Triton CUDA Graph")
        for i, key in enumerate(keys):
            ax.bar([j+(i-1.5)*0.18 for j in range(3)],
                   [c["statistics"][key]["median_us"] for c in summary["timing"]], width=0.18, label=labels[i])
        ax.set_xticks(range(3), ["32x64x64", "512x512x512", "1024x1024x1024"])
        ax.set_yscale("log")
        ax.set_ylabel("Median latency (us, log scale)")
        ax.set_title("Physical RTX 5060 Ti: identical GEMM, distinct timing scopes")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "5060ti-timing.svg")
        plt.close(fig)
    trials = [r for r in summary["l2"] if r["seed"] == 7 and r["budget"]["max_evaluations"] == 16]
    if trials:
        fig, ax = plt.subplots(figsize=(7, 4))
        for t in trials:
            ax.plot(range(1, len(t["best_so_far_ms"])+1), [v*1000 for v in t["best_so_far_ms"]],
                    marker=".", label=t["algorithm_ref"])
        ax.set_xlabel("Revealed observations (failures/cache hits included)")
        ax.set_ylabel("Best verified latency (us)")
        ax.set_title("Physical RTX 5060 Ti GEMM: seed 7, CUDA Graph, budget 16")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "5060ti-l2.svg")
        plt.close(fig)
    if summary["l3"]:
        matrix = summary["l3"]["matrix"]
        fig, ax = plt.subplots(figsize=(9, 3))
        rows = matrix["latencies_ms"]
        image = ax.imshow([[v/min(row) for v in row] for row in rows], aspect="auto")
        ax.set_xticks(range(len(matrix["configs"])))
        ax.set_yticks(range(len(rows)), [f"{s['M']}x{s['N']}x{s['K']}" for s in matrix["workload"]["shapes"]])
        ax.set_xlabel("Observed configuration index")
        ax.set_ylabel("M x N x K")
        ax.set_title("Measured 3 x 12 GEMM matrix: latency / observed row minimum")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output / "5060ti-l3.svg")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/5060ti")
    parser.add_argument("--output", default="docs/data")
    args = parser.parse_args()
    main(args.root, args.output)
