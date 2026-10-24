"""One CLI for collection, dataset exchange, replay, policies, L3 and reports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

from .backends import BackendUnavailableError, MockAdapter, get_adapter
from .experiments import ExperimentSpec, demo_hardware, demo_shapes, example_spec
from .models import MeasurementRecord, OperatorSpec, SearchBudget, WorkloadDistribution
from .operators import OPERATOR_NAMES, operator_demo_shapes, operator_space, operator_spec
from .runner import BenchmarkRunner
from .space import ConfigSpace, gemm_constraint, gemm_space, reduction_constraint
from .storage import ExperimentStore


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save_jsonl(path: Path, records: Iterable[MeasurementRecord]) -> None:
    from .models import canonical_json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(canonical_json(record.to_dict()) + "\n")


def _infer_space(records: Iterable[MeasurementRecord]) -> ConfigSpace:
    from .models import canonical_json
    domains: dict[str, dict[str, Any]] = defaultdict(dict)
    operators = set()
    for record in records:
        operators.add(record.operator.kind)
        for key, value in record.config.params.items():
            domains[key][canonical_json(value)] = value
    if not domains:
        raise ValueError("no configuration domains available in this dataset")
    if len(operators) != 1:
        raise ValueError("infer configuration spaces separately for each operator")
    constraints = ((gemm_constraint,) if operators <= {"gemm", "batched_gemm"} else
                   (reduction_constraint,) if operators <= {"softmax", "layernorm"} else ())
    return ConfigSpace({key: tuple(values.values()) for key, values in domains.items()},
                       constraints=constraints, name="inferred_from_observed_domains")


def _operator_groups(records: Iterable[MeasurementRecord]) -> list[list[MeasurementRecord]]:
    groups: dict[tuple[str, str], list[MeasurementRecord]] = defaultdict(list)
    for record in records:
        groups[record.operator.id, record.provenance].append(record)
    return [rows for key, rows in sorted(groups.items())]


def _compare_grouped(records, budgets, seeds, split="test", candidate_pool_size=512):
    from .replay import compare_searchers
    results = []
    for rows in _operator_groups(records):
        results.extend(compare_searchers(rows, _infer_space(rows), budgets=budgets, seeds=seeds, split=split,
                                         candidate_pool_size=candidate_pool_size))
    return results


def _l3_evaluate(records: Iterable[MeasurementRecord], space: ConfigSpace | None = None) -> tuple[list[dict], list[dict]]:
    from .l3 import PerformanceMatrix, ShapeDispatcher, bucket_baseline, greedy, greedy_local_swap, oracle
    groups: dict[tuple[str, str, str], list[MeasurementRecord]] = defaultdict(list)
    for record in records:
        groups[record.hardware.id, record.operator.id, record.provenance].append(record)
    results, missing = [], []
    methods = {"oracle": oracle, "greedy": greedy, "greedy_local_swap": greedy_local_swap, "bucket": bucket_baseline}
    for (hardware_id, operator_id, provenance), rows in sorted(groups.items()):
        group_space = _infer_space(rows)
        shapes = tuple(sorted({r.shape.id: r.shape for r in rows}.values(), key=lambda s: tuple(sorted(s.dims.items()))))
        configs = tuple(sorted({r.config.id: r.config for r in rows}.values(), key=lambda c: c.id))
        workload = WorkloadDistribution(shapes, tuple(1.0 for _ in shapes), name=f"{rows[0].hardware.name}-observed-workload")
        matrix = PerformanceMatrix.from_records(workload, configs, rows)
        if matrix.reference_latency_ms is None:
            missing.append({"hardware_id": hardware_id, "operator_id": operator_id,
                            "reason": "at least one workload shape has no verified successful measurement", "provenance": provenance})
            continue
        for k in (1, 2, 4, 8, 16):
            for name, solver in methods.items():
                try:
                    if name == "bucket":
                        common_dims = set.intersection(*(set(s.dims) for s in shapes))
                        dimension = "M" if "M" in common_dims else sorted(common_dims)[0]
                        result = solver(matrix, k, bucket_dimension=dimension)
                    else:
                        result = solver(matrix, k)
                except ValueError as exc:
                    missing.append({"hardware_id": hardware_id, "method": name, "k": k, "reason": str(exc)})
                    continue
                payload = result.to_dict()
                payload.update({"hardware_id": hardware_id, "hardware_name": rows[0].hardware.name,
                                "operator_id": operator_id, "matrix_coverage": matrix.coverage,
                                "operator_name": rows[0].operator.name, "operator_kind": rows[0].operator.kind,
                                "context": workload.name})
                if name == "greedy_local_swap" and k == 4:
                    try:
                        dispatcher = ShapeDispatcher.from_result(matrix, result, hardware=rows[0].hardware,
                                                                 operator=rows[0].operator, space=group_space)
                        payload["dispatcher"] = dispatcher.to_dict()
                        for shape, column in zip(shapes, result.assignment_indices):
                            if column is not None and dispatcher.select_config(shape).id != configs[column].id:
                                raise AssertionError("dispatcher does not reproduce selected assignment")
                    except ValueError as exc:
                        missing.append({"hardware_id": hardware_id, "operator_id": operator_id, "method": "dispatcher", "k": k, "reason": str(exc)})
                results.append(payload)
    return results, missing


def _policy_evaluate(records: list[MeasurementRecord], space: ConfigSpace, output: Path, seed: int) -> list[dict]:
    import hashlib
    import inspect
    from .models import stable_id
    from .policies import HeuristicPolicy, PolicyRepository, SearchPolicy, evaluate_policy
    training = tuple(r for r in records if r.split == "train")
    grouped: dict[str, list[MeasurementRecord]] = defaultdict(list)
    for record in records:
        if record.split == "test":
            grouped[record.case_id].append(record)
    repository = PolicyRepository(output / "policies" / records[0].operator.name)
    results = []
    for mode in ("select", "search"):
        policy = (HeuristicPolicy(space, seed=seed) if mode == "select" else
                  SearchPolicy(space, algorithm="surrogate", seed=seed, training_records=training))
        source_path = inspect.getsourcefile(type(policy))
        source_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
        version = "0.1.0-" + stable_id([policy.specification, source_hash])[:12]
        repository.register(policy, version=version, metadata={"frozen_before_test": True, "source": "non_llm_baseline"})
        for case in sorted(grouped):
            result = evaluate_policy(policy, grouped[case], space,
                                     mode=mode, budget=SearchBudget(16), seed=seed)
            repository.log_evaluation(result)
            results.append(result.to_dict())
    return results


def command_demo(args: argparse.Namespace) -> dict[str, Any]:
    from .analysis import generate_report
    from .datasets import split_records
    from .replay import summarize_results
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    collection_id = f"mock-demo-v2-seed-{args.seed}"
    hardware_values = demo_hardware()
    if args.hardware:
        from .hardware import get_hardware_profile
        hardware_values = tuple(get_hardware_profile(name) for name in args.hardware)
    with ExperimentStore(output / "collection.sqlite") as collection:
        runner = BenchmarkRunner(MockAdapter(), collection, seed=args.seed, experiment_id=collection_id,
                                 failure_log_path=output / "failures.jsonl")
        raw_records = []
        for name in args.operators:
            shapes = operator_demo_shapes(name)
            space = operator_space(name, small=True, shapes=shapes)
            operator = operator_spec(name)
            for hardware in hardware_values:
                raw_records.extend(runner.run(hardware, operator, shapes, space))
    partitions = split_records(raw_records, seed=args.seed, holdout=args.holdout)
    dataset_id = f"{collection_id}-{args.holdout}-split"
    records = [replace(record, experiment_id=dataset_id) for split in (partitions.train, partitions.validation, partitions.test) for record in split]
    with ExperimentStore(output / "measurements.sqlite") as store:
        store.add_many(records)
        store.export_jsonl(output / "measurements.jsonl")
        store_summary = store.summary()
    for split in ("train", "validation", "test"):
        _save_jsonl(output / "splits" / f"{split}.jsonl", (r for r in records if r.split == split))
    manifest = partitions.manifest()
    manifest.update({"provenance": "mock", "experiment_id": dataset_id,
                     "scope": "synthetic infrastructure validation; no real GPU claims",
                     "operators": list(args.operators),
                     "candidate_domains": {name: operator_space(name, small=True, shapes=operator_demo_shapes(name)).domains for name in args.operators}})
    _save_json(output / "split_manifest.json", manifest)
    print(f"Mock collection complete: {len(records)} records; replaying six search methods...", file=sys.stderr, flush=True)
    replay_results = _compare_grouped(records, args.budgets, args.seeds)
    _save_json(output / "replay.json", [result.to_dict() for result in replay_results])
    _save_json(output / "replay_summary.json", summarize_results(replay_results))
    print("Replay complete; scoring frozen policies and solving L3 variant sets...", file=sys.stderr, flush=True)
    policy_results = []
    for rows in _operator_groups(records):
        policy_results.extend(_policy_evaluate(rows, _infer_space(rows), output, args.seed))
    _save_json(output / "policy_evaluations.json", policy_results)
    l3_results, unavailable = _l3_evaluate(records)
    _save_json(output / "l3.json", l3_results)
    _save_json(output / "l3_unavailable.json", unavailable)
    report = generate_report(records, output / "report", replay_results=replay_results, l3_results=l3_results)
    # Include policy scores separately; policy generation never receives this report.
    with report.open("a", encoding="utf-8") as stream:
        stream.write("\n## Policy evaluation\n\nFrozen non-LLM policies were independently scored on held-out cases. "
                     "See [policy evaluations](../policy_evaluations.json) and [version registry](../policies/).\n")
    result = {"provenance": "mock", "records": len(records), "search_trials": len(replay_results),
              "policy_evaluations": len(policy_results), "l3_results": len(l3_results), "l3_unavailable": unavailable,
              "split": args.holdout, "report": str(report.resolve()), "database": str((output / "measurements.sqlite").resolve()),
              "statuses": dict(Counter(r.status.value for r in records)), "wall_time_s": time.perf_counter() - started,
              "store": store_summary}
    _save_json(output / "run_summary.json", result)
    return result


def command_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from .analysis import generate_report
    spec = ExperimentSpec.load(args.spec) if args.spec else example_spec(args.backend, args.operator, args.dtype)
    adapter = get_adapter(spec.backend)
    hardware = spec.hardware
    profile = None
    if args.hardware:
        from .hardware import get_hardware_profile
        profile = get_hardware_profile(args.hardware)
        hardware = profile
    if hardware is None:
        if not hasattr(adapter, "probe_hardware"):
            raise ValueError("experiment must specify hardware for this backend")
        hardware = adapter.probe_hardware()
    if hasattr(adapter, "probe_hardware"):
        actual = adapter.probe_hardware()
        if profile:
            from .hardware import validate_hardware_profile
            check = validate_hardware_profile(profile, actual)
            if not check.valid:
                raise ValueError("hardware profile mismatch: " + "; ".join(check.reasons))
            from dataclasses import asdict
            # Preserve physical identity/resources; attach theoretical values as sourced planning data.
            actual = replace(actual, compute_tflops=profile.compute_tflops,
                             bandwidth_gbps=profile.bandwidth_gbps,
                             metadata={**actual.metadata, "requested_profile": asdict(profile),
                                       "theoretical_values_source": "requested vendor specification profile; not measured"})
        hardware = actual
    budget = SearchBudget(args.max_evaluations if args.max_evaluations is not None else spec.budget.max_evaluations,
                          args.max_seconds if args.max_seconds is not None else spec.budget.max_wall_time_s)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    spec = replace(spec, hardware=hardware, budget=budget)
    spec.save(output / "experiment.json")
    with ExperimentStore(output / "measurements.sqlite") as store:
        runner = BenchmarkRunner(adapter, store, warmup=spec.warmup, repetitions=spec.repetitions, seed=spec.seed,
                                 experiment_id=spec.name, failure_log_path=output / "failures.jsonl")
        if args.searcher:
            from .search import make_searcher
            from .tuning import tune
            training = ()
            if args.training_database:
                with ExperimentStore(args.training_database, read_only=True) as source:
                    training = tuple(source.records(split="train"))
            tuning_results, records = [], []
            # Per-shape tuning budget is explicit; collection mode budget is global.
            for shape in spec.shapes:
                result = tune(runner, hardware, spec.operator, shape, spec.config_space(),
                              make_searcher(args.searcher, seed=spec.seed, training_records=training,
                                            candidate_pool_size=args.candidate_pool_size), budget, split=spec.split, resume=not args.no_resume)
                tuning_results.append(result.to_dict())
                records.extend(result.records)
            _save_json(output / "tuning.json", tuning_results)
            runner.last_run_stats = {"budget_scope": "per_shape", "evaluations": sum(r["evaluations"] for r in tuning_results),
                                     "cache_hits": sum(r["cache_hits"] for r in tuning_results)}
        else:
            records = runner.run(hardware, spec.operator, spec.shapes, spec.config_space(), budget=budget,
                                 split=spec.split, resume=not args.no_resume)
        store.export_jsonl(output / "measurements.jsonl")
        summary = store.summary()
    _save_jsonl(output / "run_records.jsonl", records)
    report = generate_report(records, output / "report")
    result = {"backend": spec.backend, "hardware": hardware.name, "provenance": adapter.provenance,
              "returned_records": len(records), "stats": runner.last_run_stats, "store": summary, "report": str(report.resolve()),
              "requested_shapes": len(spec.shapes), "successful_shapes": len({r.shape.id for r in records if r.successful}),
              "requested_shape_ids": sorted({shape.id for shape in spec.shapes}),
              "statuses": dict(Counter(r.status.value for r in records)),
              "all_successful": bool(records) and all(r.successful for r in records)}
    _save_json(output / "run_summary.json", result)
    if args.require_success and {shape.id for shape in spec.shapes} - {r.shape.id for r in records if r.successful}:
        raise ValueError(f"smoke validation failed: some shapes have no successful measurement; inspect {output / 'run_summary.json'} and the journal")
    if args.require_all_success and (not result["all_successful"] or result["successful_shapes"] != len(spec.shapes)):
        raise ValueError(f"strict GPU validation failed: every tested configuration and shape must succeed; inspect {output / 'run_summary.json'} and the journal")
    return result


def command_replay(args: argparse.Namespace) -> dict[str, Any]:
    from .analysis import generate_report
    from .replay import summarize_results
    with ExperimentStore(args.database, read_only=True) as store:
        records = store.records()
    results = _compare_grouped(records, args.budgets, args.seeds, args.split, args.candidate_pool_size)
    output = Path(args.output)
    _save_json(output / "replay.json", [r.to_dict() for r in results])
    _save_json(output / "replay_summary.json", summarize_results(results))
    report = generate_report(records, output / "report", replay_results=results)
    return {"trials": len(results), "split": args.split, "report": str(report.resolve())}


def command_l3(args: argparse.Namespace) -> dict[str, Any]:
    with ExperimentStore(args.database, read_only=True) as store:
        records = store.records()
    results, unavailable = _l3_evaluate(records)
    output = Path(args.output)
    _save_json(output / "l3.json", results)
    _save_json(output / "l3_unavailable.json", unavailable)
    return {"results": len(results), "unavailable": unavailable, "output": str(output.resolve())}


def command_report(args: argparse.Namespace) -> dict[str, Any]:
    from .analysis import generate_report
    with ExperimentStore(args.database, read_only=True) as store:
        records = store.records()
    replay_results = json.loads(Path(args.replay).read_text()) if args.replay else ()
    l3_results = json.loads(Path(args.l3).read_text()) if args.l3 else ()
    path = generate_report(records, args.output, replay_results=replay_results, l3_results=l3_results)
    return {"records": len(records), "report": str(path.resolve())}


def command_exchange(args: argparse.Namespace) -> dict[str, Any]:
    with ExperimentStore(args.database, read_only=args.command == "export") as store:
        if args.command == "export":
            return {"exported": store.export_jsonl(args.output), "output": args.output}
        if args.command == "import-jsonl":
            return store.import_jsonl(args.input)
        from .adapters import load_kernel_tuner_json
        rows = load_kernel_tuner_json(args.input, correctness=args.correctness,
                                     correctness_evidence=args.correctness_evidence, source_url=args.source_url)
        added = store.add_many(rows)
        return {"read": len(rows), "added": added, "duplicates": len(rows) - added, "provenance": "imported",
                "statuses": dict(Counter(r.status.value for r in rows)), "database": args.database}


def command_hub(args: argparse.Namespace) -> dict[str, Any]:
    from .adapters.benchmark_hub import HUB_COMMIT, discover_benchmark_hub, download_benchmark_hub, import_benchmark_hub
    entries = discover_benchmark_hub(cache_dir=args.cache_dir, include_t4=args.include_t4 or args.format == "t4", refresh=args.refresh)
    selected = [entry for entry in entries if (not args.kernel or entry.kernel.casefold() == args.kernel.casefold())
                and (not args.gpu or entry.gpu.casefold() == args.gpu.casefold())
                and (not args.format or entry.format == args.format)]
    if args.hub_action == "list":
        return {"source_commit": HUB_COMMIT, "available_entries": len(entries),
                "kernels": sorted({e.kernel for e in entries}), "gpu_labels": sorted({e.gpu for e in entries}),
                "compressed_bytes": sum(e.size_bytes for e in selected), "entries": [e.to_dict() for e in selected],
                "scope": "pinned active cachefiles; archive/hyperparameter collections excluded"}
    if not args.kernel or not args.gpu or len(selected) != 1:
        raise ValueError("select exactly one Hub entry with --kernel, --gpu, and optionally --format; use 'ktlab hub list' to inspect names")
    entry = selected[0]
    if args.hub_action == "download":
        path = download_benchmark_hub(entry, args.cache_dir)
        return {"entry": entry.to_dict(), "path": str(path.resolve()), "checksum_verified": True}
    records = import_benchmark_hub(entry, args.cache_dir, split=args.split,
                                  experiment_id=args.experiment_id or f"hub-{HUB_COMMIT[:12]}-{entry.kernel}-{entry.gpu}")
    with ExperimentStore(args.database) as store:
        added = store.add_many(records)
        if args.export_jsonl:
            store.export_jsonl(args.export_jsonl)
    result = {"entry": entry.to_dict(), "read": len(records), "added": added,
              "duplicates": len(records) - added, "database": str(Path(args.database).resolve()),
              "statuses": dict(Counter(r.status.value for r in records)),
              "provenance": "imported", "verified_successes": sum(r.successful for r in records),
              "notice": "upstream timing data; correctness unknown is excluded from validated performance oracles"}
    _save_json(Path(args.cache_dir) / f"{entry.kernel}_{entry.gpu}_journal_import.json", result)
    return result


def command_scale(args: argparse.Namespace) -> dict[str, Any]:
    from .scalability import run_scale_demo
    return run_scale_demo(args.output, budget=args.max_evaluations, candidate_pool_size=args.candidate_pool_size,
                          seed=args.seed, matrix_shapes=args.matrix_shapes, matrix_configs=args.matrix_configs)


def command_probe(args: argparse.Namespace) -> dict[str, Any]:
    from dataclasses import asdict
    result = {"python": platform.python_version(),
              "optional_modules": {name: importlib.util.find_spec(name) is not None
                                   for name in ("torch", "triton", "matplotlib", "pytest")}}
    if not args.gpu:
        if args.hardware:
            raise ValueError("--hardware device verification requires probe --gpu")
        return result
    adapter = get_adapter("triton")
    actual = adapter.probe_hardware()
    if args.hardware:
        from .hardware import get_hardware_profile, validate_hardware_profile
        profile = get_hardware_profile(args.hardware)
        validation = validate_hardware_profile(profile, actual)
        if not validation.valid:
            raise ValueError("hardware profile mismatch: " + "; ".join(validation.reasons))
        result["requested_profile"] = profile.metadata["profile_name"]
    result.update({"hardware": asdict(actual), "environment": adapter.environment(),
                   "runtime_requirements_met": True,
                   "notice": "Runtime and device preflight only; numerical validation is performed by the GPU smoke suite."})
    return result


def command_asset_experiment(args: argparse.Namespace) -> dict[str, Any]:
    from .asset_runner import execute_asset_experiment
    return execute_asset_experiment(args.spec, args.output, check_only=args.command == "check", plugins=tuple(args.plugin))


def command_assets(args: argparse.Namespace) -> dict[str, Any]:
    from .catalog import get_catalog
    from .plugins import discover_entry_points, load_plugin
    catalog = get_catalog()
    entries = discover_entry_points(catalog)
    for plugin in args.asset_plugins:
        load_plugin(plugin, catalog)
    if args.assets_action == "show":
        if not args.kind or not args.ref:
            raise ValueError("assets show requires --kind and --ref")
        asset = catalog.resolve(args.kind, args.ref)
        return {"ref": asset.ref, "fingerprint": asset.fingerprint, **asset.manifest()}
    return {"assets": [{"ref": asset.ref, "fingerprint": asset.fingerprint, **asset.manifest()}
                       for asset in catalog.list(args.kind)], "installed_entry_points": list(entries)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ktlab", description="Kernel Tuning Research Lab v0.1")
    parser.add_argument("--plugin", action="append", default=[], metavar="MODULE_OR_PATH", help="explicitly load a trusted Python extension; repeatable")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, description in (("run", "accept and search registered Kernel/algorithm assets"),
                              ("check", "automatically accept registered assets without research search")):
        asset_run = sub.add_parser(name, help=description)
        asset_run.add_argument("--spec", required=True, help="schema_version=2 asset experiment JSON")
        asset_run.add_argument("--output", default="artifacts/assets")
        asset_run.set_defaults(handler=command_asset_experiment)
    assets = sub.add_parser("assets", help="discover registered versioned operators, Kernels and algorithms")
    assets.add_argument("assets_action", choices=("list", "show"))
    assets.add_argument("--kind", choices=("operator", "kernel", "algorithm"))
    assets.add_argument("--ref")
    assets.add_argument("--plugin", action="append", default=[], dest="asset_plugins")
    assets.set_defaults(handler=command_assets)
    scaffold = sub.add_parser("scaffold", help="generate a working SDK-only asset and experiment template")
    scaffold.add_argument("asset_kind", choices=("kernel", "algorithm"))
    scaffold.add_argument("--operator", default="gemm@1.0.0")
    scaffold.add_argument("--runtime", choices=("python", "cuda"), default="python")
    scaffold.add_argument("--output", required=True)
    def create_template(args):
        from .scaffolding import scaffold
        return scaffold(args.asset_kind, args.output, operator=args.operator, runtime=args.runtime)
    scaffold.set_defaults(handler=create_template)
    proposal = sub.add_parser("proposal-suite", help="generate or run all proposal operator assets and production workloads")
    proposal.add_argument("--runtime", choices=("python", "cuda"), default="python")
    proposal.add_argument("--scale", choices=("smoke", "production"), default="smoke")
    proposal.add_argument("--hardware", help="optional expected GPU profile, e.g. 5060ti16g or H200; otherwise probe the actual device")
    proposal.add_argument("--operators", nargs="+", help="optional versioned operator references; default discovers all proposal families")
    proposal.add_argument("--algorithms", nargs="+", default=["random@1.0.0", "grid@1.0.0"])
    proposal.add_argument("--dtypes", nargs="+", help="defaults: float32 for CPU; float16 and bfloat16 for CUDA")
    proposal.add_argument("--max-evaluations", type=int, default=4)
    proposal.add_argument("--max-seconds", type=float, default=120)
    proposal.add_argument("--validation-evaluations", type=int)
    proposal.add_argument("--workspace-limit-mib", type=int, default=256)
    proposal.add_argument("--variants", action="store_true", help="also execute declared semantic parameter cases")
    proposal.add_argument("--write-only", action="store_true", help="generate editable specs without any Kernel execution")
    proposal.add_argument("--check-only", action="store_true", help="run static, synthetic and numerical acceptance without research search")
    proposal.add_argument("--allow-config-failures", action="store_true", help="research mode: retain config failures without making the whole suite fail")
    proposal.add_argument("--output", default="artifacts/proposal-suite")
    def execute_proposal_suite(args):
        from .proposal_suite import cli
        return cli(args)
    proposal.set_defaults(handler=execute_proposal_suite)
    demo = sub.add_parser("demo", help="run complete CPU Mock research workflow")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--operators", choices=OPERATOR_NAMES, nargs="+", default=list(OPERATOR_NAMES))
    demo.add_argument("--hardware", nargs="+", help="optional hardware profiles, e.g. H200 H800 4090 5060ti16g; samples remain mock")
    demo.add_argument("--holdout", choices=("case", "shape", "hardware"), default="hardware")
    demo.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    demo.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    demo.set_defaults(handler=command_demo)
    benchmark = sub.add_parser("benchmark", help="collect real or synthetic kernel measurements")
    benchmark.add_argument("--backend", choices=("mock", "triton"), default="mock")
    benchmark.add_argument("--spec")
    benchmark.add_argument("--operator", choices=OPERATOR_NAMES, default="gemm")
    benchmark.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    benchmark.add_argument("--hardware", help="expected hardware profile (H200/H800/4090/5060ti16g); real device is probed")
    benchmark.add_argument("--searcher", choices=("random", "grid", "heuristic", "local", "surrogate", "transfer"),
                           help="adaptive online tuning, with the evaluation budget applied per shape")
    benchmark.add_argument("--output", default="artifacts/benchmark")
    benchmark.add_argument("--max-evaluations", type=int)
    benchmark.add_argument("--max-seconds", type=float)
    benchmark.add_argument("--no-resume", action="store_true")
    benchmark.add_argument("--require-success", action="store_true", help="fail after saving artifacts if any requested shape has no successful config")
    benchmark.add_argument("--require-all-success", action="store_true", help="strict validation: fail if any tested config fails or any shape lacks success")
    benchmark.add_argument("--candidate-pool-size", type=int, default=512, help="bounded candidate pool for spaces with more than 4096 combinations")
    benchmark.add_argument("--training-database", help="optional train-only journal for transfer search")
    benchmark.set_defaults(handler=command_benchmark)
    replay = sub.add_parser("replay", help="compare algorithms against private offline observations")
    replay.add_argument("--database", required=True)
    replay.add_argument("--output", default="artifacts/replay")
    replay.add_argument("--split", choices=("train", "validation", "test"), default="test")
    replay.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    replay.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    replay.add_argument("--candidate-pool-size", type=int, default=512)
    replay.set_defaults(handler=command_replay)
    l3 = sub.add_parser("l3", help="solve finite-variant selection for observed workloads")
    l3.add_argument("--database", required=True)
    l3.add_argument("--output", default="artifacts/l3")
    l3.set_defaults(handler=command_l3)
    report = sub.add_parser("report", help="generate Markdown and SVG figures from a result journal")
    report.add_argument("--database", required=True)
    report.add_argument("--output", default="artifacts/report")
    report.add_argument("--replay", help="optional replay.json")
    report.add_argument("--l3", help="optional l3.json")
    report.set_defaults(handler=command_report)
    for command in ("export", "import-jsonl", "import-public"):
        child = sub.add_parser(command)
        child.add_argument("--database", required=True)
        if command == "export":
            child.add_argument("--output", required=True)
        else:
            child.add_argument("--input", required=True)
        if command == "import-public":
            child.add_argument("--correctness", choices=("unknown", "verified"), default="unknown")
            child.add_argument("--correctness-evidence")
            child.add_argument("--source-url")
        child.set_defaults(handler=command_exchange)
    spec = sub.add_parser("write-spec", help="write a portable experiment JSON")
    spec.add_argument("--backend", choices=("mock", "triton"), default="mock")
    spec.add_argument("--operator", choices=OPERATOR_NAMES, default="gemm")
    spec.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    spec.add_argument("--output", required=True)
    def write_spec(args):
        example_spec(args.backend, args.operator, args.dtype).save(args.output)
        return {"spec": args.output, "backend": args.backend}
    spec.set_defaults(handler=write_spec)
    hub = sub.add_parser("hub", help="discover, verify, download and import complete AutoTuning Benchmark Hub caches")
    hub.add_argument("hub_action", choices=("list", "download", "import"))
    hub.add_argument("--cache-dir", default="artifacts/public-hub")
    hub.add_argument("--kernel")
    hub.add_argument("--gpu")
    hub.add_argument("--include-t4", action="store_true", help="also list/download T4 exports; T4 import is not implemented")
    hub.add_argument("--format", choices=("kernel_tuner_cache", "t4"), help="select one source format when original and T4 exports coexist")
    hub.add_argument("--refresh", action="store_true", help="refresh the pinned catalogue from the primary GitHub source")
    hub.add_argument("--database", default="artifacts/public-hub/measurements.sqlite")
    hub.add_argument("--export-jsonl")
    hub.add_argument("--split", choices=("train", "validation", "test"), default="train")
    hub.add_argument("--experiment-id")
    hub.set_defaults(handler=command_hub)
    scale = sub.add_parser("scale-demo", help="bounded search in 139968 configurations and medium-scale mock L3")
    scale.add_argument("--output", default="artifacts/scale")
    scale.add_argument("--max-evaluations", type=int, default=32)
    scale.add_argument("--candidate-pool-size", type=int, default=256)
    scale.add_argument("--seed", type=int, default=19)
    scale.add_argument("--matrix-shapes", type=int, default=128)
    scale.add_argument("--matrix-configs", type=int, default=64)
    scale.set_defaults(handler=command_scale)
    hardware = sub.add_parser("hardware", help="list sourced hardware specification profiles")
    def list_hardware(args):
        from dataclasses import asdict
        from .hardware import get_hardware_profile, list_hardware_profiles
        return {"profiles": {name: asdict(get_hardware_profile(name)) for name in list_hardware_profiles()},
                "note": "specification profiles are not device measurements; unknown fields remain explicit"}
    hardware.set_defaults(handler=list_hardware)
    probe = sub.add_parser("probe", help="inspect optional runtimes without importing CUDA")
    probe.add_argument("--gpu", action="store_true", help="explicitly probe CUDA device and check native runtime compatibility")
    probe.add_argument("--hardware", help="expected profile for --gpu, e.g. 5060ti16g")
    probe.set_defaults(handler=command_probe)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from .plugins import load_plugin
        if args.command not in {"run", "check"}:
            for plugin in args.plugin:
                load_plugin(plugin)
        result = args.handler(args)
    except (BackendUnavailableError, ValueError, OSError, ImportError, TypeError, KeyError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result.get("exit_code", 0) if args.command in {"run", "check", "proposal-suite"} else 0
