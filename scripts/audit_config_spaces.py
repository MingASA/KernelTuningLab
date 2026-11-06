"""Bounded CPU-only declaration/static-space audit; never claims GPU feasibility.

Each case uses a fresh process, so peak RSS is attributable to that case. Use
--source-root with an immutable checkout to audit an older public SDK as well.
Opaque Python constraints are checked only on complete assignments.
"""
from collections import Counter
import argparse
import itertools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time


CASES = {
    "fused": ("triton/w4a16_gemm", {"M": 4, "N": 512, "K": 1024}),
    "pipeline": ("triton_pipeline/w4a16_gemm", {"M": 4, "N": 512, "K": 1024}),
    "moe": ("triton/moe_grouped_gemm", {"M": 64, "N": 128, "K": 256, "Groups": 8}),
}


def normalized_count(space, specs, max_parent_states=4096):
    """Exact conditional count via bounded parent enumeration, no hard filters.

    This counts declared inactive defaults once, including nested conditions.
    It does not infer the feasibility of arbitrary constraint callbacks.
    """
    parents = {p for spec in specs.values() for p in spec.active_when}
    if math.prod(len(space.domains[p]) for p in parents) > max_parent_states:
        return {"status": "unknown", "reason": "activation-parent enumeration budget"}
    parents = tuple(sorted(parents)); seen = set(); count = 0
    for values in itertools.product(*(space.domains[p] for p in parents)):
        assignment = dict(zip(parents, values)); active = {}
        def enabled(name):
            if name not in active:
                active[name] = all(enabled(p) and assignment[p] in (v if isinstance(v, (list, tuple)) else (v,))
                                   for p, v in specs[name].active_when.items())
            return active[name]
        canonical = tuple((p, assignment[p] if enabled(p) else specs[p].default) for p in parents)
        if canonical in seen:
            continue
        seen.add(canonical)
        count += math.prod(len(space.domains[k]) if enabled(k) else 1 for k in specs if k not in parents)
    return {"status": "exact", "count": count, "parent_states": len(seen),
            "notice": "Conditional normalization only; static/GPU constraints not included."}


def worker(case, hardware, seconds):
    from kernel_tuning_lab.asset_space import build_effective_space
    from kernel_tuning_lab.catalog import get_catalog
    from kernel_tuning_lab.hardware import get_hardware_profile
    from kernel_tuning_lab.models import OperatorSpec, ShapeSpec
    catalog = get_catalog(); ref, dims = CASES[case]
    kernel = catalog.resolve("kernel", ref); definition = catalog.resolve("operator", kernel.operator)
    hw = get_hardware_profile(hardware)
    op = OperatorSpec(definition.id, kind=definition.id, implementation=kernel.ref, dtype="float16",
                      version=definition.version, parameters=dict(definition.parameters))
    shape = ShapeSpec(dims); before = time.perf_counter()
    space = build_effective_space(kernel, hw, op, shape)
    result = {"case": case, "kernel": kernel.ref, "hardware_profile": hardware, "shape": dims,
              "hardware_is_static_planning_profile": True, "GPU_feasibility": "not_measured_by_this_audit",
              "space_resolution_wall_s": time.perf_counter()-before,
              "declared_cartesian_count": math.prod(len(s.values) for s in kernel.space.parameters.values()),
              "resolved_context_cartesian_count": space.cardinality,
              "normalized": normalized_count(space, kernel.space.parameters)}
    before = time.perf_counter()
    if hasattr(space, "build_feasible"):
        from kernel_tuning_lab.space_tools import SpaceBuildBudget
        built = space.build_feasible(hw, op, shape, budget=SpaceBuildBudget(
            max_nodes=200_000, max_configs=50_000, max_wall_time_s=seconds))
        result["static_build"] = {"status": built.status, "complete": built.complete,
            "count": len(built.configs), "diagnostics": dict(built.diagnostics)}
        before_sample = time.perf_counter()
        sampled = space.sample_feasible(random.Random(7), 32, hw, op, shape,
            budget=SpaceBuildBudget(max_nodes=30_000, max_wall_time_s=5))
        result["sample"] = {"requested": 32, "returned": len(sampled.configs), "status": sampled.status,
            "wall_s": time.perf_counter()-before_sample, "diagnostics": dict(sampled.diagnostics),
            "configs": [dict(c.params) for c in sampled.configs]}
    else:
        checked = valid = 0; reasons = Counter(); complete = True
        for c in space.iter_configs():
            if time.perf_counter()-before >= seconds:
                complete = False; break
            v = space.validate(c, hw, op, shape); checked += 1; valid += v.valid
            reasons.update(v.reasons)
        result["static_build"] = {"status": "complete" if complete else "budget_exhausted",
            "complete": complete, "count": valid, "checked": checked, "rejections": dict(reasons)}
        before_sample = time.perf_counter(); candidates = space.sample_configs(random.Random(7), 512)
        legal = [c for c in candidates if space.validate(c, hw, op, shape).valid]
        result["sample"] = {"method": "legacy initial random pool", "requested": 512,
            "returned": len(legal), "proposed": len(candidates), "wall_s": time.perf_counter()-before_sample,
            "declared_fraction": len(candidates)/space.cardinality}
    result["static_and_sample_wall_s"] = time.perf_counter()-before
    result["peak_process_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result["RSS_notice"] = "Fresh Linux process high-water RSS, includes imports; max_bytes in builder is estimated retained payload, not an OS RSS cap."
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path); p.add_argument("--source-root", type=Path, default=Path.cwd())
    p.add_argument("--seconds", type=float, default=15); p.add_argument("--hardware", default="4090")
    p.add_argument("--worker", choices=CASES)
    args = p.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        p.error("--seconds must be within (0,60]")
    if args.worker:
        print(json.dumps(worker(args.worker, args.hardware, args.seconds), allow_nan=False)); return
    if args.output is None:
        p.error("--output is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.source_root,
                            capture_output=True, text=True, timeout=5).stdout.strip()
    results = {"source_commit": commit, "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "seconds_per_static_build": args.seconds, "cases": []}
    for case in CASES:
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", case,
            "--hardware", args.hardware, "--seconds", str(args.seconds)], cwd=args.source_root,
            env={**os.environ, "PYTHONPATH": str(args.source_root.resolve()), "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True, text=True, timeout=args.seconds+30)
        if r.returncode:
            raise RuntimeError(f"{case} CPU audit failed: {r.stderr}")
        results["cases"].append(json.loads(r.stdout))
        args.output.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps({"source_commit": commit, "cases": [(c["case"], c["static_build"]["status"],
        c["static_build"]["count"]) for c in results["cases"]]}))


if __name__ == "__main__":
    main()
