"""Small, serial physical experiments using the existing v2 runner and journal.

Generate with --write-only; run one family at a time on an otherwise idle GPU.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess

from kernel_tuning_lab.asset_experiments import (AlgorithmSelection, AssetExperiment,
                                                KernelSelection, ValidationBudget)
from kernel_tuning_lab.asset_runner import execute_asset_experiment
from kernel_tuning_lab.models import SearchBudget, ShapeSpec


def base(name, operator, kernels, shapes, *, parameters=None, algorithms=("grid",), budget=8, seed=7):
    return AssetExperiment(name=name, operator=operator, kernels=tuple(kernels),
        algorithms=tuple(AlgorithmSelection(a) for a in algorithms), shapes=tuple(ShapeSpec(s) for s in shapes),
        dtype="float16", hardware="5060ti16g", parameters=parameters or {},
        budget=SearchBudget(budget, 180), validation=ValidationBudget(12, 2, 180, 17, False),
        preflight_seconds=60, seed=seed, warmup=10, repetitions=31, split="validation",
        timing_mode="cuda_graph", cuda_graph_batch_size=512)


def specs(family):
    gemm = {"BLOCK_M": (16, 32, 64), "BLOCK_N": (32, 64), "BLOCK_K": (32,),
            "num_warps": (4,), "num_stages": (2, 3)}
    if family == "l2":
        space = {**gemm, "BLOCK_N": (32, 64, 128), "BLOCK_K": (32, 64), "num_warps": (4, 8)}
        for seed, budget in ((7, 4), (7, 8), (7, 16), (19, 16), (31, 16)):
            yield base(f"l2-seed{seed}-budget{budget}", "gemm", (KernelSelection("triton/gemm", space),),
                       ({"M": 512, "N": 1024, "K": 512},), algorithms=("random", "heuristic", "surrogate"),
                       seed=seed, budget=budget)
    elif family == "l3":
        for seed in (7, 19, 31):
            yield base(f"l3-dense-seed{seed}", "gemm", (KernelSelection("triton/gemm", gemm),),
                ({"M": 1, "N": 2048, "K": 1024}, {"M": 512, "N": 512, "K": 512},
                 {"M": 257, "N": 769, "K": 513}), seed=seed, budget=12)
    elif family == "representative":
        quant = {"BLOCK_M": (16, 32), "BLOCK_N": (64,), "BLOCK_K": (32,), "num_warps": (4,),
                 "num_stages": (2,), "GROUP_M": (1,), "SPLIT_K": (1, 2)}
        yield base("w4-fused-pipeline", "w4a16_gemm", (
            KernelSelection("triton/w4a16_gemm", quant),
            KernelSelection("triton_pipeline/w4a16_gemm", {**quant, "DEQUANT_BLOCK_SIZE": (512,)})),
            ({"M": 1, "N": 2048, "K": 2048}, {"M": 128, "N": 1024, "K": 2048}), budget=4)
        attention = {"BLOCK_M": (16, 32), "BLOCK_N": (32, 64), "num_warps": (4,), "num_stages": (2,),
                     "USE_EXP2": (True,), "PRE_LOAD_V": (False,)}
        for causal in (True, False):
            yield base(f"attention-causal-{causal}", "attention", (KernelSelection("triton/attention", attention),),
                tuple({"Batch": 1, "Heads": 4, "KVHeads": 2, "SeqQ": q, "SeqKV": kv, "D": 64}
                      for q, kv in ((128, 128), (1024, 1024), (1, 4096))),
                parameters={"causal": causal}, budget=4)
        moe = {"BLOCK_M": (16, 32), "BLOCK_N": (64,), "BLOCK_K": (32,), "num_warps": (4,),
               "num_stages": (2,), "GROUP_M": (1,), "SPLIT_K": (1, 2),
               "SCHEDULE": ("grouped", "persistent"), "PERSISTENT_CTA_MULT": (1,),
               "REDUCE_BLOCK_SIZE": (256,), "USE_TMA": (False, True),
               "MAP_BLOCK_SIZE": (32,), "MAP_NUM_WARPS": (1,)}
        for distribution in ("balanced", "skewed"):
            yield base(f"moe-{distribution}", "moe_grouped_gemm", (KernelSelection("triton/moe_grouped_gemm", moe),),
                ({"M": 128, "N": 256, "K": 256, "Groups": 4},
                 {"M": 256, "N": 512, "K": 512, "Groups": 8}),
                parameters={"distribution": distribution}, budget=16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("l2", "l3", "representative"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--write-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "family": args.family, "serial": True, "cases": []}
    for spec in specs(args.family):
        path = output / "specs" / (spec.name + ".json")
        spec.save(path)
        def telemetry():
            return subprocess.run(["nvidia-smi", "--query-gpu=timestamp,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
                "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
        before = None if args.write_only else telemetry()
        result = ({"status": "generated"} if args.write_only else
                  execute_asset_experiment(path, output / "runs" / spec.name))
        manifest["cases"].append({"name": spec.name, "spec": str(path), "result": result,
                                  "telemetry_before": before, "telemetry_after": None if args.write_only else telemetry()})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(spec.name, result["status"], flush=True)
        if not args.write_only and result["status"] != "passed":
            raise SystemExit(result.get("exit_code", 2))


if __name__ == "__main__":
    main()
