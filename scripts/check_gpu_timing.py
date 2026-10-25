"""Independent same-implementation timing control; run with the project Python.

Writes raw observations, never feeds unlike scopes into a performance matrix.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

from kernel_tuning_lab.backends.triton_gemm import TritonGemmAdapter
from kernel_tuning_lab.catalog import get_catalog
from kernel_tuning_lab.kernel_runtime import AssetKernelAdapter
from kernel_tuning_lab.models import KernelConfig, OperatorSpec, ShapeSpec


def telemetry():
    return subprocess.run([
        "nvidia-smi", "--query-gpu=timestamp,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
        "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()


def run(output, rounds=5):
    import torch
    import triton.testing

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    adapter = TritonGemmAdapter()
    hardware = adapter.probe_hardware()
    catalog = get_catalog()
    asset = AssetKernelAdapter(catalog.resolve("kernel", "triton/gemm@1.0.0"),
                               catalog.resolve("operator", "gemm@1.0.0"), accept_claims=False)
    graph_asset = AssetKernelAdapter(catalog.resolve("kernel", "triton/gemm@1.0.0"),
                                     catalog.resolve("operator", "gemm@1.0.0"), accept_claims=False,
                                     timing_mode="cuda_graph")
    config = KernelConfig({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2})
    report = {"environment": adapter.environment(), "asset_environment": asset.environment(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "config": config.params, "seed": 7, "warmup": 50, "rounds": rounds,
              "notice": "Same gemm_kernel source, inputs and config; scopes remain separate. Graph replay excludes host launch and uses hot reused buffers.",
              "cases": []}
    for m, n, k in ((32, 64, 64), (512, 512, 512), (1024, 1024, 1024)):
        shape = ShapeSpec({"M": m, "N": n, "K": k})
        operator = OperatorSpec("gemm", kind="gemm", implementation="triton_gemm", dtype="float16")
        prepared = adapter.prepare(hardware, operator, shape, 7)
        compiled = adapter.compile(prepared, config)
        assert adapter.check_correctness(compiled)[0]
        asset_prepared = asset.prepare(hardware, operator, shape, 7)
        assert torch.equal(prepared.a, asset_prepared.context.inputs["a"])
        assert torch.equal(prepared.b, asset_prepared.context.inputs["b"])
        asset_compiled = asset.compile(asset_prepared, config)
        assert asset.check_correctness(asset_compiled)[0]
        graph_prepared = graph_asset.prepare(hardware, operator, shape, 7)
        graph_compiled = graph_asset.compile(graph_prepared, config)
        assert graph_asset.check_correctness(graph_compiled)[0]
        adapter.warmup(compiled, 50)
        asset.warmup(asset_compiled, 50)
        case = {"shape": shape.dims, "estimated_input_output_reference_bytes": 2 * (m*k+k*n+2*m*n),
                "telemetry_before": telemetry(), "rounds": []}
        for index in range(rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            independent = []
            for _ in range(50):
                start.record()
                compiled.launch()
                end.record()
                end.synchronize()
                independent.append(start.elapsed_time(end))
            # A batch amortizes event boundaries, but still includes Python launch gaps.
            start.record()
            for _ in range(100):
                compiled.launch()
            end.record()
            end.synchronize()
            batch = start.elapsed_time(end) / 100
            graph = triton.testing.do_bench_cudagraph(compiled.launch, rep=20, return_mode="all")
            torch_graph = triton.testing.do_bench_cudagraph(
                lambda: torch.mm(prepared.a, prepared.b, out=prepared.c), rep=20, return_mode="all")
            case["rounds"].append({"round": index, "v1_event_ms": adapter.measure(compiled, 50),
                "independent_event_ms": independent, "event_batch_per_call_ms": batch,
                "v2_full_flow_ms": asset.measure(asset_compiled, 50),
                "v2_graph_ms": graph_asset.measure(graph_compiled, 50),
                "triton_graph_ms": graph, "torch_mm_graph_ms": torch_graph})
            assert asset.check_after_measurement(asset_compiled)[0]
            assert graph_asset.check_after_measurement(graph_compiled)[0]
            assert adapter.check_correctness(compiled)[0]
        case["telemetry_after"] = telemetry()
        report["cases"].append(case)
        (output / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
        print(shape.dims, {key: statistics.median([value for row in case["rounds"] for value in row[key]])
                             for key in ("v1_event_ms", "independent_event_ms", "v2_full_flow_ms", "v2_graph_ms", "triton_graph_ms", "torch_mm_graph_ms")}, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    run(args.output, args.rounds)
