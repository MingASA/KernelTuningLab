"""Opt-in physical GPU regressions: KT_RUN_GPU_TESTS=1 pytest -q this_file."""

import os

import pytest

from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.catalog import get_catalog
from kernel_tuning_lab.kernel_runtime import AssetKernelAdapter
from kernel_tuning_lab.models import KernelConfig, OperatorSpec, ShapeSpec
from kernel_tuning_lab.runner import BenchmarkRunner

pytestmark = pytest.mark.skipif(os.environ.get("KT_RUN_GPU_TESTS") != "1", reason="explicit physical CUDA opt-in")


@pytest.mark.parametrize("kernel_ref,shape,restrictions", [
    ("triton/gemm", {"M": 37, "N": 71, "K": 65}, {}),
    ("triton_pipeline/w4a16_gemm", {"M": 7, "N": 33, "K": 129}, {"SPLIT_K": (2,)}),
    ("triton/moe_grouped_gemm", {"M": 19, "N": 32, "K": 128, "Groups": 4},
     {"SPLIT_K": (2,), "SCHEDULE": ("persistent",), "USE_TMA": (False,)}),
])
def test_graph_replays_full_plan_and_validates_actual_timed_outputs(kernel_ref, shape, restrictions):
    pytest.importorskip("torch")
    catalog = get_catalog()
    kernel = catalog.resolve("kernel", kernel_ref)
    definition = catalog.resolve("operator", kernel.operator)
    adapter = AssetKernelAdapter(kernel, definition, accept_claims=False, timing_mode="cuda_graph")
    hardware = adapter.probe_hardware()
    operator = OperatorSpec(definition.id, kind=definition.id, implementation=kernel.ref,
                            dtype="float16", parameters=dict(definition.parameters), version=definition.version)
    shape = ShapeSpec(shape)
    space = build_effective_space(kernel, hardware, operator, shape, restrictions)
    config = next(c for c in space.iter_configs() if space.validate(c, hardware, operator, shape).valid)
    runner = BenchmarkRunner(adapter, warmup=3, repetitions=7, seed=7)
    record = runner.measure(hardware, operator, shape, config, space, resume=False)
    assert record.successful, record.failure_reason
    assert record.correctness_verified
    assert record.measurement_mode == "cuda_graph_events_full_flow_amortized"
    assert record.metadata["correctness"]["post_measurement"]["status"] == "passed"
    assert len(record.runtime_samples_ms) == 7
