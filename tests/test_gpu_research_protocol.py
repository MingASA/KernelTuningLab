"""CPU regressions for protocol mistakes exposed by physical validation."""

from dataclasses import replace

import pytest

from kernel_tuning_lab.l3 import PerformanceMatrix
from kernel_tuning_lab.models import (HardwareSpec, KernelConfig, MeasurementRecord,
                                      OperatorSpec, RecordStatus, ShapeSpec, WorkloadDistribution)


def _record():
    return MeasurementRecord(
        HardwareSpec("physical-test", architecture="sm_120"), OperatorSpec("gemm"),
        ShapeSpec({"M": 16, "N": 16, "K": 32}), KernelConfig({"tile": 16}),
        RecordStatus.SUCCESS, runtime_samples_ms=(0.01, 0.02), provenance="measured",
        measurement_mode="cuda_events_hot_single_kernel", environment={"torch": "test"},
        metadata={"correctness_verified": True, "protocol": {"atol": 0.02}})


@pytest.mark.parametrize("changes", [
    {"measurement_mode": "cuda_synchronized_full_flow_perf_counter"},
    {"environment": {"torch": "different"}},
    {"metadata": {"correctness_verified": True, "protocol": {"atol": 1.0}}},
    {"warmup": 19},
])
def test_l3_rejects_incompatible_measurements(changes):
    record = _record()
    workload = WorkloadDistribution((record.shape,), (1,))
    with pytest.raises(ValueError, match="measurement protocols"):
        PerformanceMatrix.from_records(workload, (record.config,), [record, replace(record, **changes)])


def test_l3_allows_seeds_and_shape_specific_space_identity():
    record = _record()
    second = replace(record, seed=7, experiment_id="repeat", shape=ShapeSpec({"M": 17, "N": 16, "K": 32}),
                     metadata={**record.metadata, "protocol": {"atol": 0.02, "config_space": {"fingerprint": "tail"}}})
    table = PerformanceMatrix.from_records(WorkloadDistribution((record.shape, second.shape), (1, 1)),
                                            (record.config,), [record, second])
    assert table.complete
    assert table.metadata["measurement_context_id"]


def test_graph_timing_is_explicit_and_cuda_only():
    from kernel_tuning_lab.asset_experiments import AssetExperiment
    from kernel_tuning_lab.catalog import get_catalog
    from kernel_tuning_lab.kernel_runtime import AssetKernelAdapter
    catalog = get_catalog()
    operator = catalog.resolve("operator", "gemm")
    with pytest.raises(ValueError, match="CUDA Kernel"):
        AssetKernelAdapter(catalog.resolve("kernel", "python/gemm"), operator, timing_mode="cuda_graph")
    with pytest.raises(ValueError, match="timing_mode"):
        AssetExperiment.from_dict({"name": "bad", "operator": "gemm", "timing_mode": "silently_replace"})
    for size in (True, 0, 4097):
        with pytest.raises(ValueError, match="batch_size"):
            AssetKernelAdapter(catalog.resolve("kernel", "triton/gemm"), operator,
                               timing_mode="cuda_graph", cuda_graph_batch_size=size)


def test_graph_scope_and_batch_size_change_cache_identity():
    from kernel_tuning_lab.catalog import get_catalog
    from kernel_tuning_lab.kernel_runtime import AssetKernelAdapter
    catalog = get_catalog()
    kernel, operator = catalog.resolve("kernel", "triton/gemm"), catalog.resolve("operator", "gemm")
    full = AssetKernelAdapter(kernel, operator)
    graph = AssetKernelAdapter(kernel, operator, timing_mode="cuda_graph")
    bigger = AssetKernelAdapter(kernel, operator, timing_mode="cuda_graph", cuda_graph_batch_size=256)
    record = _record()
    records = [replace(record, measurement_mode=a.measurement_mode, metadata={"protocol": a.protocol_metadata()})
               for a in (full, graph, bigger)]
    assert len({r.cache_key for r in records}) == 3
    assert full.protocol_metadata()["cuda_graph"] is None
    assert graph.protocol_metadata()["cuda_graph"]["host_steps_replayed"] is False


@pytest.mark.parametrize("kernel_ref", [f"{prefix}/w{bits}a16_gemm"
                                        for prefix in ("triton", "triton_pipeline") for bits in (4, 3, 2)])
def test_quantized_partial_m_group_is_legal(kernel_ref):
    from kernel_tuning_lab.asset_space import build_effective_space
    from kernel_tuning_lab.catalog import get_catalog
    catalog = get_catalog()
    kernel = catalog.resolve("kernel", kernel_ref)
    definition = catalog.resolve("operator", kernel.operator)
    hardware = HardwareSpec("test", architecture="sm_120", smem_per_block=101376)
    operator = OperatorSpec(definition.id, kind=definition.id, implementation=kernel.ref,
                            dtype="float16", version=definition.version, parameters=dict(definition.parameters))
    shape = ShapeSpec({"M": 1, "N": 65, "K": 129})
    space = build_effective_space(kernel, hardware, operator, shape, {"GROUP_M": (8,)})
    assert any(space.validate(c, hardware, operator, shape).valid for c in space.iter_configs())


@pytest.mark.parametrize("bits,bm,stages,group_size,expected", [
    (3, 64, 2, 32, 122880), (3, 64, 3, 32, 221184),
    (4, 16, 3, 32, 149504), (4, 16, 3, 64, 84992),
])
def test_fused_quantized_staging_respects_actual_hardware_capacity(bits, bm, stages, group_size, expected):
    from kernel_tuning_lab.assets import SpaceContext
    from kernel_tuning_lab.builtin_assets.quantized import _gemm_constraint, _pipeline_constraint
    config = KernelConfig({"BLOCK_M": bm, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8,
                           "num_stages": stages, "GROUP_M": 1, "SPLIT_K": 1})
    operator = OperatorSpec(f"w{bits}a16_gemm", kind=f"w{bits}a16_gemm", dtype="bfloat16",
                            parameters={"group_size": group_size})
    shape = ShapeSpec({"M": 2, "N": 7, "K": 65})
    small = HardwareSpec("5060ti", architecture="sm_120", smem_per_block=101376)
    hopper = HardwareSpec("H200", architecture="sm_90", smem_per_block=232448)
    reason = _gemm_constraint(config, SpaceContext(small, operator, shape))
    if expected > small.smem_per_block:
        assert reason and "staging" in reason
    else:
        assert reason is None
    assert _gemm_constraint(config, SpaceContext(hopper, operator, shape)) is None
    # The materialized dense pipeline has no staged integer unpack buffers.
    pipeline = KernelConfig({**config.params, "DEQUANT_BLOCK_SIZE": 512})
    assert _pipeline_constraint(pipeline, SpaceContext(small, operator, shape)) is None
