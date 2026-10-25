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
