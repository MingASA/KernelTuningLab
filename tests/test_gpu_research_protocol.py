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
