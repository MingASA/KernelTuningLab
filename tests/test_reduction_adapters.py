"""GPU adapter validation stays runnable without optional GPU libraries."""

from dataclasses import replace
import subprocess
import sys
import unittest

from kernel_tuning_lab.backends.triton_reductions import TritonLayerNormAdapter, TritonSoftmaxAdapter
from kernel_tuning_lab.models import HardwareSpec, KernelConfig, OperatorSpec, ShapeSpec


class ReductionAdapterContracts(unittest.TestCase):
    def setUp(self):
        self.hardware = HardwareSpec("test-H200", "sm_90", sm_count=132, smem_per_block=227328)
        self.config = KernelConfig({"BLOCK_SIZE": 256, "num_warps": 4, "num_stages": 1})
        self.shape = ShapeSpec({"M": 7, "N": 193})

    def adapter(self, adapter_type):
        # Only physical runtime probing is replaced. All schema checks are real.
        adapter = object.__new__(adapter_type)
        adapter.probe_hardware = lambda: self.hardware
        return adapter

    def test_import_does_not_load_torch_or_triton(self):
        command = "import sys; import kernel_tuning_lab.backends.triton_reductions; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_non_power_of_two_rows_and_all_dtypes_are_accepted(self):
        for cls, kind in ((TritonSoftmaxAdapter, "softmax"), (TritonLayerNormAdapter, "layernorm")):
            for dtype in ("float16", "bfloat16", "float32"):
                operator = OperatorSpec(kind, kind=kind, implementation=f"triton_{kind}", dtype=dtype)
                result = self.adapter(cls).validate(self.hardware, operator, self.shape, self.config)
                self.assertTrue(result.valid, result.reasons)
                result = self.adapter(cls).validate(self.hardware, operator, ShapeSpec({"M": 33, "N": 1025}),
                                                    KernelConfig({"BLOCK_SIZE": 2048, "num_warps": 8, "num_stages": 2}))
                self.assertTrue(result.valid, result.reasons)

    def test_invalid_shapes_parameters_and_unsupported_knobs_are_rejected(self):
        adapter = self.adapter(TritonLayerNormAdapter)
        operator = OperatorSpec("layernorm", kind="layernorm", implementation="triton_layernorm")
        bad_inputs = (
            (operator, self.shape, KernelConfig({"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 1})),
            (operator, self.shape, KernelConfig({**self.config.params, "cluster": 2})),
            (replace(operator, parameters={"affine": True}), self.shape, self.config),
            (replace(operator, parameters={"elementwise_affine": True}), self.shape, self.config),
            (replace(operator, parameters={"eps": 0}), self.shape, self.config),
            (replace(operator, parameters={"axis": 0}), self.shape, self.config),
            (operator, ShapeSpec({"M": 7, "N": 193, "Batch": 2}), self.config),
        )
        for op, shape, config in bad_inputs:
            self.assertFalse(adapter.validate(self.hardware, op, shape, config).valid)


if __name__ == "__main__":
    unittest.main()
