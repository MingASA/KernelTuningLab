from pathlib import Path
import tempfile
import unittest

from kernel_tuning_lab.backends.mock import MockAdapter
from kernel_tuning_lab.models import HardwareSpec, OperatorSpec, SearchBudget, ShapeSpec
from kernel_tuning_lab.runner import BenchmarkRunner
from kernel_tuning_lab.search import GridSearch
from kernel_tuning_lab.space import ConfigSpace, gemm_constraint
from kernel_tuning_lab.storage import ExperimentStore
from kernel_tuning_lab.tuning import tune


class OnlineTuningContracts(unittest.TestCase):
    def setUp(self):
        self.hardware = HardwareSpec("MOCK-online")
        self.operator = OperatorSpec("gemm")
        self.shape = ShapeSpec({"M": 64, "N": 64, "K": 64})
        self.space = ConfigSpace({"BLOCK_M": (16, 32, 64, 128), "BLOCK_N": (32,), "BLOCK_K": (32,),
                                  "num_warps": (4,), "num_stages": (2,)}, constraints=(gemm_constraint,))

    def test_cached_observations_do_not_spend_new_physical_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExperimentStore(Path(directory) / "results.sqlite") as store:
                runner = BenchmarkRunner(MockAdapter(0, 0, 0), store=store)
                first = tune(runner, self.hardware, self.operator, self.shape, self.space, GridSearch(), SearchBudget(2))
                second = tune(runner, self.hardware, self.operator, self.shape, self.space, GridSearch(), SearchBudget(2))
                self.assertEqual(first.evaluations, 2)
                self.assertEqual(second.evaluations, 2)
                self.assertEqual(second.cache_hits, 2)
                self.assertEqual(second.attempts, 4)
                self.assertTrue(second.best_record.successful)

    def test_compile_failures_consume_budget_and_have_no_recommendation(self):
        runner = BenchmarkRunner(MockAdapter(1, 0, 0))
        result = tune(runner, self.hardware, self.operator, self.shape, self.space, GridSearch(), SearchBudget(3))
        self.assertEqual(result.evaluations, 3)
        self.assertTrue(all(record.status.value == "compile_error" for record in result.records))
        self.assertIsNone(result.best_config)

    def test_zero_budget_reveals_no_records(self):
        runner = BenchmarkRunner(MockAdapter(0, 0, 0))
        result = tune(runner, self.hardware, self.operator, self.shape, self.space, GridSearch(), SearchBudget(0))
        self.assertEqual(result.attempts, 0)
        self.assertIsNone(result.best_config)


if __name__ == "__main__":
    unittest.main()
