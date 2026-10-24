"""Small exact checks: ``python3 -m unittest kernel_tuning_lab.l3.selfcheck``."""

from dataclasses import replace
from itertools import combinations
import random
import unittest

from ..models import HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec, RecordStatus, ShapeSpec, WorkloadDistribution
from ..space import ConfigSpace
from . import (OracleLimitError, PerformanceMatrix, ShapeDispatcher, UnknownShapeError,
               bucket_baseline, evaluate_selection, greedy, greedy_local_swap, k_loss_curve, oracle)


def matrix(rows, weights=None):
    shapes = tuple(ShapeSpec({"M": i + 1}) for i in range(len(rows)))
    workload = WorkloadDistribution(shapes, tuple(weights or [1] * len(rows)))
    configs = tuple(KernelConfig({"x": j}) for j in range(len(rows[0])))
    return PerformanceMatrix(workload, configs, rows, provenance="mock")


class L3CorrectnessChecks(unittest.TestCase):
    def test_weighted_objective_and_relative_loss_are_ratios_of_expectations(self):
        table = matrix(((1, 2), (100, 10)), (0.8, 0.2))
        result = oracle(table, 1)
        self.assertEqual(result.selected_indices, (1,))
        self.assertAlmostEqual(result.expected_latency_ms, 3.6)
        self.assertAlmostEqual(result.reference_latency_ms, 2.8)
        self.assertAlmostEqual(result.relative_loss, 3.6 / 2.8 - 1)
        self.assertNotAlmostEqual(result.relative_loss, 0.8)

    def test_exact_small_oracle_and_improving_swap(self):
        table = matrix(((4, 1, 9), (4, 1, 9), (4, 9, 1), (4, 9, 1)))
        self.assertEqual(greedy(table, 2).expected_latency_ms, 2.5)
        exact = oracle(table, 2)
        improved = greedy_local_swap(table, 2)
        self.assertEqual(exact.selected_indices, (1, 2))
        self.assertEqual(exact.expected_latency_ms, 1.0)
        self.assertEqual(improved.expected_latency_ms, exact.expected_latency_ms)
        self.assertEqual(improved.metadata["swaps"], 1)

    def test_random_small_oracle_matches_independent_exhaustive_objective(self):
        rng = random.Random(17)
        for _ in range(12):
            rows = tuple(tuple(rng.randint(1, 20) for _ in range(5)) for _ in range(4))
            table = matrix(rows, (1, 2, 3, 4))
            for k in (1, 2, 3):
                expected = min(sum(w * min(row[j] for j in chosen)
                                   for w, row in zip(table.workload.normalized_weights, rows))
                               for size in range(1, k + 1) for chosen in combinations(range(5), size))
                self.assertAlmostEqual(oracle(table, k).expected_latency_ms, expected)

    def test_missing_cells_remain_missing_and_uncovered_objective_is_conditional(self):
        table = matrix(((4, None), (None, None)), (1, 3))
        for solve in (oracle, greedy, greedy_local_swap, bucket_baseline):
            result = solve(table, 2)
            self.assertIsNone(result.expected_latency_ms)
            self.assertIsNone(result.relative_loss)
            self.assertIsNone(result.reference_latency_ms)
            self.assertEqual(result.coverage, 0.25)
            self.assertEqual(result.conditional_latency_ms, 4)
            self.assertEqual(result.metadata["reference_scope"], "observed_cells_only")
        dispatch = ShapeDispatcher.from_result(table, oracle(table, 2))
        with self.assertRaises(UnknownShapeError):
            dispatch(table.workload.shapes[1])
        with self.assertRaises(ValueError):
            PerformanceMatrix(table.workload, table.configs, table.latencies_ms, provenance="predicted")

    def test_oracle_refuses_excess_work_but_accepts_exact_lower_bound_certificate(self):
        table = matrix(((1, 9, 9), (9, 1, 9), (9, 9, 1)))
        with self.assertRaises(OracleLimitError):
            oracle(table, 1, max_combinations=2)
        exact = oracle(table, 3, max_combinations=1)
        self.assertEqual(exact.expected_latency_ms, 1)
        self.assertTrue(exact.exact)
        self.assertIn("optimality_certificate", exact.metadata)

    def test_dispatch_checks_selected_membership_and_operator_legality(self):
        table = matrix(((3, 1), (1, 3)))
        result = oracle(table, 1)
        hardware, operator = HardwareSpec("mock"), OperatorSpec("demo")
        valid_space = ConfigSpace({"x": (0, 1)})
        dispatcher = ShapeDispatcher.from_result(table, result, hardware=hardware, operator=operator, space=valid_space)
        self.assertEqual(dispatcher(table.workload.shapes[0]).id, result.configs[0].id)
        with self.assertRaises(UnknownShapeError):
            dispatcher(ShapeSpec({"M": 999}))
        with self.assertRaises(ValueError):
            ShapeDispatcher.from_result(table, result, hardware=hardware)
        bad = replace(result, assignment_indices=(1, 1))
        with self.assertRaises(ValueError):
            ShapeDispatcher.from_result(table, bad)
        invalid_space = ConfigSpace({"x": (0, 1)}, constraints=(lambda c, h, o, s: "forbidden",))
        with self.assertRaisesRegex(ValueError, "illegal dispatch"):
            ShapeDispatcher.from_result(table, result, hardware=hardware, operator=operator, space=invalid_space)

    def test_from_records_deduplicates_and_preserves_source_and_missing_failures(self):
        table = matrix(((1, 2), (3, 4)))
        hardware, operator = HardwareSpec("mock"), OperatorSpec("demo")
        one = MeasurementRecord(hardware, operator, table.workload.shapes[0], table.configs[0],
                                RecordStatus.SUCCESS, runtime_samples_ms=(1, 2, 3), provenance="mock")
        failed = MeasurementRecord(hardware, operator, table.workload.shapes[1], table.configs[0],
                                   RecordStatus.COMPILE_ERROR, failure_reason="expected", provenance="mock")
        records = PerformanceMatrix.from_records(table.workload, table.configs, [one, one, failed])
        self.assertEqual(records.provenance, "mock")
        self.assertEqual(records.latencies_ms, ((2.0, None), (None, None)))
        self.assertEqual(records.metadata["record_count"], 2)
        other_hardware = replace(one, hardware=HardwareSpec("other"))
        with self.assertRaisesRegex(ValueError, "contexts"):
            PerformanceMatrix.from_records(table.workload, table.configs, [one, other_hardware])
        unverified = replace(one, metadata={"correctness_verified": False})
        unknown = PerformanceMatrix.from_records(table.workload, table.configs, [unverified])
        self.assertEqual(unknown.coverage, 0)

    def test_default_k_curve_covers_requested_budgets_without_exceeding_them(self):
        table = matrix(((1, 5, 4), (5, 1, 4), (5, 4, 1)))
        curve = k_loss_curve(table)
        self.assertEqual({result.k for result in curve}, {1, 2, 4, 8, 16})
        self.assertEqual(len(curve), 15)
        self.assertTrue(all(result.variant_count <= result.k for result in curve))
        for result in curve:
            self.assertTrue(set(result.assignment_indices) <= set(result.selected_indices))
        with self.assertRaises(ValueError):
            evaluate_selection(table, (0, 1), k=1)


if __name__ == "__main__":
    unittest.main()
