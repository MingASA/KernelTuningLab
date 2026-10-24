"""Small contract checks for budget accounting and held-out information isolation."""

from dataclasses import replace
import tempfile
import unittest

from kernel_tuning_lab.datasets import split_records, validate_split_isolation
from kernel_tuning_lab.models import HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec, RecordStatus, SearchBudget, ShapeSpec
from kernel_tuning_lab.policies import HeuristicPolicy, PolicyRepository, SearchPolicy, evaluate_holdout, evaluate_policy
from kernel_tuning_lab.replay import compare_searchers, run_replay, summarize_results
from kernel_tuning_lab.search import GridSearch, SEARCHER_NAMES, TransferSearch, make_searcher
from kernel_tuning_lab.space import ConfigSpace


class ReplayPolicyContracts(unittest.TestCase):
    def setUp(self):
        self.hardware = HardwareSpec("MOCK-contract")
        self.operator = OperatorSpec("contract")
        self.shape = ShapeSpec({"M": 64, "N": 64, "K": 64})
        self.space = ConfigSpace({"x": (1, 2, 3, 4)})
        self.rows = tuple(MeasurementRecord(self.hardware, self.operator, self.shape, config,
                                           RecordStatus.SUCCESS, runtime_samples_ms=(float(5 - config.get("x")),),
                                           wall_time_s=0.01, provenance="mock", split="test")
                          for config in self.space.iter_configs())

    def test_every_baseline_is_reproducible_and_reaches_reference(self):
        for name in SEARCHER_NAMES:
            left = run_replay(self.rows, make_searcher(name, seed=7), self.space, SearchBudget(4))
            right = run_replay(self.rows, make_searcher(name, seed=7), self.space, SearchBudget(4))
            self.assertEqual([s.config.id for s in left.trace], [s.config.id for s in right.trace])
            self.assertEqual(left.regret, 0)
            self.assertEqual(left.evaluations, 4)

    def test_unrevealed_labels_cannot_change_search_decisions(self):
        captured = []
        class SpyGrid(GridSearch):
            def suggest(inner, hardware, operator, shape, space, history, budget, batch_size=1):
                captured.append(tuple((r.config.get("x"), r.median_ms) for r in history))
                self.assertFalse(hasattr(history, "oracle_best_latency_ms"))
                self.assertTrue(all(r.config.get("x") <= len(history) for r in history))
                return super().suggest(hardware, operator, shape, space, history, budget, batch_size)
        first = run_replay(self.rows, SpyGrid(), self.space, SearchBudget(2))
        first_history = list(captured)
        captured.clear()
        perturbed = (*self.rows[:-1], replace(self.rows[-1], runtime_samples_ms=(0.0001,)))
        second = run_replay(perturbed, SpyGrid(), self.space, SearchBudget(2))
        self.assertEqual(first_history, captured)
        self.assertEqual([s.config.id for s in first.trace], [s.config.id for s in second.trace])
        self.assertNotEqual(first.regret, second.regret)  # Only the scorer sees the changed optimum.

    def test_missing_failures_and_unknown_correctness_consume_budget(self):
        failed = replace(self.rows[1], status=RecordStatus.COMPILE_ERROR,
                         runtime_samples_ms=(), failure_reason="compile failed")
        unknown = replace(self.rows[2], status=RecordStatus.CORRECTNESS_UNKNOWN,
                          runtime_samples_ms=(0.0001,), failure_reason="correctness unavailable")
        result = run_replay((failed, unknown, self.rows[3]), GridSearch(), self.space, SearchBudget(4))
        self.assertEqual(result.evaluations, 4)
        self.assertEqual(result.missing_evaluations, 1)
        self.assertEqual(result.failed_evaluations, 2)
        self.assertEqual(result.oracle_best_latency_ms, 1)
        self.assertEqual(result.regret, 0)
        self.assertIsNone(result.source_measurement_cost_s)
        self.assertEqual(result.cost_coverage, 0.75)

    def test_known_source_cost_enforces_wall_budget_separately_from_cpu(self):
        rows = tuple(replace(r, wall_time_s=1.0) for r in self.rows)
        result = run_replay(rows, GridSearch(), self.space, SearchBudget(4, 0.5))
        self.assertEqual(result.evaluations, 1)
        self.assertEqual(result.source_measurement_cost_s, 1.0)
        self.assertLess(result.cpu_wall_time_s, 0.5)
        self.assertEqual(result.stop_reason, "wall_time_budget")

    def test_case_and_unseen_identity_splits(self):
        records = tuple(replace(record, hardware=replace(self.hardware, name=f"GPU-{gpu}"),
                                shape=ShapeSpec({"M": 32 * (shape + 1)}))
                        for gpu in range(4) for shape in range(4) for record in self.rows)
        for holdout in ("case", "shape", "hardware"):
            forward = split_records(records, seed=9, holdout=holdout)
            backward = split_records(reversed(records), seed=9, holdout=holdout)
            self.assertEqual(forward.manifest(), backward.manifest())
            self.assertTrue(forward.train and forward.validation and forward.test)
            validate_split_isolation((*forward.train, *forward.validation, *forward.test), holdout=holdout)
        with self.assertRaises(ValueError):
            validate_split_isolation((self.rows[0], replace(self.rows[1], split="train")))

    def test_training_rejects_test_rows_and_target_case(self):
        with self.assertRaises(ValueError):
            TransferSearch(training_records=self.rows)
        policy = SearchPolicy(self.space, algorithm="transfer",
                              training_records=tuple(replace(r, split="train") for r in self.rows))
        with self.assertRaises(ValueError):
            evaluate_policy(policy, self.rows, self.space)
        with self.assertRaises(ValueError):
            compare_searchers((*self.rows, replace(self.rows[0], split="train")), self.space)

    def test_invalid_policy_has_legal_fallback_and_select_uses_no_deployment_measurements(self):
        class InvalidPolicy(HeuristicPolicy):
            def select_config(inner, hardware, op, shape):
                return (KernelConfig({"x": 999}),)
        score = evaluate_policy(InvalidPolicy(self.space), self.rows, self.space, mode="select")
        self.assertEqual(score.deployment_evaluations, 0)
        self.assertEqual(score.scorer_lookups, 1)
        self.assertTrue(any(e["event"] == "heuristic_fallback" for e in score.diagnostics))
        self.assertTrue(self.space.validate(score.replay.best_config, self.hardware, self.operator, self.shape).valid)

    def test_policy_versions_are_immutable_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = PolicyRepository(directory)
            policy = SearchPolicy(self.space, seed=1)
            manifest = repository.register(policy, "v1")
            self.assertEqual(repository.register(policy, "v1"), manifest)
            with self.assertRaises(ValueError):
                repository.register(SearchPolicy(self.space, seed=2), "v1")
            evaluation = evaluate_policy(policy, self.rows, self.space, budget=SearchBudget(2))
            self.assertTrue(repository.log_evaluation(evaluation).is_file())

    def test_unseen_gpu_interface_rejects_training_gpu_overlap(self):
        source = tuple(replace(r, shape=ShapeSpec({"M": 128}), split="train") for r in self.rows)
        policy = SearchPolicy(self.space, algorithm="transfer", training_records=source)
        with self.assertRaises(ValueError):
            evaluate_holdout(policy, self.rows, self.space, unseen="hardware")
        self.assertEqual(len(evaluate_holdout(policy, self.rows, self.space, unseen="case")), 1)

    def test_replay_rejects_incompatible_source_contexts(self):
        variants = (
            replace(self.rows[1], provenance="measured"),
            replace(self.rows[1], measurement_mode="another_protocol"),
            replace(self.rows[1], metadata={"protocol": {"cache_state": "cold"}}),
            replace(self.rows[1], environment={"triton": "another-version"}),
            replace(self.rows[1], warmup=9),
            replace(self.rows[1], repetitions=7),
        )
        for changed in variants:
            with self.assertRaisesRegex(ValueError, "source contexts are incompatible"):
                run_replay((self.rows[0], changed), GridSearch(), self.space)
        repeated = replace(self.rows[0], experiment_id="independent-repeat", seed=123)
        result = run_replay((*self.rows, repeated), GridSearch(), self.space, SearchBudget(4))
        self.assertEqual(result.regret, 0)
        self.assertTrue(result.source_context_id)

    def test_summaries_do_not_pool_operators_hardware_or_protocols(self):
        result = run_replay(self.rows, GridSearch(), self.space, SearchBudget(2))
        cohorts = (result,
                   replace(result, operator_id="another-op", operator_name="layernorm"),
                   replace(result, hardware_id="another-gpu"),
                   replace(result, provenance="measured"),
                   replace(result, source_context_id="another-protocol"))
        summary = summarize_results(cohorts)
        self.assertEqual(len(summary), 5)
        self.assertTrue(all(row["runs"] == 1 for row in summary))
        self.assertTrue(all("hardware_id" in row and "operator_id" in row and "source_context_id" in row
                            for row in summary))

    def test_comparison_separates_multiple_protocols_of_same_case(self):
        alternate = tuple(replace(record, environment={"driver": "alternate"}) for record in self.rows)
        results = compare_searchers((*self.rows, *alternate), self.space, budgets=(2,), seeds=(7,),
                                    searcher_names=("grid",))
        self.assertEqual(len(results), 2)
        self.assertEqual(len({result.case_id for result in results}), 1)
        self.assertEqual(len({result.source_context_id for result in results}), 2)

    def test_select_scores_ranked_top_one_without_oracle_candidate_selection(self):
        first, oracle_winner = self.rows[0].config, self.rows[-1].config
        class RankedPolicy(HeuristicPolicy):
            def select_config(inner, hardware, op, shape):
                return (first, oracle_winner)
        result = evaluate_policy(RankedPolicy(self.space), self.rows, self.space, mode="select")
        self.assertEqual(result.scorer_lookups, 1)
        self.assertEqual(result.deployment_evaluations, 0)
        self.assertEqual(result.replay.best_latency_ms, self.rows[0].median_ms)
        self.assertEqual(result.replay.best_config, first)
        self.assertEqual(result.predicted_candidates, (first, oracle_winner))


if __name__ == "__main__":
    unittest.main()
