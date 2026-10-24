# Search, replay, and policy protocol

The core is standard-library Python. No GPU package, AI SDK, model download, or paid API is required.

## Search contract

```python
from kernel_tuning_lab.search import make_searcher
searcher = make_searcher("surrogate", seed=7)
configs = searcher.suggest(hardware, operator, shape, space, history, remaining_budget, batch_size=1)
```

`make_searcher(name, seed=0, training_records=(), candidate_pool_size=512, exhaustive_threshold=4096)` preserves exhaustive candidate ordering for small spaces. Above the threshold, algorithms maintain a bounded reproducible legal candidate pool and refill it as observations consume candidates. Random/heuristic/surrogate search use sampled pools; local search samples bounded single-dimension neighbors before considering its pool; transfer also retains legal training candidates; grid uses an index-decoded cursor with bounded validation chunks. They use `space.config_at()` / `sample_configs()` rather than expanding the Cartesian product, including spaces with 100000 or more candidates. Validation work per refill/chunk is bounded, so very sparse legal regions can cause an empty candidate result before the full declared space has been searched. These modes are candidate-pool approximations and do not claim exhaustive search. Pool size and threshold are recorded in policy specifications and replay results.

`history` is a sequence of revealed `MeasurementRecord` objects. The replay evaluator supplies `SearchHistory`, whose `attempted_config_ids` additionally identifies already requested configurations without source measurements. This contains no information about unrequested configurations. It represents missing observations without fabricating measurement records. Outputs are legal `KernelConfig` candidates; an empty list stops the search. All requested configurations, including missing, invalid, compile failures and correctness failures, count against the evaluation budget.

The algorithms are `random`, `grid`, `heuristic`, `local`, `surrogate`, and `transfer`. Random and local search use an explicit random seed. Grid search uses declared configuration-domain ordering. The heuristic orders GEMM candidates by padding, parallelism, and coarse resource use. Surrogate search fits a distance-weighted log-latency predictor using only the current case's revealed successful observations, with an exploration bonus. Transfer normalizes observations within each training case and weights source shapes/hardware by similarity. Its source records must be training data and exclude the target case. Unknown correctness is never a successful observation.

## Online adaptive tuning

```python
from kernel_tuning_lab.tuning import tune
result = tune(runner, hardware, operator, shape, space, make_searcher("surrogate", seed=7),
              SearchBudget(16, 60.0), split="train", resume=True)
```

`TuningResult` exposes revealed records, a verified best record/config, new physical evaluation count, request count, cache hits, actual wall time, and a stop reason. Each call to the searcher receives only the current tuning call's revealed history. Cache hits reveal existing observations without consuming new physical evaluations; all fresh invalid/compile/correctness/runtime failures consume the evaluation budget. Cache lookup and policy execution count against actual wall time. Once the physical limit is reached, no further requests are admitted. A single compile/measurement is atomic and may exceed the wall-time limit; the result records actual elapsed time. Repeated already-revealed candidates stop the search, preventing cache-only loops. Cached successes are revalidated against the current configuration space, and only successful legal records can become the recommendation.

## Replay and scoring

```python
from kernel_tuning_lab.models import SearchBudget
from kernel_tuning_lab.replay import run_replay, compare_searchers, summarize_results
result = run_replay(one_case_records, searcher, space, SearchBudget(16), seed=7)
results = compare_searchers(records, space, budgets=(8,16,32,64,128), seeds=(0,1,2),
                            split="test", training_records=train_records)
summary = summarize_results(results)
```

The scorer privately owns source measurements. Algorithms receive hardware, operator, shape, configuration space, remaining budget, and only the observations revealed so far. Nested mutable fields are copied when observations leave the private table. Oracle latency and regret are computed by the scorer after the final algorithm call. No oracle values enter the algorithm's history. Python policies are trusted code: this boundary prevents ordinary accidental data leakage, and is not a sandbox against malicious Python introspection or file access.

`run_replay` requires exactly one `(hardware, operator, shape)` case, one split, and one compatible source context. It rejects mixed provenance, measurement modes, protocol metadata, environments, warmup counts, or repetition counts. Imported contexts also include the source file digest (or source manifest blob ID/URL when unavailable), preserving distinct upstream experiments. Separate experiment IDs and random seeds are allowed within a compatible measured/mock context. `source_context_id(record)` exposes the reproducible context identity. `compare_searchers` checks case-level split isolation, partitions each target case by source context, and creates a fresh seeded searcher for every budget/case/context trial. Training may span different hardware and environments, but must exclude every target case. Repeated config rows are selected by stable record ID, never by latency. Missing observations remain missing. Regret is `best_found / best_known_legal - 1`, with `None` when no valid result/reference exists. Partial-table regret is relative to the observed reference, not the unknowable global optimum.

For small spaces, `oracle_coverage` is the exact fraction of legal configurations with any source record, including recorded failures. Above the exhaustive threshold, the scorer validates only supplied source records and never enumerates the candidate space: `oracle_coverage=None` and `coverage_scope="known_observations_only"`. `known_coverage_raw_space` gives observed independent domain-valid configurations divided by the full unpruned `space_cardinality`; that denominator includes configurations that may violate constraints. It is not legal-space coverage. Results and summaries explicitly record `candidate_pool_approximate`, pool size and threshold, so a large partial replay cannot be mistaken for a global optimum or exhaustive measurement campaign.

`ReplayResult.to_dict()` includes algorithm, case identities, readable operator/hardware names, source context, requested budget, actual evaluation count, best latency/config, regret, known optimum, source coverage, failure/missing counts, and a per-attempt trace. `target_ratio=1.05` means reaching latency at most 1.05 times the known optimum; `target_evaluations` records the first qualifying evaluation. Summaries average shapes and seeds within matching hardware, operator, provenance, source context, split, algorithm, count/time budgets, and target ratio. They do not pool incomparable operators, GPUs or measurement protocols. They report regret mean/standard deviation, mean within-case seed variance, reached-target counts, actual attempts and costs; raw results retain individual seed trajectories.

Actual CPU execution time is `cpu_wall_time_s`. Source measurement cost is `source_measurement_cost_s`, taken only from a recorded `wall_time_s` or an explicit `metadata.measurement_cost_known=True`. It is `None` if any requested source cost is unknown; `known_measurement_cost_s` and `cost_coverage` preserve partial information. A false cost-known flag overrides a positive duration. Partial compilation/sample timings are not mislabeled as a total measurement cost. Mock results remain labeled `provenance="mock"`: the bundled runner's recorded source wall time is CPU collection overhead, not a GPU measurement cost, while its runtime samples are synthetic. In replay wall-time budgets count actual CPU time plus known source cost; an atomic evaluation may cross the limit. A missing source cost cannot be used to enforce a physical time limit, so evaluation limits remain necessary.

## Dataset isolation

```python
from kernel_tuning_lab.datasets import split_records, validate_split_isolation
splits = split_records(records, seed=7, holdout="case")
splits = split_records(records, seed=7, holdout="shape")     # unseen shapes
splits = split_records(records, seed=7, holdout="hardware")  # unseen GPUs
manifest = splits.manifest()
```

Splitting groups entire cases (or all matching shape/GPU identities) before assigning records. It never partitions individual configurations from a case across sets. Input ordering does not change assignment. `split_records` intentionally replaces existing labels to create a new experimental split; `validate_split_isolation` validates existing labels. Fractions use largest-remainder allocation, retaining nonempty requested partitions when enough independent groups exist. Tiny datasets may leave a partition empty; do not duplicate cases to fill it. Manifest case identities should be retained alongside experiments. Training and transfer accept only train records; validation selects algorithm choices, and the frozen policy is evaluated once on test. Target-case observations may guide within-budget target search but must never refit global policy parameters.

## Policy contract and independent evaluation

Policies implement both methods:

```python
select_config(hardware, op, shape) -> tuple[KernelConfig, ...]
search_config(hardware, op, shape, history, budget) -> PolicyDecision
```

`select_config` predicts zero-shot candidates without new target measurements. `PolicyDecision(candidates=(), recommendation=None)` requests the next measurements and reports an incumbent or prediction. Calls repeat with fresh history until the budget is exhausted or candidates are empty. A recommendation is considered verified only when backed by a successful revealed measurement. Search scoring uses measured history, never an unmeasured recommendation.

```python
from kernel_tuning_lab.policies import HeuristicPolicy, SearchPolicy, TransferPolicy
from kernel_tuning_lab.policies import evaluate_policy, evaluate_holdout, PolicyRepository
policy = SearchPolicy(space, algorithm="surrogate", seed=7)
repository = PolicyRepository("artifacts/policies")
manifest = repository.register(policy, version="experiment_001")
evaluation = evaluate_policy(policy, one_case_records, space, mode="search", budget=SearchBudget(16))
repository.log_evaluation(evaluation)
unseen_gpu_scores = evaluate_holdout(policy, test_records, space, unseen="hardware")
```

`HeuristicPolicy` is an analytic baseline; `TransferPolicy` reuses training measurements. `FunctionPolicy` adapts Python functions or decision rules. `SafePolicy` catches policy exceptions and validates candidate domains and hardware/operator constraints. Invalid outputs trigger a legal heuristic fallback with diagnostics. Empty legal spaces produce no candidate. Recommendations unsupported by revealed data never become verified measurements. `evaluate_policy` uses a private replay scorer and a copy of the policy; evaluating one test case does not update the policy subsequently evaluated on another case. The evaluator checks declared training records against the held-out case. `evaluate_holdout(..., unseen="shape"|"hardware")` additionally rejects training overlap at that identity level.

Select-mode scoring evaluates only the ranked first candidate, with one offline table lookup when a legal prediction exists. Other ranked predictions remain in `PolicyEvaluation.predicted_candidates` for inspection. The scorer never uses the oracle to select the fastest prediction from a set. `deployment_evaluations=0` clearly separates zero-shot use from `scorer_lookups`, while search mode counts requested target measurements. Neither prediction nor offline lookup creates a new `MeasurementRecord`; a zero-shot prediction remains unverified for deployment even when its existing source performance is scored offline.

The local policy repository stores immutable manifests, source snapshots and SHA-256 hashes, source specification, training case identities, and append-only evaluation JSONL logs. Changing source/parameters under an existing version is rejected. These artifacts provide versioning and reproducibility without an external service.

An AI Agent can write a local Python module exporting `create_policy(space, seed=0)`, which returns an object implementing the contract. `load_policy(path, space, seed=0)` loads trusted code and wraps it in output validation. Agent workflows should retain the split manifest, edit/train against train data, compare on validation, register a frozen version, and call the independent evaluator against held-out test data. Generated code execution is ordinary Python; adversarial-code process/container isolation is a future extension.
