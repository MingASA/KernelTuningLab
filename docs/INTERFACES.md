# Public interfaces v0.1

For new independent Kernels, operators and algorithms, start with the [public asset SDK and v2 workflow](EXTENDING.md). The original research interfaces remain documented below.

Core imports require only the Python standard library. Every duration states its unit; versioned measurement JSON uses `schema_version=1`. Canonical content IDs do not depend on dictionary key order.

## Data models

| Model | Principal fields |
| --- | --- |
| `HardwareSpec` | name, architecture, SM count, per-block shared memory bytes, registers per SM, theoretical TFLOPS, GB/s bandwidth, L2/memory bytes, warp/thread limits, features and source metadata |
| `OperatorSpec` | name, kind, implementation/version, dtype, operator parameters, correctness baseline and metadata |
| `ShapeSpec` | positive integer `dims`, such as M/N/K/Batch or M/N |
| `KernelConfig` | extensible JSON `params`; each implementation defines the supported schema |
| `MeasurementRecord` | hardware/operator/shape/config, status, compile time, raw runtime samples, total wall time, protocol, environment, failure reason, provenance, split, seed and experiment ID |
| `WorkloadDistribution` | unique shapes, nonnegative finite weights and name; `.normalized_weights` safely normalizes even very large weights |
| `SearchBudget` | nonnegative `max_evaluations`, optional `max_wall_time_s`; `.exhausted(evaluations, elapsed_s)` and `.remaining(...)` |

Specifications/configurations expose deterministic `.id` properties. `ShapeSpec.get` and `KernelConfig.get` provide parameter access. Model dictionaries should be treated as immutable after construction; evaluation interfaces snapshot revealed observations before passing them to user code.

`RecordStatus` values are `SUCCESS`, `INVALID_CONFIG`, `COMPILE_ERROR`, `CORRECTNESS_ERROR`, `RUNTIME_ERROR`, and `CORRECTNESS_UNKNOWN`. Slow correct results remain successful. Imported samples without correctness evidence remain unknown and `.successful` is false. Failed/unknown records retain a reason. Successful runtime samples must be finite and positive.

`MeasurementRecord` exposes `.successful`, `.median_ms`, `.quantile_ms(q)`, `.case_id`, `.cache_key`, `.id`, `.to_dict()` and `MeasurementRecord.from_dict(data)`. Compile/runtime sample units are milliseconds; total `wall_time_s` is seconds. Provenance is `mock`, `measured`, or `imported`. Predictions cannot be measurement records; missing measurements have no record or a `None` matrix cell.

## Configuration spaces

```python
ConfigSpace(domains, constraints=(), name="custom")
space.validate(config, hardware, operator, shape)  # ValidationResult(valid, reasons)
space.iter_configs()                            # lazy Cartesian traversal
space.legal_configs(hardware, operator, shape)   # exhaustive small-space list
space.config_at(index)                          # O(number of dimensions)
space.sample_configs(random.Random(seed), count, exclude_ids=())
```

`cardinality` gives the Cartesian size and `fingerprint` provides compact protocol identity. Seeded sampling avoids product enumeration; bounded rejection may yield fewer than requested near exhaustion. Do not mutate nested domain values in place; define a new space for a changed schema. A constraint returns `None` or a rejection reason. Domains retain their parameter/value order, and `ExperimentSpec.save/load` preserves Grid traversal order.

`operator_spec(name, backend="mock", dtype="float16")` and `operator_space(name, small=False, shapes=())` cover `gemm`, `batched_gemm`, `softmax`, `layernorm`. GEMM spaces expose BLOCK_M/N/K, num_warps, num_stages. Row reductions use BLOCK_SIZE, num_warps, num_stages. Unsupported dtype/layout/affine/axis settings are rejected explicitly.

## Hardware and portable experiments

`get_hardware_profile(name)`, `list_hardware_profiles()`, and `validate_hardware_profile(profile, actual)` provide sourced H200 SXM, H800 SXM5, RTX 4090 and RTX 5060 Ti 16GB planning profiles. They are specification data. Real adapters probe physical identity/resources before measurement; see [HARDWARE.md](HARDWARE.md). `validate_triton_runtime` checks the declared native sm_12x baseline without importing GPU libraries.

`ExperimentSpec` stores backend/operator/shapes/domains/hardware/budget/seed/warmup/repetitions/split and supports `.save(path)`, `.load(path)`, `.config_space()`. GPU examples leave hardware unset for physical probing. CLI profile checks do not substitute datasheet identity for an actual probe.

## Storage and execution

```python
with ExperimentStore(path, read_only=False) as store:
    store.add(record)                  # bool: newly inserted
    store.add_many(records)            # atomic batch, returns inserted count
    store.records(**filters)           # list
    store.iter_records(**filters)      # streaming iterator
    store.get_cached(cache_key)
    store.import_jsonl(path)           # atomic; read/added/duplicates
    store.export_jsonl(path)           # streaming, atomic replace
```

Filters include hardware_id, operator_id, shape_id, config_id, split, provenance, experiment_id and status. Read-only opens reject missing/invalid journals and writes. Identical record content deduplicates; independent repetitions remain distinct. A case cannot span splits within one experiment. Malformed JSONL tails and split conflicts roll back the import. Exports cannot overwrite the journal.

`BenchmarkRunner(adapter, store=None, warmup=2, repetitions=7, seed=0, experiment_id="default", failure_log_path=None)` provides:

```python
runner.measure(hardware, operator, shape, config, space, split="train", resume=True)
runner.run(hardware, operator, shapes, space, budget=None, split="train", resume=True)
```

Every completed valid/invalid attempt is journalled. Compatible cached results avoid physical evaluation; current legality is rechecked before cache reuse. Preparation, compilation, correctness, warmup and measurement are separate adapter phases. Global collection visits shapes round-robin. Budgets are admission bounds; already running compilation is not preempted. `last_run_stats` reports costs/cache/overrun semantics. A zero budget admits no requests.

CLI `--require-all-success` is the strict numerical gate: every attempted config and requested shape must succeed. `run_records.jsonl` snapshots that invocation, while `measurements.jsonl` exports full journal history. The summary includes `requested_shape_ids` for exact coverage checks. `probe --gpu --hardware ...` performs explicit physical preflight. `gpu_validation.summarize_gpu_smoke(output_dir)` checks all ten physical stages, device capacities, runtime identity and shared adapter/kernel sources, then writes `validation_summary.json`; see [RTX5060TI.md](RTX5060TI.md).

`KernelAdapter` and `register_adapter(name, factory, replace=False)` are the backend extension boundary. `ktlab --plugin MODULE_OR_PATH ...` loads an explicitly requested trusted module. See [PLUGINS.md](PLUGINS.md) for a complete callback table and a runnable example.

## Search and evaluation

```python
searcher = make_searcher(name, seed=0, training_records=(),
                         candidate_pool_size=512, exhaustive_threshold=4096)
searcher.suggest(hardware, operator, shape, space, history, budget, batch_size=1)
tune(runner, hardware, operator, shape, space, searcher, budget,
     split="train", resume=True)
run_replay(records, searcher, space, budget=SearchBudget(), seed=0, target_ratio=1.05)
compare_searchers(records, space, budgets=(8,16,32,64,128), seeds=(0,1,2), ...)
```

Names are random/grid/heuristic/local/surrogate/transfer. Custom Searcher objects can be passed directly to `tune`/`run_replay`. SearchHistory contains only revealed records and attempted missing-config IDs. Deep snapshots prevent algorithm mutations from changing stored observations. The oracle source table and per-step regret never enter search history.

Large-space algorithms use bounded candidate pools/index traversal, with explicit approximation markers. Exact legal-space coverage can remain unknown; raw observed fraction is reported separately. Offline missing/failed attempts consume logical budget. Source measurement cost and actual CPU replay time are separate; unavailable source cost is never invented. `TuningResult` recommends only successful revealed configurations, counting new attempts and cache hits separately.

Replay accepts a single case and compatible measurement context. Comparison splits protocol/environment/source cohorts, and summaries retain hardware/operator/provenance/context identity. Transfer training may use other hardware but only train records and no target case. Contexts include imported file identity when available. Regret references the best known successful observation, never an unmeasured global optimum.

## Policies and split isolation

`split_records(records, seed=0, train_fraction=.6, validation_fraction=.2, holdout="case")` returns train/validation/test tuples and a manifest. Holdout may group by case, shape or hardware; a case's configurations never cross sets. `require_training_records` rejects validation/test and held-out identity overlap.

Policies implement `select_config(hardware, op, shape)` and/or `search_config(hardware, op, shape, history, budget)`. Search returns `PolicyDecision(candidates, recommendation)`. Baselines are `HeuristicPolicy`, `TransferPolicy`, `SearchPolicy` and `FunctionPolicy`; `SafePolicy` validates outputs and supplies a legal fallback.

`evaluate_policy(policy, records, space, mode="search", budget=SearchBudget(), seed=0)` is independent of generation. Select mode scores ranked Top-1 with zero deployment measurements; remaining predictions are retained without oracle-based deployment selection. Search recommendations are supported by revealed observations. `evaluate_holdout` validates unseen shape/GPU use. `PolicyRepository.register/log_evaluation` records immutable versions, source snapshots and experiment logs. `load_policy` loads a trusted `create_policy(space, seed)` module; it is output validation, not a malicious-code sandbox.

## L3 and reporting

`PerformanceMatrix(workload, configs, latencies_ms, provenance="unspecified")` stores shape rows/config columns and `None` missing cells. `.from_records(...)` filters/aggregates successful observations while preserving context and provenance. Predictions are not accepted as observed cells.

`oracle`, `greedy`, `greedy_local_swap`, `bucket_baseline`, `evaluate_selection` and `k_loss_curve` share the weighted expected-latency objective. Oracle either proves exactness or raises `OracleLimitError`; it never silently substitutes an approximation. SelectionResult exposes selected/global assignment indices, expected/conditional latency, coverage, loss and `.to_dict()`.

`ShapeDispatcher.from_result(matrix, result, hardware=..., operator=..., space=...)` validates finite variant membership and measured legal assignments; `.select_config(shape)` uses an exact table and rejects unknown shapes. Missing matrices have explicitly conditional scores, never fabricated predictions. See [L3.md](L3.md).

`generate_report(records, output_dir, replay_results=(), l3_results=())` returns a Markdown path and writes SVG/optional PNG plus audit JSON. Comparable hardware/operator/protocol/source cohorts have separate figures. Insufficient/unverified data is labelled rather than scored. Benchmark Hub discovery/download/import and the bundled real fixture are documented in [PUBLIC_DATA.md](PUBLIC_DATA.md).
## Public asset integration (experiment schema v2)

Contributor imports come from `kernel_tuning_lab.sdk`. `OperatorDefinition`, `KernelDefinition` and `SearchAlgorithmDefinition` register independently through `AssetCatalog`; references are resolved by exact version and content fingerprint. `ParameterSpec`/`SpaceDefinition` belong to the Kernel, and experiments only narrow domains. `SpaceView` exposes read-only algorithm inputs while the platform retains authoritative validation and budgets.

`ExecutionContext`, `ExecutionStep` and `ExecutionPlan` cover single-call and composed implementations. Shared input/reference/tolerance services live in operator assets, not in each Kernel. `CorrectnessClaim` is an explicitly scoped assertion, distinct from platform verification. `MeasurementRecord.correctness_verified` prevents declared-only observations from entering default verified Replay/L3/analysis.

`AssetExperiment` is the strict v2 configuration schema; `ktlab check` runs executable acceptance and `ktlab run` continues to budgeted search. `ktlab scaffold` creates runnable SDK-only templates. `asset_runner.execute_asset_experiment` provides supervised phase deadlines and persistent failure/timeout evidence. Direct calls to lower-level runner APIs retain their documented admission-time budget semantics.

The complete contributor guide, fields, examples, exit statuses and archive layout are in [EXTENDING.md](EXTENDING.md). The interfaces below document the compatible v1 research building blocks as well.

