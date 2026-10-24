# Experiment protocol v0.1

## Asset experiments (schema v2)

`ktlab check` and `ktlab run` add versioned asset resolution, executable acceptance and supervised phase deadlines to the research building blocks documented below. Full contributor details are in [EXTENDING.md](EXTENDING.md).

The operator provides seeded inputs, independent references, tolerances and boundary cases. The Kernel declares a finite space and an implementation returning one callable or an ordered `ExecutionPlan`. Correctness, warmup and timing invoke the complete initialization-plus-steps flow. Allocation/build/one-time setup are separated; CUDA full-flow timing synchronizes around a wall-clock sample and is not mixed with v1 single-launch CUDA Event timing.

Preflight, acceptance and each research trial have separate hard process deadlines. Acceptance has a whole-experiment configuration allowance and records only its sampled coverage. V2 research `max_evaluations` limits revealed candidate observations, including failures and cache hits; physical executions and cache reuse are reported separately. V1 direct Runner admission limits below remain unchanged.

Asset inputs are read-only and outputs must not alias their mutable storage. Input integrity is checked outside timing, and outputs are reset before individual timed calls to prevent reuse of stale results. The final actual timed output is rechecked without another invocation; a failed post-check invalidates that measurement's samples. Validation may sample multiple input seeds for the same configuration, including fixed kernels, within the same whole-experiment attempt budget. See [HARDENING.md](HARDENING.md) for precise scope and evidence limits.

The supervisor freezes the parsed experiment before worker launch, holds an exclusive OS output lock, preserves previous attempt metadata in `attempts/`, and cleans up its worker process group on termination or exit. The cumulative journal retains interrupted candidates as diagnostics while completed-cache lookup skips them so they can be retried. SIGINT/SIGTERM cancellations use exit codes 130/143 and are distinguished from hard timeouts (124).

Version/content/scope-bound submitter declarations can skip numerical comparisons, while output structure and finite values are still checked. Such observations remain `declared` with `correctness_verified=false`. Direct comparison records `verified` plus CPU/GPU evidence. Default verified reports, Replay and L3 exclude declared observations. Persisted matching correctness counterexamples revoke affected declarations and are restored from the journal before the next acceptance pass.

## Dataset and protocol identity

Every record stores a complete hardware spec, operator implementation/version/dtype/parameters, shape, config, seed, experiment ID and environment. GPU data includes Python/Torch/Triton/CUDA/driver versions, the actual device name and kernel source digest. A datasheet profile is a requested hardware identity; real execution probes the device and uses probed resource limits. Profiles alone are not execution evidence.

`ExperimentSpec` JSON is portable across machines. Save a copy beside every journal. Built-in schemas cover contiguous GEMM, independent batched GEMM, row Softmax and non-affine row LayerNorm. Hardware-specific TMA/cluster/warp-specialization knobs are not attached to these implementations.

## Measurement phases and units

The following single-launch phase description applies to the original v1 adapters; the asset execution mode above uses the same lifecycle with its complete-flow measurement scope.

1. Validate schema/domain/operator/hardware legality. Save rejected candidates as `invalid_config`.
2. Prepare fixed seeded input tensors and the correctness reference. Preparation time is recorded separately in metadata.
3. Call the adapter JIT compilation step. Record `compile_time_ms`, including compiler-cache lookup overhead if the compiler cache is warm. Save compilation exceptions as `compile_error`.
4. Execute the compiled kernel and compare with the operator's baseline. Save tolerance, maximum error, correctness mode and `correctness_error` separately.
5. Run configured warmup iterations, synchronize and collect individual runtime samples. CUDA Event samples use milliseconds. The built-in mode times one hot kernel launch per sample; host launch gaps may be included. Cache flushing, isolated clocks, CUDA Graph replay and cold-input benchmarking are future modes.
6. Store raw samples plus P10/P50/P90 analysis, repetitions, warmup, mode, failure details and total observed `wall_time_s`. Slow correct configurations are successful measurements. NaN/infinite/nonpositive samples are persisted as runtime failures.

GPU correctness references use FP32 Torch operations and cast to the output dtype; GEMM disables TF32 for the reference. Tolerances are adapter protocol parameters and part of cache identity. Mock uses a deterministic synthetic correctness flag and explicitly records that no GPU correctness calculation occurred.

## Budget semantics

`SearchBudget(max_evaluations, max_wall_time_s)` is an admission bound. Every newly attempted configuration consumes one evaluation, including invalid configuration, compile failure, correctness failure and runtime failure. Compatible cache hits do not create a new physical evaluation. Time covers preparation, compilation, correctness, warmup and measurement. An already running compiler/kernel is not preempted; any overrun is reported. A zero budget admits no requests, including cached ones.

`benchmark` collection mode uses one budget shared across all shapes, visited in round-robin order. `benchmark --searcher ...` runs adaptive tuning independently for each shape, and applies the stated budget **per shape**, as recorded in `tuning.json` and CLI output. Resume retains compatible observations but each invocation sets its own allowance for additional physical measurements.

Offline Replay consumes one logical evaluation per revealed/failed/missing request. It separately records known source measurement cost, unknown-cost coverage and actual CPU replay wall time. A source wall-time cost is not imputed from kernel latency or treated as a real new physical measurement. Time-bounded replay uses CPU elapsed plus known source cost and reports missing cost explicitly. Configurations with missing source data generate a `missing` trace, never a fabricated measurement record.

An explicitly failed correctness check remains an observed failure even though its `correctness_verified` flag is false. Only unverified successes and incomplete parent-terminated attempts are excluded from the default completed Replay oracle; failure costs must not become missing data. Search algorithms receive read-only spaces and bounded batches. Transfer training uses verified labels from compatible implementations/parameters/content and normalizes each source measurement protocol separately.

## Isolation and seeds

Split cases before training policies. Default Demo holds out an entire fictitious GPU; `--holdout shape` holds out shapes across all GPUs. Save `split_manifest.json`, separate `splits/*.jsonl`, random seeds and immutable policy versions. Validation supports development; test is reserved for frozen policy scoring. No automatic selection of global algorithm hyperparameters from test outcomes is implemented.

Search histories contain only requested records and identities of already attempted missing configurations. Oracle minima are evaluator-only. Regret is `(best found latency / known successful observed optimum) - 1`; partial tables are explicitly a reference to the best **known** configuration, not a global optimum. For a 95% throughput target, use latency ratio `1/0.95`; the default ratio `1.05` means latency within 5% of the known best.

## L3 objective

For normalized workload weights `w`, choose at most `K` configs and minimize `sum_i w_i min_j L_ij`. Report relative loss against the weighted unrestricted per-shape observed best: the ratio of weighted latencies, not the weighted mean of per-shape ratios. Configurations lacking an observed successful cell cannot be dispatched for that shape. Partial tables retain `None`; predicted costs cannot enter observed matrices.

Oracle exhaustively solves small tables or proves optimality by matching an unrestricted lower bound. It raises an explicit combination-limit error instead of claiming exactness beyond its limit. Greedy, local swap and bucket baselines share the objective. Dispatcher creation verifies membership, budget, observed assignment, context and current legality. Unknown shapes are rejected by the finite lookup dispatcher; extrapolation requires an independently evaluated policy.

## Imported and synthetic data

`provenance` is `measured`, `imported`, or `mock`. Predictions are not `MeasurementRecord`s. Imported timings with no correctness evidence use `correctness_unknown` and cannot populate validated tuning or L3 performance tables. Source verification timing alone is not proof of a correctness check. Promoting imported data to verified requires explicit evidence supplied by the caller and saved in metadata. See [PUBLIC_DATA.md](PUBLIC_DATA.md).

All mock report figures are visibly labelled synthetic. Mock runs demonstrate software behavior only; no accuracy, speedup or cross-GPU transfer claim follows from their results.
