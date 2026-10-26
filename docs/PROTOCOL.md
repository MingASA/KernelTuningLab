# Experiment protocol v0.1

## Asset experiments (schema v2)

`ktlab check` and `ktlab run` add versioned asset resolution, executable acceptance and supervised phase deadlines to the research building blocks documented below. Full contributor details are in [EXTENDING.md](EXTENDING.md).

The operator provides seeded inputs, independent references, tolerances and boundary cases. The Kernel declares a finite space and an implementation returning one callable or an ordered `ExecutionPlan`. Correctness, warmup and timing invoke the complete initialization-plus-steps flow. Allocation/build/one-time setup are separated; CUDA full-flow timing synchronizes around a wall-clock sample and is not mixed with v1 single-launch CUDA Event timing.

Preflight, acceptance and each research trial have separate hard process deadlines. Acceptance has a whole-experiment configuration allowance and records only its sampled coverage. V2 research `max_evaluations` limits revealed candidate observations, including failures and cache hits; physical executions and cache reuse are reported separately. V1 direct Runner admission limits below remain unchanged.

Asset inputs are read-only and outputs must not alias their mutable storage. Input integrity is checked outside timing, and outputs are reset before individual timed calls to prevent reuse of stale results. The final actual timed output is rechecked without another invocation; a failed post-check invalidates that measurement's samples. Validation may sample multiple input seeds for the same configuration, including fixed kernels, within the same whole-experiment attempt budget. See [HARDENING.md](HARDENING.md) for precise scope and evidence limits.

The supervisor freezes the parsed experiment before worker launch, holds an exclusive OS output lock, preserves previous attempt metadata in `attempts/`, and cleans up its worker process group on termination or exit. The cumulative journal retains interrupted candidates as diagnostics while completed-cache lookup skips them so they can be retried. SIGINT/SIGTERM cancellations use exit codes 130/143 and are distinguished from hard timeouts (124).

Version/content/scope-bound submitter declarations can skip numerical comparisons, while output structure and finite values are still checked. Such observations remain `declared` with `correctness_verified=false`. Direct comparison records `verified` plus CPU/GPU evidence. Default verified reports, Replay and L3 exclude declared observations. Persisted matching correctness counterexamples revoke affected declarations and are restored from the journal before the next acceptance pass.

## Multi-case online research

`ktlab research` expands a strict `kind=research` configuration into original v2
jobs, serially across algorithms, Kernels, dtypes, seeds and budgets. Each job
keeps its own supervised preflight/acceptance/trials. A compatible shared v2
journal is retained for each named operator/Kernel/dtype group; the existing
cache key, not the group directory name, determines reuse. Repeating the command
revalidates the current source/environment and replays suggestions using that
cache. No previous completion flag bypasses validation.

Online trial traces record exact revealed record IDs, cache-hit flags, failed
observations, elapsed trial time, source costs and new physical costs. Source
costs are not charged again for cache hits. Compilation is part of physical
measurement wall time, not an additional additive cost. Unfinished attempts keep
known elapsed time; unknown compile costs remain unknown. Preflight, validation
and process-startup/job costs are separate from the candidate observation budget.

Reports read only current-attempt trial IDs, including explicitly revealed cached
records. They never scan an accumulated journal for an unobserved best. Curve and
performance cohorts preserve hardware/operator/implementation/Shape/dtype,
source/environment, measurement scope, repetitions/warmup, Graph batch size and
space/protocol identities. Missing/failed selectors remain in requested-seed
denominators; no 0 latency or global optimum is inferred. Algorithm source and
options are also kept distinct. No relative performance is pooled across
incompatible implementations or timing scopes. See [RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md).

## Dataset and protocol identity

L3's existing K constraint is a **config dictionary budget**. Selection serialization
now labels `budget_kind=config_count` and `config_count`; `variant_count` remains
a legacy alias. It does not constrain physical compiler specializations or binary
bytes. Actual Triton compiler cache keys and cubin/PTX hashes/sizes, when available,
are stored per compiled execution step. Shape/stride constexpr parameters can
produce different binaries for the same config, and a multi-step plan may contain
several compiled kernels. Unknown identities stay unknown. See
[RESEARCH_PHASE1.md](RESEARCH_PHASE1.md) for retention-policy-specific counts.

### Explicit v2 CUDA Graph timing

V2 specifications accept `timing_mode="cuda_graph"` and `cuda_graph_batch_size=128`
(1–4096). The default remains `full_flow`. Graph mode records
`measurement_mode="cuda_graph_events_full_flow_amortized"`: each Event sample
times one replay of a graph containing the declared number of complete
`initialize → ordered steps` GPU sequences, then divides by that count. Capture,
lazy-launch warmup and one untimed replay are excluded from latency, but charged
to the physical evaluation's wall time; capture cost is recorded separately in
the correctness execution metadata. Build/setup stay outside capture.

This mode is opt-in for capture-compatible CUDA plans using stable, preallocated
buffers. Host callbacks run during capture and are **not replayed**; host routing,
Python work and dynamic allocation cannot be represented as GPU-only performance.
An incompatible capture remains a compile failure. Output poisoning happens before
each replay outside timing, rather than between its internal calls. Initial
repeated-call correctness and final actual timed output/input integrity checks
still apply. Hot buffers and cache reuse are explicit; this is not a cold-cache
or end-to-end deployment measurement. Retain full-flow experiments separately.

The graph mode, batch size, runtime/source fingerprints and correctness protocol
participate in cache identity. L3 rejects mixed measurement modes, software/source
contexts, warmup/repetition counts and correctness protocols. Shape-dependent
configuration domains may differ; they do not change measurement scope.

On the physical 5060 Ti WSL environment, even a single-launch Event may contain
large CPU submission gaps. `scripts/check_gpu_timing.py` compares v1, v2 full flow,
independent Events, v2 graphs and Triton graphs on identical GEMM inputs/configs.
Never silently interpret Event or synchronized wall measurements as GPU-only cost.

Every record stores a complete hardware spec, operator implementation/version/dtype/parameters, shape, config, seed, experiment ID and environment. GPU data includes Python/Torch/Triton/CUDA/driver versions, the actual device name and kernel source digest. A datasheet profile is a requested hardware identity; real execution probes the device and uses probed resource limits. Profiles alone are not execution evidence.

`ExperimentSpec` JSON is portable across machines. Save a copy beside every journal. Built-in schemas cover contiguous GEMM, independent batched GEMM, row Softmax and non-affine row LayerNorm. Hardware-specific TMA/cluster/warp-specialization knobs are not attached to these implementations.

## Measurement phases and units

The following single-launch phase description applies to the original v1 adapters; the asset execution mode above uses the same lifecycle with its complete-flow measurement scope.

1. Validate schema/domain/operator/hardware legality. Save rejected candidates as `invalid_config`.
2. Prepare fixed seeded input tensors and the correctness reference. Preparation time is recorded separately in metadata.
3. Call the adapter JIT compilation step. Record `compile_time_ms`, including compiler-cache lookup overhead if the compiler cache is warm. Save compilation exceptions as `compile_error`.
4. Execute the compiled kernel and compare with the operator's baseline. Save tolerance, maximum error, correctness mode and `correctness_error` separately.
5. Run configured warmup iterations, synchronize and collect individual runtime samples. CUDA Event samples use milliseconds. The built-in v1 mode times one hot kernel launch per sample; host launch gaps may be included. Cache flushing, isolated clocks and cold-input benchmarking remain future modes; explicit v2 CUDA Graph replay is specified above.
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
