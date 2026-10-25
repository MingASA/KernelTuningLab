# v0.1 delivery status

## Implemented

- CPU-only core package, unified versioned models, explicit units and a per-operator config space.
- SQLite measurement journal, normalized dimension tables, transactional bulk/JSONL import, atomic streaming export, read-only analysis and full-record deduplication.
- Runner with separate preparation/JIT/runtime phases, correctness, raw samples, warmup, environment/source metadata, cache resume and failure logs.
- Seeded Mock implementations for GEMM, batched GEMM, Softmax and LayerNorm.
- Lazy Triton implementations for these four operators; named H200 SXM / H800 SXM / RTX 4090 / RTX 5060 Ti 16GB hardware profiles and actual device probing. Consumer Blackwell has an explicit native runtime check.
- Ten-stage strict GPU correctness gate with per-run records, source hashes and device-specific summary; RTX 5060 Ti 16GB is the lightweight pre-Hopper entry point.
- Random, Grid, Heuristic, Local, distance-surrogate and historical Transfer search; online tuning and offline reveal-only Replay. Large spaces use bounded candidate pools and index sampling rather than a full Cartesian table.
- Case/shape/GPU holdouts, immutable Policy source snapshots, output legality checks, fallback and independent scoring without a paid model API.
- L3 exact oracle, greedy, local swap, bucket selection, K-loss curves and finite validated dispatch.
- Markdown/SVG and optional PNG reports: heatmaps, regret-budget, transfer, matched sensitivity, K-loss, failure and coverage statistics.
- Kernel Tuner import and full pinned Benchmark Hub catalogue discovery, hash-verified downloads and whole-file normalization. Two complete caches totaling 15,492 upstream records were actually imported; a 24-record excerpt and MIT notice are bundled. Unknown correctness remains unknown.
- CLI, portable experiment examples, runnable Mock Demo and core regressions.
- One-command online research comparisons from an SDK plugin and a matrix configuration, reusing supervised v2 jobs/journals across operators, implementations, dtypes, seeds and budgets. Automatic scope-isolated search/performance/cost/failure reports, exact revealed-record traces and resume are documented in [RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md).
- Runnable scale exercise with 139,968 candidates, 24 budgeted online trials and a 128×64 synthetic L3 problem; million-candidate regression cases prohibit eager enumeration.

## Validation boundary

The 2026-10-25 physical RTX 5060 Ti 16GB WSL run passed the ten-stage strict core
gate (80 records) and the repaired default proposal suite (16 jobs, 576 research
observations and 140 acceptance attempts). Actual compilation, numerical behavior
and timing were exercised. First failures and sampled coverage are retained in
[5060TI_RESULTS.md](5060TI_RESULTS.md). **H200, H800 and RTX 4090 still require
independent physical validation.** CPU paths continue to require no CUDA packages.

The generic research workflow was separately exercised on the same physical GPU:
72/72 jobs, 144 trials and 432 verified observations across GEMM and fused/pipeline
W4A16, with an external SDK algorithm versus Random/Heuristic, two dtypes, two
seeds and two budgets. Interruption recovery, a fully cached replay, deliberately
failed/inapplicable jobs, and separate Graph/full-call reporting also ran.
This is workflow acceptance on a four-config space, not a claim of algorithm
superiority or exhaustive operator coverage; compact evidence is in
[research-workflow-validation.json](data/research-workflow-validation.json).

The original presentation mentions previous RTX 4090/H200 measurements but supplies no raw measurement table in this repository. Those figures were read for project context and are not represented as newly reproduced results.

## Explicitly unfinished

- Concrete TileLang/CUTLASS backends; the plugin contract and factory registration exist.
- Fused affine LayerNorm, strided/broadcast BMM, paged/training attention and automatic GPTQ/AWQ checkpoint layout import.
- Clusters, warp specialization and arbitrary kernel code generation. Split-K and descriptor/TMA loads are sampled on 5060 Ti; Hopper paths still need physical validation.
- Nsight Systems/Compute orchestration, concurrent-device workers, thermal/clock control, cold-cache modes and GPU driver fault reset. V2 now has explicit CUDA Graph full-plan timing, process isolation and hard phase deadlines, including compiler calls; direct v1 Runner calls retain admission-only limits.
- Imputation or active acquisition for incomplete L3 matrices; unknown cells are preserved rather than predicted.
- Generalization of finite L3 dispatcher to unseen shape ranges and deployment binary-size measurement.
- Fully sandboxed generated policy execution. Current Python policies are trusted code with validated outputs.
- Automated LLM policy generation/evolution, paid API integrations or claims that a learned algorithm beats expert libraries.
- Remaining 25 original active Hub caches were catalogued but not downloaded/imported in this run. T4 export import, archived collections and specialized source adapters are not implemented.
- Streaming parsing of arbitrarily large source gzip JSON caches; current public importer loads one source file in memory. SQLite JSONL exchange itself is streamed.
- Broad cross-GPU and production-scale matrix, GPU fault isolation and isolated clock/cold-cache performance validation; the 5060 Ti sampled matrix is now measured. CPU regressions cover core objectives, budgets, isolation, scale, source import, CLI and dispatch invariants.

## Next research steps

1. On each target GPU, run the same checked-in experiment specs with a small budget, verify tails and dtypes, archive complete environment/driver/compiler versions, and inspect raw distributions.
2. Collect comparable full or dense-enough tables with the same implementation/version and protocol; isolate software-stack changes before attributing differences to hardware.
3. Freeze algorithms on train/validation hardware and shapes, then report unseen-GPU/unseen-shape Replay outcomes with seeds, observed coverage, regret and measured search cost.
4. Validate the added quantized GEMM, attention and MoE implementations on physical hardware, then evaluate L2/L3 against comparable verified expert-library baselines.
5. Introduce learned hardware cost models and policy evolution only after preserving the existing split, evidence and budget contracts.

## Automatic asset integration

The public SDK now separates operator semantics, Kernel implementations and algorithms. Versioned files/modules/entry points, working scaffolds, Kernel-owned finite/conditional spaces, immutable algorithm views, executable static/synthetic/numerical acceptance, scoped correctness claims, complete execution plans and phase-deadline supervision are implemented. V2 configurations drive matching, validation, search and archival without new core registration or glue code. V1 research APIs remain available.

CPU examples include independent GEMM implementations, a two-step workspace pipeline, an additional AXPY operator, conditional spaces, scoped assertions and deliberate failure cases. CPU runs record real CPU measurements; physical 5060 Ti CUDA evidence is now archived separately. Continuous domains, distributed/asynchronous execution and a Web UI are not part of this delivery. The proposal operator additions are listed below. See [EXTENDING.md](EXTENDING.md) and [AUTO_INTEGRATION_PLAN.md](AUTO_INTEGRATION_PLAN.md).

The final automatic-integration regression suite passed **196 tests**. Installed-wheel workflows outside the repository and SDK-free `python3 -S` CPU workflows also passed. Evidence is in `artifacts/auto-integration-validation/summary.json`.

## Proposal operator coverage

All operator families explicitly named in the proposal are registered: W4/W3/W2A16 GEMM, standalone W4/W3/W2 dequantization, attention and MoE grouped GEMM. Each has independent references and real Python/Triton implementations; quantized GEMM also has timed multi-step variants. Production shapes, contextual spaces, workspace budgets, generated-case scopes, resource checks and public optimization references are documented in [PROPOSAL_OPERATORS.md](PROPOSAL_OPERATORS.md). Historical CPU artifacts remain under `artifacts/proposal-operators-validation/`; physical 5060 Ti artifacts are under `artifacts/5060ti/`. GPU sampling does not validate all production Shapes or all configurations.

## Execution and evidence hardening

Input integrity, output alias checks, timed-output reset and post-measurement correctness invalidate stale or corrupted results. Fixed kernels can receive multiple seeded checks within the validation budget. Replay preserves failed observations and costs, default Transfer excludes unverified labels and incompatible implementations, and candidate streams remain bounded. Output ownership, frozen configuration snapshots, retained attempt metadata, cancellation/worker-exit cleanup and retryable interrupted records strengthen recovery. SQLite rejects unsupported schemas before mutation and indexes dense case split checks. Contracts and remaining production limits are in [HARDENING.md](HARDENING.md); local regression evidence is in `artifacts/hardening-validation/`.

