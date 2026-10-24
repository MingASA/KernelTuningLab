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
- Runnable scale exercise with 139,968 candidates, 24 budgeted online trials and a 128×64 synthetic L3 problem; million-candidate regression cases prohibit eager enumeration.

## Validation boundary

The local host has no CUDA environment or GPU. Actual compilation, numerical behavior and performance of the four Triton operators on **H200, H800, RTX 4090 and RTX 5060 Ti 16GB remain to be verified on those physical machines**. CPU/Mock validation and source-backed specification profiles are not GPU validation. No CUDA packages are required or installed to run CPU paths.

The original presentation mentions previous RTX 4090/H200 measurements but supplies no raw measurement table in this repository. Those figures were read for project context and are not represented as newly reproduced results.

## Explicitly unfinished

- Concrete TileLang/CUTLASS backends; the plugin contract and factory registration exist.
- Fused affine LayerNorm, strided/broadcast BMM, paged/training attention and automatic GPTQ/AWQ checkpoint layout import.
- Clusters, warp specialization and arbitrary kernel code generation. Split-K and optional descriptor/TMA paths now exist in the proposal operators; physical validation is pending.
- Nsight Systems/Compute orchestration, concurrent-device workers, thermal/clock control, CUDA Graph/cold-cache modes and GPU driver fault reset. V2 experiments have process isolation and hard phase deadlines, including compiler calls; direct v1 Runner calls retain admission-only limits.
- Imputation or active acquisition for incomplete L3 matrices; unknown cells are preserved rather than predicted.
- Generalization of finite L3 dispatcher to unseen shape ranges and deployment binary-size measurement.
- Fully sandboxed generated policy execution. Current Python policies are trusted code with validated outputs.
- Automated LLM policy generation/evolution, paid API integrations or claims that a learned algorithm beats expert libraries.
- Remaining 25 original active Hub caches were catalogued but not downloaded/imported in this run. T4 export import, archived collections and specialized source adapters are not implemented.
- Streaming parsing of arbitrarily large source gzip JSON caches; current public importer loads one source file in memory. SQLite JSONL exchange itself is streamed.
- Physical operator/GPU test matrix, GPU fault isolation and detailed performance validation; CPU regressions cover core objectives, budgets, isolation, scale, source import, CLI and dispatch invariants.

## Next research steps

1. On each target GPU, run the same checked-in experiment specs with a small budget, verify tails and dtypes, archive complete environment/driver/compiler versions, and inspect raw distributions.
2. Collect comparable full or dense-enough tables with the same implementation/version and protocol; isolate software-stack changes before attributing differences to hardware.
3. Freeze algorithms on train/validation hardware and shapes, then report unseen-GPU/unseen-shape Replay outcomes with seeds, observed coverage, regret and measured search cost.
4. Add quantized GEMM as a separate implementation with its own legality constraints and correctness baseline, then evaluate L2/L3 against verified expert-library baselines.
5. Introduce learned hardware cost models and policy evolution only after preserving the existing split, evidence and budget contracts.

## Automatic asset integration

The public SDK now separates operator semantics, Kernel implementations and algorithms. Versioned files/modules/entry points, working scaffolds, Kernel-owned finite/conditional spaces, immutable algorithm views, executable static/synthetic/numerical acceptance, scoped correctness claims, complete execution plans and phase-deadline supervision are implemented. V2 configurations drive matching, validation, search and archival without new core registration or glue code. V1 research APIs remain available.

CPU examples include independent GEMM implementations, a two-step workspace pipeline, an additional AXPY operator, conditional spaces, scoped assertions and deliberate failure cases. CPU runs record real CPU measurements; CUDA remains deferred on this host. Continuous domains, distributed/asynchronous execution, a Web UI and the remaining production operators from the proposal are not part of this delivery. See [EXTENDING.md](EXTENDING.md) and [AUTO_INTEGRATION_PLAN.md](AUTO_INTEGRATION_PLAN.md).

The final automatic-integration regression suite passed **196 tests**. Installed-wheel workflows outside the repository and SDK-free `python3 -S` CPU workflows also passed. Evidence is in `artifacts/auto-integration-validation/summary.json`.

## Proposal operator coverage

All operator families explicitly named in the proposal are registered: W4/W3/W2A16 GEMM, standalone W4/W3/W2 dequantization, attention and MoE grouped GEMM. Each has independent references and real Python/Triton implementations; quantized GEMM also has timed multi-step variants. Production shapes, contextual spaces, workspace budgets, generated-case scopes, resource checks and public optimization references are documented in [PROPOSAL_OPERATORS.md](PROPOSAL_OPERATORS.md). Validation artifacts are under `artifacts/proposal-operators-validation/`. GPU compilation/numerics/performance remain unverified on this host.

