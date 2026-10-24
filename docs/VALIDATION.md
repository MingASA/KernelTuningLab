# v0.1 validation record

The final execution pass was delegated to a Luna experiment agent. Full output is stored in files rather than embedded in research discussions. No CUDA runtime or GPU was available; no physical GPU result is claimed.

## Latest execution and evidence hardening

Luna ran the final source and regression additions: **288 tests passed in 33.72s**, up from a baseline of 245. The additional cases exercise stale timed outputs, input mutation/aliasing, post-measurement failures, declared evidence, fixed-configuration multiple-seed validation, failed Replay observations, Transfer isolation, bounded proposals, schema-safe SQLite opening, interrupted cache recovery, output ownership, cancellation and frozen experiment snapshots.

The CPU proposal matrix completed **17 semantic cases across 8 operator families**, with 216 new research evaluations, 432 revealed observations and 136 validation attempts. Repeating the two-kernel/two-algorithm CPU GEMM example completed 8 trials with **0 new research evaluations and 24 cache hits** on the second run, retaining the previous attempt metadata. These are real CPU execution checks, not GPU performance results.

A bounded synthetic SQLite exercise inserted, exported and imported exactly 5,000 records; reimport added zero and counted 5,000 duplicates. Local operation costs were approximately 1.06s for insertion, 0.39s for export and 2.41s for import plus duplicate import. They measure journal operations on this host, not GPU kernel throughput. A rebuilt wheel imported the new lifecycle/context modules and completed a CPU asset check from outside the checkout.

The CUDA attention matrix returned exit code 3 with both requested dtypes **deferred**, and no GPU evaluations. The machine-readable commands, return codes, artifact references and source/test hashes are in `artifacts/hardening-validation/summary.json`. Detailed contracts and remaining boundaries are in [HARDENING.md](HARDENING.md).

## Initial v0.1 checks

| Check | Observed result | Evidence |
| --- | --- | --- |
| Full CPU test suite | **129 passed** | `artifacts/final_validation/pytest.txt` |
| Four-operator Mock workflow | 2304 records, 2880 search trials, 64 policy evaluations, 240 L3 results | `artifacts/final-demo/run_summary.json` |
| Failure distinctions | Invalid config, compile, correctness and runtime failures retained separately from slow success | Mock journal and core regressions |
| Isolation | Hidden-label perturbation, mutable-history snapshots, source-protocol separation, case/shape/GPU holdout and Top-1 policy scoring | Replay/Policy and integration tests |
| Large configuration spaces | Million-candidate tests prohibit eager product enumeration; sampling, six searchers and partial Replay retain budgets | `tests/test_scale_contracts.py` |
| Runnable scale exercise | 139968 candidates, 24 online trials, 768 new evaluations; 128×64 synthetic matrix; 15 L3 selections and 3 small exact references | `artifacts/scale-final/summary.json` |
| L3 correctness | Weighted objective, independent small brute force, partial coverage, swap improvement and legal dispatch | `tests/test_l3.py` and scale tests |
| Storage robustness | Transaction rollback, malformed JSONL tails, duplicate imports, atomic export failure preservation, read-only missing-path failures | Core/integration tests |
| Complete upstream imports | 4362 A100 convolution + 11130 A4000 dedispersion = **15492** records | `artifacts/public-hub/measurements.sqlite` and source manifests |
| Public-data evidence boundary | 15331 correctness-unknown, 6 compile errors, 155 runtime errors; **0 verified successes** | Public import summary and report |
| Packaging | Wheel built; backend/plugin/scale/Hub modules, offline sample and upstream MIT notice checked inside archive | `artifacts/wheels/` and final validation summary |

## RTX 5060 Ti 16GB follow-up

The follow-up adds the `5060ti16g` profile, physical 16GB/8GB discrimination, native Blackwell runtime checks and a ten-stage local correctness gate before Hopper validation. Contract regressions cover all four operators, shared-memory limits, runtime rejection before allocation, exact shape coverage, physical/source/runtime identity, malformed artifacts and failed retries clearing stale success markers.

The final follow-up CPU suite passed **158 tests**. All four strict Mock smokes returned eight successful records covering both requested shapes; the default probe and shell syntax check also passed. The Luna agent's complete outcomes, return codes and rebuilt wheel digest are recorded in `artifacts/rtx5060ti-validation/summary.json`. These Mock runs remain synthetic. GPU preflight on the development host stops at missing Torch/Triton dependencies; no GPU packages were installed and no physical 5060 Ti run was performed. The executable procedure for the user's machine is in [RTX5060TI.md](RTX5060TI.md).

## Reproduce

For the automatic asset integration follow-up, the Luna agent ran the complete regression suite: **196 tests passed**. SDK-only external plugins, fixed kernels without fake knobs, million-point bounded spaces, correctness evidence, budgets, timeouts and the four operator templates are covered. Wheel installation in a fresh environment outside the repository passed SDK discovery, scaffold, check and run. The same CPU workflow also passed with `python3 -S`. Logs, return codes and the package digest are under `artifacts/auto-integration-validation/summary.json`.

```bash
ktlab run --spec examples/extensions/cpu_gemm.json --output artifacts/extensions/cpu
ktlab run --spec examples/extensions/custom_operator.json --output artifacts/extensions/axpy
ktlab run --spec examples/extensions/conditional_space.json --output artifacts/extensions/conditional
ktlab run --spec examples/extensions/scoped_claim.json --output artifacts/extensions/claims
ktlab scaffold kernel --output artifacts/local-kernel
ktlab run --spec artifacts/local-kernel/experiment.json --output artifacts/local-kernel-run
```

Observed follow-up examples include eight CPU GEMM trials, twelve new research evaluations, eight validation evaluations and twelve cross-algorithm cache hits. The scoped-claim example retains both declared and directly reference-checked results; its M=5 cases are outside the declaration and are checked. Deliberate wrong-output, illegal-algorithm, widened-space and unsupported-capability examples return failure; CUDA examples return deferred. These are CPU infrastructure/implementation checks, not physical GPU evidence. The final regression count and package digest are recorded in the follow-up summary.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m kernel_tuning_lab demo --output artifacts/demo
.venv/bin/python -m kernel_tuning_lab scale-demo --output artifacts/scale
.venv/bin/python -m kernel_tuning_lab hub list
.venv/bin/python -m kernel_tuning_lab hub import --kernel convolution_milo --gpu A100 --database artifacts/hub.sqlite
.venv/bin/python -m kernel_tuning_lab hub import --kernel dedispersion_milo --gpu A4000 --database artifacts/hub.sqlite
```

The Hub catalogue is pinned and downloads are verified by byte count, Git blob SHA and SHA-256 sidecars. Previously verified local files can be reused. Kernel Tuner correctness evidence is separate from source integrity, and default unknown entries cannot become validated oracles.

## Boundaries

- All synthetic latency/quality figures are explicitly Mock. The large-space global optimum was not measured, so its global regret is unknown.
- Public records are upstream observations; this host did not rerun their kernels or independently verify outputs.
- H200 SXM, H800 SXM5, RTX 4090 and RTX 5060 Ti 16GB profiles support planning and device checks. Physical compilation, accuracy, runtime distributions and resource behavior on those models remain unverified.
- The GPU smoke script checks four operators and supported dtypes across ten stages, saves failure diagnostics and requires every tested config and requested shape to succeed. It clears stale completion markers before a retry and verifies physical/source/runtime identity in the aggregate. Run it on the matching CUDA host.
- Tests establish the current contract boundaries and sampled scales, not arbitrary cluster concurrency, malicious-code isolation or unlimited-memory imports. Explicit future work is listed in [STATUS.md](STATUS.md).

The human-readable outputs are indexed in [examples/README.md](../examples/README.md). Command-level logs, return codes and the final digest live in `artifacts/final_validation/summary.json`.
