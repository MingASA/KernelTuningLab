# Reproducible examples

`experiments/` contains portable Mock and CUDA specifications for all four operators. GPU specs leave physical hardware unset so the actual device is probed at execution; pass `--hardware H200`, `H800`, `4090`, or `5060ti16g` to check the requested profile. `hardware/` holds sourced vendor specification snapshots, not measurement records.

```bash
python3 -m kernel_tuning_lab benchmark --spec examples/experiments/mock_layernorm.json --require-success --output artifacts/spec-layernorm
python3 -m kernel_tuning_lab demo --output artifacts/demo
```

On a CUDA host with the optional dependencies installed:

```bash
python3 -m kernel_tuning_lab benchmark --spec examples/experiments/triton_batched_gemm.json --hardware H800 --require-success --output artifacts/h800-bmm
KT_PYTHON_BIN=.venv/bin/python bash scripts/run_gpu_smoke.sh 4090 artifacts/4090
KT_PYTHON_BIN=.venv/bin/python bash scripts/run_gpu_smoke.sh 5060ti16g artifacts/5060ti
```

The GPU script performs physical/runtime preflight and ten small experiments: four operators in FP16/BF16 and the two reductions in FP32. Each example contains two shapes, including uneven tail dimensions. Reduction smoke domains deliberately cover the largest row so an eight-attempt budget validates both shapes. Every tested config must succeed; current-run snapshots preserve history separately. A successful suite writes `validation_summary.json`. Physical 5060 Ti evidence is recorded in [5060TI_RESULTS.md](../docs/5060TI_RESULTS.md); other target GPUs still require their own execution. See `docs/RTX5060TI.md` for the local-to-Hopper workflow.

`report/` is generated from the full four-operator Mock run; its captions prominently identify synthetic data. `public_report/` is generated from two complete downloaded upstream files; correctness is unknown and no performance oracle is derived from it. The real sample and retained upstream license live under `kernel_tuning_lab/adapters/data/`; see `docs/PUBLIC_DATA.md` for source and reproduction.

The checked-in `public_report/` contains 15,492 records; the bundled 24-record excerpt remains the offline package fixture. `results/` retains the final Mock run summary, split manifest, comparable replay summaries and explicit L3 unavailable list. `scale/` records the 139,968-candidate exercise and small exact reference.

```bash
ktlab scale-demo --output artifacts/scale
ktlab hub list
ktlab hub import --kernel convolution_milo --gpu A100 --database artifacts/hub.sqlite
ktlab --plugin examples/plugins/deterministic_mock.py benchmark --spec examples/experiments/plugin_mock.json --output artifacts/plugin
```
## Automatic integration examples

`proposal/production/specs/` contains generated CUDA production configurations, not measurement results. `ktlab proposal-suite` discovers and tests all proposal operators through the asset layer; see [PROPOSAL_OPERATORS.md](../docs/PROPOSAL_OPERATORS.md). CPU and deferred validation logs are indexed in `artifacts/proposal-operators-validation/`.

`extensions/` provides independent SDK-only assets and schema-v2 configurations. Run `ktlab run --spec examples/extensions/cpu_gemm.json --output artifacts/extensions/cpu` to combine two Kernel implementations, two algorithms and two shapes automatically. `custom_operator.json`, `conditional_space.json` and `scoped_claim.json` cover new operator semantics, mixed/conditional parameters and declared-vs-reference evidence. The four `failure_*.json` files intentionally fail with saved diagnostics. `gpu_gemm.json` selects 5060 Ti 16GB and remains deferred without CUDA.

`ktlab scaffold kernel|algorithm --output DIRECTORY` generates a working asset and experiment; see [EXTENDING.md](../docs/EXTENDING.md). New execution evidence is under `artifacts/auto-integration-validation/`, including CPU runs, failures, timeouts and template checks.

## Shape/config study

`research_phase1/` freezes the explicit T1 GEMM study: tail acceptance, pilot,
10-Shape/64-config primary matrix, matched library/simple baselines, repeated
anchors, limited BF16 and W4A16 supplements, and unmeasured cross-hardware
templates. `cohort.py` is an SDK algorithm plugin; `design.py` regenerates JSON
specifications. Use the existing `ktlab research` command or its serial telemetry
wrapper `scripts/collect_phase1.py`. The default primary matrix requests 1,920
observations plus independent acceptance, not 64 total measurements. Complete
real results, raw-data selection/Replay/L3 commands and limits are in
[RESEARCH_PHASE1.md](../docs/RESEARCH_PHASE1.md).

