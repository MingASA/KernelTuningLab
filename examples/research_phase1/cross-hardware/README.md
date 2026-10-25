# Cross-hardware replication (not executed on these GPUs)

The four JSON files reuse T1 v1, the ten Shapes, FP16, the 64-config factorial,
seeds 7/19/31, warmup 10, 31 samples and Graph 512-call timing. All four target
GPUs remain **unmeasured in this study**. Run serially on the actual target:

```bash
ktlab probe --gpu
ktlab research --spec examples/research_phase1/cross-hardware/h20.json \
  --output artifacts/h20/t1-v1 --write-only
ktlab research --spec examples/research_phase1/cross-hardware/h20.json \
  --output artifacts/h20/t1-v1
```

Use separate output directories per physical device/software environment. Check
available host RAM/CPU and compile one worker at a time. For multi-GPU servers,
record the selected CUDA device/physical identity explicitly; the optional
`collect_phase1.py` telemetry wrapper was validated on a single visible GPU and
its current nvidia-smi query is not a per-device scheduler.

`hardware.py` currently has four profiles and no H20 constructor. `h20.json`
therefore sets `hardware=null`: the Runner obtains actual properties with the
existing Torch probe. Do not copy H200 SM count, L2 size or performance into an
H20 profile just because both can report the same compute capability. Archive
the real device name, architecture, SM count, visible memory, shared-memory
limits, supported compiler targets and driver/runtime versions. Fields the
probe cannot determine remain unknown; vendor-derived peak numbers need a
separately cited source. That probe is the basis for a future H20 profile.

H200/H800 templates select the existing **SXM** profiles, and 4090 selects the
desktop profile. A mismatching variant or MIG slice must not be relabelled SXM.
Use `hardware=null` and a distinct dataset identity for an unrecognized variant,
then validate its measured properties before introducing a profile.

T1 uses ordinary Tensor Core dot operations and grouped ordering, with no
Hopper-only TMA/WGMMA/cluster path. Its shared-memory estimate is checked against
the actual probe and exact compiled shared-memory metadata. Static profile
acceptance of a config is only a preliminary constraint check. Common **verified**
configs are the intersection of successful, reference-verified Shape/config
cells on all measured GPUs. Failures, unsupported configs and missing cells stay
separate. Additional hardware-specific candidates require a separately named
cohort and must not replace the public 64-config cohort silently.

For each Shape, join only the same implementation/source version, dtype,
layout and measurement scope. Within the common observed set, report:

- Source top-config latency on the target divided by target finite observed best.
- Spearman rank correlation, strict Top-5/10/20% overlap and near-optimal coverage.
- Counts of common observed, failed, unsupported and untested cells.
- Software/protocol differences that could confound a hardware interpretation.

Build a separate `PerformanceMatrix` per GPU; never mix GPU latencies in one
matrix. Standard SQLite/JSONL imports preserve original record identities.
Future Leave-One-GPU-Out splits can use the existing `datasets.split_records`
with `holdout="hardware"`, after multiple real GPU datasets exist. With only
the 5060 Ti, no multi-GPU training or transfer result is asserted.
