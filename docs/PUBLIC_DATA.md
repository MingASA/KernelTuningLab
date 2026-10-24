# Public autotuning data

The v0.1 importer reads **actual Kernel Tuner cache JSON**, including gzip files,
and explicit `results` / `env` JSON envelopes. A full Benchmark Hub catalogue,
verified whole-file downloads, and normalized imports are implemented using the
Python standard library. An attributed upstream excerpt is also bundled for
offline use. Full downloaded datasets remain in artifacts rather than the wheel.

## Primary sources checked

- [AutoTuningAssociation benchmark hub](https://github.com/AutoTuningAssociation/benchmark_hub)
  publishes original Kernel Tuner caches alongside T4 exports, source kernels,
  and T1 input specifications. We checked the original cache format directly.
- [Pinned A100 convolution source cache](https://github.com/AutoTuningAssociation/benchmark_hub/blob/ca104e907d22c7a78067c234fb933da731d18a90/cachefiles/convolution_milo/A100.json.gz)
  contains 4,362 entries for `convolution_kernel`, `NVIDIA A100-PCIE-40GB`, problem
  size `[4096, 4096]`. Timing entries retain 32 individual execution samples.
- [Pinned kernel input specification](https://github.com/AutoTuningAssociation/benchmark_hub/blob/ca104e907d22c7a78067c234fb933da731d18a90/kernels/convolution_milo.json)
  describes configuration parameters and legality conditions. A numeric problem
  size describes the source launch domain; the generic importer labels dimensions
  `X`, `Y`, `Z` and does not guess GEMM semantics.
- [Kernel Tuner API documentation](https://kerneltuner.github.io/kernel_tuner/stable/user-api.html)
  documents the returned result list and environment, optional verification, and
  its separate selected-best `store_results` format.
- [Kernel Tuner cache implementation](https://github.com/KernelTuner/kernel_tuner/blob/master/kernel_tuner/util.py)
  defines cache headers and serialized failure markers. The importer recognizes
  `InvalidConfig`, `CompilationFailedConfig`, and `RuntimeFailedConfig`.

The data source is pinned to commit
`ca104e907d22c7a78067c234fb933da731d18a90`, rather than a mutable branch.
Its repository is [MIT licensed](https://github.com/AutoTuningAssociation/benchmark_hub/blob/ca104e907d22c7a78067c234fb933da731d18a90/LICENSE).
The full notice is redistributed in
`kernel_tuning_lab/adapters/data/BENCHMARK_HUB_LICENSE.txt`.

## Full Benchmark Hub discovery and imports

The adapter discovers the complete active `cachefiles/` tree at the fixed commit
through the primary [GitHub tree API](https://api.github.com/repos/AutoTuningAssociation/benchmark_hub/git/trees/ca104e907d22c7a78067c234fb933da731d18a90?recursive=1).
The checked tree was **not truncated**. The saved catalogue contains **54 gzip
files: 27 original Kernel Tuner caches and 27 corresponding T4 exports**.
Original caches total **477,951,676 compressed bytes**; both formats together
total **942,969,143 compressed bytes**. These are catalogue byte totals, not a
claim that every file has been downloaded.

| Kernel directory | Original cache files | Compressed bytes | Source GPU labels |
| --- | --- | --- | --- |
| `convolution_milo` | 7 | 7,725,655 | A100, A4000, A6000, A6000_original, MI250X, W6600, W7800 |
| `dedispersion_milo` | 6 | 20,208,680 | A100, A4000, A6000, MI250X, W6600, W7800 |
| `gemm_milo` | 7 | 254,018,224 | A100, A4000, A6000, MI250X, W6600, W7700, W7800 |
| `hotspot_milo` | 7 | 195,999,117 | A100, A4000, A6000, MI250X, MI50, W6600, W7800 |

There are eight physical GPU labels in this active collection; `A6000_original`
is an additional collection version, not a ninth GPU. The README describes 24
core spaces, while the actual pinned tree additionally contains this A6000
version, GEMM/W7700, and Hotspot/MI50. **H200, H800, and RTX 4090 are absent from
the checked active catalogue.** Local hardware profiles for those devices do
not create public benchmark measurements. Archived `currently_unused/` and
hyperparameter-tuning collections are separate from this active catalogue and
are explicitly excluded from these counts.

The full catalogue records kernel, source GPU label, path, pinned raw URL, byte
size, format, source commit, license, and Git blob SHA for every file. Original
caches are returned by default; `include_t4=True` also exposes T4 entries.
Discovery always persists both formats when `cache_dir` is provided and can
subsequently reuse that catalogue offline. Mutable branch names are rejected;
an explicit alternate revision must be a full 40-character commit SHA.

```python
from kernel_tuning_lab.adapters.benchmark_hub import (
    discover_benchmark_hub, download_benchmark_hub, import_benchmark_hub,
)

catalogue = discover_benchmark_hub(cache_dir="artifacts/public-hub")
entry = next(row for row in catalogue
             if row.kernel == "convolution_milo" and row.gpu == "A100")
original_gzip = download_benchmark_hub(entry, "artifacts/public-hub")
records = import_benchmark_hub(entry, "artifacts/public-hub")
```

`download_benchmark_hub` retains the exact original compressed bytes and verifies
both the tree byte size and the Git blob hash (`SHA1("blob <size>\\0" + bytes)`).
Its sidecar adds SHA-256, source commit/URL/path, format, license, and the unknown
correctness boundary. Reusing a local file rechecks integrity. A checksum or size
mismatch fails before import. A full MIT license notice is retained alongside
the artifacts. T4 files can be catalogued/downloaded but are explicitly rejected
by `import_benchmark_hub`; their distinct schema is not guessed.

### Full files actually downloaded and normalized

The following imports were performed in this workspace, with no sampling:

| Full source | Device from header | Rows | Correctness unknown | Compile failures | Runtime failures | Verified successes |
| --- | --- | --- | --- | --- | --- | --- |
| convolution_milo / A100 | NVIDIA A100-PCIE-40GB | 4,362 | 4,201 | 6 | 155 | 0 |
| dedispersion_milo / A4000 | NVIDIA RTX A4000 | 11,130 | 11,130 | 0 | 0 | 0 |
| Total | Two distinct operators and GPUs | **15,492** | **15,331** | **6** | **155** | **0** |

Convolution's source shape is `[4096,4096]`; dedispersion's is
`[25000,2048,1]`. The generic importer retains these as launch dimensions X/Y/Z.
Full-file presence establishes completeness relative to that source file; it
does not establish independent correctness or every feasible configuration in
an unknown legal universe. Normalized records retain `provenance="imported"`,
the full source manifest, file hash, upstream failures, and unknown correctness.

Evidence and reusable data are in `artifacts/public-hub/`:

- `benchmark_hub_catalogue_ca104e907d22c7a78067c234fb933da731d18a90.json`
  contains the entire 54-file active catalogue.
- `cachefiles/convolution_milo/A100.json.gz` plus `.source.json` retains the
  original full source and its verification manifest. SHA-256:
  `de449bddefb42721a9c92d63b0984caeb04cc48519d58135a487e832f8196b9d`.
- `cachefiles/dedispersion_milo/A4000.json.gz` plus `.source.json` retains the
  second full source. SHA-256:
  `657e9de1ba7a029324c71a62d13e1dd06a8ae5311af0255c15a086d0bbad6044`.
- `convolution_milo_A100.jsonl` and `dedispersion_milo_A4000.jsonl` contain all
  normalized records, ready for the journal's JSONL importer.
- `import_summary.json` records actual imported counts/statuses and source IDs.
  `BENCHMARK_HUB_LICENSE.txt` retains the full MIT notice.

These are upstream observations, not new local GPU measurements. No imported
timing was promoted to a verified optimization oracle. The remaining 25 original
caches and all 27 T4 exports were catalogued, not imported in this run.

## Bundled real excerpt

`kernel_tuning_lab/adapters/data/benchmark_hub_convolution_A100.sample.json`
contains **24 original entries** selected at evenly spaced insertion-order
indices through the source's 4,362 rows. Parameter values, runtime samples,
source timing summaries, and timestamps are retained. Selection does not rank
entries by speed. The only addition is `_kernel_tuning_lab_source`, recording
the upstream URL and commit, compressed/uncompressed SHA-256, original/excerpt
row counts, source indices, license, and correctness limitation.

This is a real upstream measurement excerpt. It is **not** a local GPU rerun,
an independent validation, or a complete performance table. Its small size
cannot establish a full-space optimum, search difficulty, or cross-GPU transfer
quality. It is separate from the synthetic measurements used by the mock demo.

Import it without a network connection:

```python
from kernel_tuning_lab.adapters import bundled_sample_path, load_kernel_tuner_json

records = load_kernel_tuner_json(bundled_sample_path())
assert len(records) == 24
assert not any(record.successful for record in records)
```

The explicit network command below reproduces the excerpt and license from
the pinned source; ordinary package imports never download data:

```bash
python -m kernel_tuning_lab.adapters.fetch_sample
```

## Correctness is an explicit trust boundary

The inspected cache has `verification_time = 0`. A runtime measurement does not
prove that the output was checked. Consequently **default imported timings have
`RecordStatus.CORRECTNESS_UNKNOWN`, not `SUCCESS`**. Samples remain available
in `runtime_samples_ms`; `successful` is false and `median_ms` is `None`.
Metadata contains `correctness_verified = False` and the reason is preserved.
Reports count this as an unknown outcome separately from genuine failures.
Replay scoring and L3 matrices must exclude these rows from verified oracles.

An importer caller can assert verification only by passing both
`correctness="verified"` and a nonempty `correctness_evidence` string applying to
**all** successful source entries being imported. This declaration is retained
as caller-supplied evidence; the importer does not execute an independent check.
Never use this flag just to make unknown data eligible for scoring. For mixed
verification evidence, partition the input or construct per-record statuses
with explicit evidence. Upstream failure entries keep their failure status even
when verified import is requested.

## API and format mapping

```python
load_kernel_tuner_json(
    path, *, hardware=None, operator=None, shape=None,
    correctness="unknown", correctness_evidence=None, source_url=None,
    experiment_id="kernel_tuner_import", split="train", tune_params_keys=None,
) -> list[MeasurementRecord]
```

| Source | Lab representation |
| --- | --- |
| `device_name` | Hardware name; unspecified architecture/capacities stay unknown |
| `kernel_name` | Operator name with implementation `kernel_tuner`; dtype/kind unknown unless overridden |
| Numeric `problem_size` | Positive integer `ShapeSpec` dimensions `X`, `Y`, `Z`, ... |
| `tune_params_keys` / `tune_params` | Exact configuration keys; additional metrics are never guessed to be tuning parameters |
| `times` | Individual positive runtime samples in milliseconds |
| Numeric `time`, no samples | One aggregate value, explicitly marked `upstream_aggregate_only`; repetition count unknown |
| `compile_time` | Milliseconds; availability tracked separately from a zero compatibility default |
| Five source cost components | Sum in seconds only when all are available; availability and individual components retained |
| Failure marker string in `time` | Invalid, compile, or runtime failure; known correctness failure markers are also supported |
| `env` / `environment` | Environment metadata, retained without inventing missing versions |
| Source URL/header/metrics | Record metadata; provenance always `imported` |

The generic reader accepts: (1) the `cache` mapping with its upstream header;
(2) an envelope with `results`, optional `env`/`environment`, and explicit tuning
keys/domains; (3) a bare result list with caller-supplied tuning keys and enough
hardware/operator/shape context. Symbolic problem sizes require an explicit
`ShapeSpec`. Configurations are not replayed, compiled, or relabelled as this
project's Triton GEMM.

Malformed entries, absent tuning keys, nonpositive/nonfinite sample values, and
unknown failure markers fail loudly with `KernelTunerImportError`. The importer
does not silently repair unfinished JSON files. Missing warmup counts, dtype,
compiler versions, hardware limits, and verification evidence remain unknown.
Repeated imports produce stable records suitable for store deduplication.

## Reporting and current limits

```python
from kernel_tuning_lab.analysis import generate_report
report = generate_report(records, "artifacts/public_import")
```

This produces Markdown, seven standalone SVGs, and audit JSON. If matplotlib is
installed it lazily renders matching PNGs. Unverified imported rows appear in
the inventory and outcome/coverage sections, with unavailable verified
performance metrics explicitly marked insufficient. They do not populate
verified performance heatmaps, transfer matrices, or sensitivity comparisons.

Coverage uses an explicitly labelled observed rectangle; it is not a claim
about the complete legal candidate universe. The source/excerpt row counts are
also reported so a 24-row excerpt cannot appear as a complete 4,362-row study.
Sensitivity comparisons vary exactly one configuration parameter with the rest
held fixed. Transfer diagnostics require measurements of the selected source
configuration on the target and use matching operator/shape/provenance/split/
experiment identities. MOCK charts carry a prominent synthetic-data label.

Not implemented in v0.1: T4 schema import, Kernel Tuner `store_results` selected
best export import, benchmark-hub T1 legality execution, automatic whole-suite batch ingestion,
recreating upstream verification inputs, physical hardware reruns, and policy
training on these unknown-correctness measurements. These are explicit future
extensions rather than inferred guarantees from the public timing cache.
