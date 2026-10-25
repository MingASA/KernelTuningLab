# RTX 5060 Ti 16GB: local validation before Hopper

The local validation profile is **`rtx5060ti-16gb`**. Accepted CLI aliases include `5060ti16g`, `5060ti`, `RTX 5060 Ti`, and `NVIDIA GeForce RTX 5060 Ti 16GB`. Bare 5060 Ti aliases select the 16GB model; device verification requires the actual memory capacity and rejects the 8GB card, plain RTX 5060, wrong architecture, or a datasheet substituted for a probe.

## Hardware and software contract

This is consumer Blackwell with compute capability **12.0 / `sm_120`**. Hopper uses a different target, `sm_90`. NVIDIA lists 16GB/8GB variants and 448 GB/s bandwidth. The planning profile derives 36 SMs from 4608 CUDA cores and 128 cores per RTX Blackwell SM; physical probing remains authoritative. Per-block opt-in shared memory is 99 KiB, with 64K 32-bit registers per SM. Dense FP16/FP32-accumulation throughput and L2 size remain unknown in the profile rather than being inferred from AI TOPS or other GPUs.

Sources: [CUDA device capability table](https://developer.nvidia.com/cuda/gpus), [RTX 5060 family](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/), [GeForce specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/), [RTX Blackwell architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf), [Blackwell resource limits](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html).

The declared native Triton baseline is **PyTorch >= 2.7, PyTorch's CUDA runtime >= 12.8, Triton >= 3.3**, on Linux or a Linux environment in WSL2. Newer CUDA 13.x builds are accepted. The adapter checks this baseline before input allocation or JIT compilation. `torch.version.cuda` is checked; a system `nvcc` version alone does not establish that the installed Torch wheel supports the GPU. The environment records Torch's compiled architectures as diagnostic metadata without incorrectly rejecting generic PTX forward compatibility.

[PyTorch 2.7 introduced Blackwell/CUDA 12.8 and Triton 3.3 support](https://pytorch.org/blog/pytorch-2-7/). [Current PyTorch release guidance moves Blackwell users toward CUDA 13.x builds](https://pytorch.org/blog/pytorch-2-12-release-blog/); select a driver matching that runtime. These minima declare this backend's supported native path, and do not claim that every older CUDA application is incompatible: [NVIDIA documents separate PTX forward-compatibility rules](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html).

## Set up the validation machine

Use a dedicated environment. Install a Blackwell-capable Torch wheel using the current [official installation guidance](https://pytorch.org/get-started/locally/) and a matching NVIDIA driver. For a machine prepared for CUDA 13.0 wheels:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -e '.[gpu-blackwell]'
.venv/bin/python -m kernel_tuning_lab probe --gpu --hardware 5060ti16g
```

An explicitly archived CUDA 12.8 baseline is available in the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/), e.g. `torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`. Use its matching dependencies; do not independently replace Torch's required Triton version with an arbitrary one. Native Windows Triton is outside this Linux backend's scope.

Default `ktlab probe` stays CPU-only. Adding `--gpu` explicitly loads the GPU dependencies and records the actual model, architecture, usable capacity, driver, Torch/CUDA/Triton versions and kernel hashes. Preflight confirms device/runtime declarations; it does not assert numerical correctness.

## Run the strict local gate

```bash
KT_PYTHON_BIN=.venv/bin/python bash scripts/run_gpu_smoke.sh 5060ti16g artifacts/5060ti
```

The suite is intentionally light on VRAM: each operator uses two small shapes, including uneven GEMM tails or reduction widths 193 and 1025. Batched GEMM includes explicit batches; reductions also exercise constant and large-offset rows. There are ten stages: four operators in FP16/BF16 and Softmax/LayerNorm in FP32. Each stage admits eight evaluations with seeded inputs, correctness checks, warmup and repeated timing. Kernels use the portable config spaces and are compiled for the physical GPU; Hopper-only variants are not introduced.

Each stage uses `--require-all-success`: **any failed tested config or uncovered requested shape fails the gate**, even if another config worked. Invalid config, compile, numerical and runtime failures remain distinguishable in saved records. Successful results and compatible cache reuse are recorded separately.

| Artifact | Meaning |
| --- | --- |
| `preflight.json` | Physical device and runtime checks |
| `<operator>-<dtype>/run_summary.json` | Current stage coverage/status counts |
| `<operator>-<dtype>/run_records.jsonl` | Exact current-run records used for the gate |
| `<operator>-<dtype>/measurements.sqlite` and `measurements.jsonl` | Full journal/history, including previous runs |
| `validation_summary.json` | Ten-stage aggregate, source hashes, cache counts and device identity |

The aggregate checks physical `measured` provenance, GPU correctness evidence, expected operator/dtype, exact requested shapes, device capacity, runtime identity and all ten stages. Mock/imported/unknown records cannot pass. Adapter or kernel source changes across preflight and stages also fail, including files shared by different operators. Stage failure returns nonzero and retains diagnostics; a missing final success summary is not approval. The script clears previous completion summaries and current-run snapshots before starting, while keeping the measurement caches. An interrupted retry cannot retain an earlier success marker or combine unfinished stages with old snapshots.

Rebuild the aggregate from existing stage files with:

```bash
.venv/bin/python -m kernel_tuning_lab.gpu_validation artifacts/5060ti
```

## Move to Hopper

After the local gate passes, transfer the source and portable experiment specifications along with its validation summary. On each target server, install its compatible runtime and repeat the gate with that physical profile:

```bash
KT_PYTHON_BIN=.venv/bin/python bash scripts/run_gpu_smoke.sh H200 artifacts/h200
KT_PYTHON_BIN=.venv/bin/python bash scripts/run_gpu_smoke.sh H800 artifacts/h800
```

The 5060 Ti result verifies the recorded device's code paths and numerical behavior. Hopper still needs its own compilation, resource and correctness checks; tuning results and compiled kernels are specific to architecture/environment. Use separate output directories so the observations and hardware-bound caches remain explicit. Deployment and upload are not automated by this script.

The 2026-10-25 physical WSL validation passed all ten strict stages (80 records),
and the repaired default proposal suite passed all sixteen dtype/operator jobs.
The validated stack is Python 3.12.13, Torch 2.13.0+cu132 and Triton 3.7.1.
See [5060TI_RESULTS.md](5060TI_RESULTS.md) for immutable first-failure evidence,
resource-legality fixes, timing scopes and the limited sampling boundary.
