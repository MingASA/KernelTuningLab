# Hardware profiles: H200 SXM, H800 SXM5, RTX 4090, RTX 5060 Ti 16GB

The profiles support CPU-side experiment planning and configuration legality.
They are vendor specifications, not physical probes or benchmark observations.
No CUDA validation was available during this implementation. Real experiments
must use the actual device probe in their measurement records, check the requested
profile, and retain its environment metadata. The baseline configuration space
uses portable GEMM tile, warp, and stage parameters on these devices.

## Public API

```python
from kernel_tuning_lab.hardware import (
    get_hardware_profile, list_hardware_profiles, validate_hardware_profile,
)

list_hardware_profiles()  # ("h200-sxm", "h800-sxm", "rtx4090", "rtx5060ti-16gb")
profile = get_hardware_profile("H800")
validation = validate_hardware_profile(profile, actual_probe)
```

The getter accepts case-insensitive common aliases, including `H200`, `H800`,
`4090`, and full NVIDIA model names. Bare H200/H800 names select the SXM profile.
PCIe, NVL, RTX 4090 D, and laptop products require separate profiles; none are
silently treated as these SKUs. Each call returns fresh metadata so callers cannot
mutate the registry through a returned object.

## Verified planning values

| Canonical profile | Architecture | SMs | Dense FP16 Tensor TFLOPS | Bandwidth, GB/s | Nominal memory |
| --- | --- | ---: | ---: | ---: | --- |
| `h200-sxm` | `sm_90` | 132 | 989.5, derived from rounded sparse headline | 4800 | 141 GB |
| `h800-sxm` | `sm_90`, architecture inference | unknown | unknown | unknown | 80 GB |
| `rtx4090` | `sm_89` | 128 | 165.2 with FP32 accumulation | 1008 | 24 GB |
| `rtx5060ti-16gb` | `sm_120` | 36, derived | unknown | 448 | 16 GB |

For H200 SXM, NVIDIA's product table gives 1,979 FP16 Tensor TFLOPS with sparsity,
141 GB, and 4.8 TB/s. Dividing the rounded sparse rate by two gives the recorded
989.5 dense planning peak; this derivation is not a measurement. NVIDIA's DevTech
presentation explicitly identifies H200 SXM with 132 SMs and 50 MB L2.
[H200 product specifications](https://www.nvidia.com/en-us/data-center/h200/),
[NVIDIA DevTech CUDA optimization presentation, slide 4](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/CUDA-Programming-and-Optimization.pdf).

For RTX 4090, NVIDIA's Ada whitepaper lists 128 SMs, 24 GB GDDR6X, 1008 GB/s,
and 73,728 KB of L2. Its dense FP16 Tensor rate is **165.2 TFLOPS with FP32
accumulation**. The 330.3 TFLOPS figure uses FP16 accumulation and is preserved as
a separate metadata value. Sparse rates are not used as dense throughput.
[Ada architecture whitepaper v2.02, Appendix A, Table 2](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf).

For H800, NVIDIA's official virtualization documentation distinguishes H800 SXM5
80 GB from H800 PCIe 80 GB and H800 NVL 94 GB, and classifies H800 as Hopper.
The current CUDA model table does not separately list H800. Its `sm_90` profile
is an explicitly labeled inference from that Hopper classification. We did not
retrieve a primary SXM source confirming its SM count, bandwidth, Tensor peak,
or L2; these fields remain `0` or `None`. They are not copied from H100 or H800
PCIe. A physical probe can supply device resource limits before execution.
[NVIDIA Hopper H800 product classification](https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h800.html),
[CUDA compute-capability model table](https://developer.nvidia.com/cuda/gpus).

## Architecture limits and units

NVIDIA's tuning guides give 64K 32-bit registers per SM for both architectures.
The profile shared-memory limit is the architectural opt-in maximum per block:
227 KiB for Hopper and 99 KiB for Ada. These values are distinct from the default
static shared-memory allowance, and the actual probe takes precedence.
Hopper features include TMA and thread-block clusters; adding those capabilities
to a profile does not add unsupported knobs to the portable Triton GEMM kernel.
[Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html),
[Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html).

The H200 and RTX 4090 compute capabilities are 9.0 and 8.9 respectively in
NVIDIA's model table. Warp size 32 and maximum block size 1024 follow the CUDA
architecture limits.
[CUDA GPU model capabilities](https://developer.nvidia.com/cuda/gpus),
[CUDA architecture specifications](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).

Throughput is theoretical dense TFLOPS, where an FMA counts as two operations.
Bandwidth uses decimal **bytes** per second (`bandwidth_gbps` is GB/s, despite its
historical field spelling). Cache and shared-memory KB/MB are normalized using
the architectural KiB/MiB convention. `memory_bytes` converts the vendor's nominal
GB label using 10^9 bytes/GB; it is not a claim about exact CUDA-visible memory.
The nominal vendor label is retained separately. Use the probe for exact usable
memory after reservations, ECC, and runtime effects.

## Validation and limitations

Validation checks the model family, architecture, known SM/register counts,
minimum block limits, and nominal memory compatibility. It rejects explicit
PCIe/NVL variants when an SXM profile was requested, MIG identities, and a second
datasheet profile supplied as though it were a device probe. Nominal memory
comparison allows vendor SI/GiB label conventions and runtime reservations;
the accepted ratio of probed bytes to nominal decimal bytes is 0.90–1.15.
Theoretical peak compute and bandwidth are not checked against runtime estimates.

A generic CUDA name such as `NVIDIA H200` can establish family compatibility but
cannot prove the SXM board form factor. Explicit probe variant metadata is checked
when available. Validation never fills unknown probe capacities or promotes
specifications to measured evidence. Profiles preserve source URLs, individual
field sources, units, unknown fields, inference notes, and verification date.

Run the CPU checks with:

```bash
.venv/bin/python -m pytest tests/test_hardware.py -q
```

Real GPU correctness, compilation, resource limits, and performance on these
models remain to be checked when those devices are available.

The 5060 Ti profile explicitly selects the 16GB desktop version and rejects an
8GB or unknown-capacity probe. Its source-backed limits, native Blackwell runtime
baseline and lightweight validation-to-Hopper workflow are documented in
[RTX5060TI.md](RTX5060TI.md). The GeForce model's 99 KiB block shared-memory
limit is independent of Hopper's 227 KiB; available architecture features do not
add unsupported knobs to the portable kernels.
