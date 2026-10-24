# Proposal operator suite

Status: **deferred**; runtime: **cuda**; scale: **smoke**

| Operator / case | Dtype | Status | Run report |
| --- | --- | --- | --- |
| attention-cuda-float16-smoke-default | float16 | deferred | [report](runs/attention-cuda-float16-smoke-default-96516df2/report.md) |
| attention-cuda-bfloat16-smoke-default | bfloat16 | deferred | [report](runs/attention-cuda-bfloat16-smoke-default-48f17963/report.md) |
| dequantize_w2-cuda-float16-smoke-default | float16 | deferred | [report](runs/dequantize_w2-cuda-float16-smoke-default-4ce1cb08/report.md) |
| dequantize_w2-cuda-bfloat16-smoke-default | bfloat16 | deferred | [report](runs/dequantize_w2-cuda-bfloat16-smoke-default-9b60c2e1/report.md) |
| dequantize_w3-cuda-float16-smoke-default | float16 | deferred | [report](runs/dequantize_w3-cuda-float16-smoke-default-c62ce87b/report.md) |
| dequantize_w3-cuda-bfloat16-smoke-default | bfloat16 | deferred | [report](runs/dequantize_w3-cuda-bfloat16-smoke-default-8c894aab/report.md) |
| dequantize_w4-cuda-float16-smoke-default | float16 | deferred | [report](runs/dequantize_w4-cuda-float16-smoke-default-4fcaa03c/report.md) |
| dequantize_w4-cuda-bfloat16-smoke-default | bfloat16 | deferred | [report](runs/dequantize_w4-cuda-bfloat16-smoke-default-7bbc46fc/report.md) |
| moe_grouped_gemm-cuda-float16-smoke-default | float16 | deferred | [report](runs/moe_grouped_gemm-cuda-float16-smoke-default-0c9c1111/report.md) |
| moe_grouped_gemm-cuda-bfloat16-smoke-default | bfloat16 | deferred | [report](runs/moe_grouped_gemm-cuda-bfloat16-smoke-default-5a43acf3/report.md) |
| w2a16_gemm-cuda-float16-smoke-default | float16 | deferred | [report](runs/w2a16_gemm-cuda-float16-smoke-default-417baa39/report.md) |
| w2a16_gemm-cuda-bfloat16-smoke-default | bfloat16 | deferred | [report](runs/w2a16_gemm-cuda-bfloat16-smoke-default-ce41cd00/report.md) |
| w3a16_gemm-cuda-float16-smoke-default | float16 | deferred | [report](runs/w3a16_gemm-cuda-float16-smoke-default-ddb1edaa/report.md) |
| w3a16_gemm-cuda-bfloat16-smoke-default | bfloat16 | deferred | [report](runs/w3a16_gemm-cuda-bfloat16-smoke-default-19d48d29/report.md) |
| w4a16_gemm-cuda-float16-smoke-default | float16 | deferred | [report](runs/w4a16_gemm-cuda-float16-smoke-default-9906b4aa/report.md) |
| w4a16_gemm-cuda-bfloat16-smoke-default | bfloat16 | deferred | [report](runs/w4a16_gemm-cuda-bfloat16-smoke-default-271afe0d/report.md) |

CPU results verify the Python contracts, not GPU kernels. CUDA compilation and performance require the physical device.
