# Proposal operator suite

Status: **generated**; runtime: **cuda**; scale: **production**

| Operator / case | Dtype | Status | Run report |
| --- | --- | --- | --- |
| attention-cuda-float16-production-default | float16 | generated | [report](runs/attention-cuda-float16-production-default-dc6d6f0b/report.md) |
| attention-cuda-bfloat16-production-default | bfloat16 | generated | [report](runs/attention-cuda-bfloat16-production-default-a6bf705d/report.md) |
| dequantize_w2-cuda-float16-production-default | float16 | generated | [report](runs/dequantize_w2-cuda-float16-production-default-e29bd775/report.md) |
| dequantize_w2-cuda-bfloat16-production-default | bfloat16 | generated | [report](runs/dequantize_w2-cuda-bfloat16-production-default-5242ad41/report.md) |
| dequantize_w3-cuda-float16-production-default | float16 | generated | [report](runs/dequantize_w3-cuda-float16-production-default-e4a4a25a/report.md) |
| dequantize_w3-cuda-bfloat16-production-default | bfloat16 | generated | [report](runs/dequantize_w3-cuda-bfloat16-production-default-5cfdb8a5/report.md) |
| dequantize_w4-cuda-float16-production-default | float16 | generated | [report](runs/dequantize_w4-cuda-float16-production-default-4d6ae17d/report.md) |
| dequantize_w4-cuda-bfloat16-production-default | bfloat16 | generated | [report](runs/dequantize_w4-cuda-bfloat16-production-default-d77c76eb/report.md) |
| moe_grouped_gemm-cuda-float16-production-default | float16 | generated | [report](runs/moe_grouped_gemm-cuda-float16-production-default-d793c71f/report.md) |
| moe_grouped_gemm-cuda-bfloat16-production-default | bfloat16 | generated | [report](runs/moe_grouped_gemm-cuda-bfloat16-production-default-498419ca/report.md) |
| w2a16_gemm-cuda-float16-production-default | float16 | generated | [report](runs/w2a16_gemm-cuda-float16-production-default-0e0b3440/report.md) |
| w2a16_gemm-cuda-bfloat16-production-default | bfloat16 | generated | [report](runs/w2a16_gemm-cuda-bfloat16-production-default-283af42a/report.md) |
| w3a16_gemm-cuda-float16-production-default | float16 | generated | [report](runs/w3a16_gemm-cuda-float16-production-default-57969c0f/report.md) |
| w3a16_gemm-cuda-bfloat16-production-default | bfloat16 | generated | [report](runs/w3a16_gemm-cuda-bfloat16-production-default-1ff931de/report.md) |
| w4a16_gemm-cuda-float16-production-default | float16 | generated | [report](runs/w4a16_gemm-cuda-float16-production-default-a98681b1/report.md) |
| w4a16_gemm-cuda-bfloat16-production-default | bfloat16 | generated | [report](runs/w4a16_gemm-cuda-bfloat16-production-default-7b92d3f0/report.md) |

CPU results verify the Python contracts, not GPU kernels. CUDA compilation and performance require the physical device.
