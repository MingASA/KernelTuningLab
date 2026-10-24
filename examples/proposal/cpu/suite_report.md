# Proposal operator suite

Status: **passed**; runtime: **python**; scale: **smoke**

| Operator / case | Dtype | Status | Run report |
| --- | --- | --- | --- |
| attention-python-float32-smoke-default | float32 | passed | [report](runs/attention-python-float32-smoke-default-e00cb98d/report.md) |
| dequantize_w2-python-float32-smoke-default | float32 | passed | [report](runs/dequantize_w2-python-float32-smoke-default-5670c367/report.md) |
| dequantize_w3-python-float32-smoke-default | float32 | passed | [report](runs/dequantize_w3-python-float32-smoke-default-6c6dc83c/report.md) |
| dequantize_w4-python-float32-smoke-default | float32 | passed | [report](runs/dequantize_w4-python-float32-smoke-default-05f8f022/report.md) |
| moe_grouped_gemm-python-float32-smoke-default | float32 | passed | [report](runs/moe_grouped_gemm-python-float32-smoke-default-28d846e0/report.md) |
| w2a16_gemm-python-float32-smoke-default | float32 | passed | [report](runs/w2a16_gemm-python-float32-smoke-default-06117f0c/report.md) |
| w3a16_gemm-python-float32-smoke-default | float32 | passed | [report](runs/w3a16_gemm-python-float32-smoke-default-deac515d/report.md) |
| w4a16_gemm-python-float32-smoke-default | float32 | passed | [report](runs/w4a16_gemm-python-float32-smoke-default-86196b6b/report.md) |

CPU results verify the Python contracts, not GPU kernels. CUDA compilation and performance require the physical device.
