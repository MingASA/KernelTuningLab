#!/usr/bin/env bash
# Run later on a CUDA host: bash scripts/run_gpu_smoke.sh H200 artifacts/h200
set -euo pipefail

KT_GPU_PROFILE="${1:?Usage: bash scripts/run_gpu_smoke.sh H200|H800|4090|5060ti16g [output-dir]}"
KT_GPU_OUTPUT="${2:-artifacts/gpu-smoke}"
KT_PYTHON_BIN="${KT_PYTHON_BIN:-python3}"

mkdir -p "$KT_GPU_OUTPUT"
# Discard completion markers from earlier attempts while retaining measurement caches.
rm -f -- "$KT_GPU_OUTPUT/validation_summary.json"
for KT_OPERATOR in gemm batched_gemm softmax layernorm; do
    for KT_DTYPE in float16 bfloat16; do
        rm -f -- "$KT_GPU_OUTPUT/$KT_OPERATOR-$KT_DTYPE/run_summary.json" \
            "$KT_GPU_OUTPUT/$KT_OPERATOR-$KT_DTYPE/run_records.jsonl"
    done
done
for KT_OPERATOR in softmax layernorm; do
    rm -f -- "$KT_GPU_OUTPUT/$KT_OPERATOR-float32/run_summary.json" \
        "$KT_GPU_OUTPUT/$KT_OPERATOR-float32/run_records.jsonl"
done
"$KT_PYTHON_BIN" -m kernel_tuning_lab probe --gpu --hardware "$KT_GPU_PROFILE" > "$KT_GPU_OUTPUT/preflight.json"

for KT_OPERATOR in gemm batched_gemm softmax layernorm; do
    for KT_DTYPE in float16 bfloat16; do
        "$KT_PYTHON_BIN" -m kernel_tuning_lab benchmark \
            --backend triton --operator "$KT_OPERATOR" --dtype "$KT_DTYPE" \
            --hardware "$KT_GPU_PROFILE" --max-evaluations 8 --require-success --require-all-success \
            --output "$KT_GPU_OUTPUT/$KT_OPERATOR-$KT_DTYPE"
    done
done

for KT_OPERATOR in softmax layernorm; do
    "$KT_PYTHON_BIN" -m kernel_tuning_lab benchmark \
        --backend triton --operator "$KT_OPERATOR" --dtype float32 \
        --hardware "$KT_GPU_PROFILE" --max-evaluations 8 --require-success --require-all-success \
        --output "$KT_GPU_OUTPUT/$KT_OPERATOR-float32"
done

"$KT_PYTHON_BIN" -m kernel_tuning_lab.gpu_validation "$KT_GPU_OUTPUT"
