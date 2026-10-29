"""Explicit environment forwarding for reviewed providers/generated programs.

No credentials or arbitrary parent variables are forwarded. This is hygiene
for trusted code, not filesystem/network isolation or a malicious-code sandbox.
"""
import os


def reviewed_environment(*,cpu_only=False):
    names=('PATH','LANG','LC_ALL','LD_LIBRARY_PATH','SYSTEMROOT','PYTHONPATH','PYTHONHASHSEED',
        'CUDA_VISIBLE_DEVICES','CUDA_HOME','CUDA_PATH','CUDA_CACHE_PATH','TRITON_CACHE_DIR',
        'TORCHINDUCTOR_CACHE_DIR','TORCH_CUDA_ARCH_LIST','MAX_JOBS','CXX','CC','TMPDIR','TEMP','TMP',
        'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
        'KT_FLUTE_BUILD_MANIFEST','VLLM_TUNED_CONFIG_FOLDER','VLLM_BATCH_INVARIANT','VLLM_TRITON_USE_TD')
    env={key:os.environ[key] for key in names if key in os.environ}
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if cpu_only:env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return env
