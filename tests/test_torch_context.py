"""Reference-state and cache-identity guards without an installed Torch."""
from contextlib import contextmanager
from types import SimpleNamespace
import os

import pytest

from kernel_tuning_lab.torch_context import strict_fp32_reference, torch_execution_context


class TorchState:
    def __init__(self, modern=False, value='medium'):
        self.mode=value
        self.autocast_enabled=True
        self.legacy_sets=[]
        matmul=SimpleNamespace(allow_tf32=True,allow_fp16_reduced_precision_reduction=True,
            allow_bf16_reduced_precision_reduction=True,allow_fp16_accumulation=False)
        if modern:matmul.fp32_precision=value
        self.backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul))

    def get_float32_matmul_precision(self):return self.mode
    def set_float32_matmul_precision(self,value):self.mode=value;self.legacy_sets.append(value)
    def is_autocast_enabled(self,device='cuda'):return self.autocast_enabled
    def get_autocast_dtype(self,device):return 'bfloat16'

    @contextmanager
    def autocast(self,device_type,enabled):
        previous=self.autocast_enabled;self.autocast_enabled=enabled
        try:yield
        finally:self.autocast_enabled=previous


@pytest.mark.parametrize('modern,value',[(False,'highest'),(False,'high'),(False,'medium'),(True,'none'),(True,'tf32'),(True,'bfx9')])
def test_strict_reference_restores_full_precision_and_autocast_after_exception(modern,value):
    torch=TorchState(modern,value)
    with pytest.raises(ValueError,match='reference failed'):
        with strict_fp32_reference(torch):
            assert not torch.autocast_enabled
            assert (torch.backends.cuda.matmul.fp32_precision if modern else torch.mode)==('ieee' if modern else 'highest')
            raise ValueError('reference failed')
    assert torch.autocast_enabled
    assert (torch.backends.cuda.matmul.fp32_precision if modern else torch.mode)==value
    assert torch.legacy_sets==([] if modern else ['highest',value])


def test_execution_identity_changes_with_math_and_compiler_controls_not_cache_location(monkeypatch):
    torch=TorchState(modern=True)
    original=torch_execution_context(torch)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    assert torch_execution_context(torch)!=original
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=True
    monkeypatch.setenv('TRITON_CACHE_DIR','/a/different/compiler/cache')
    assert torch_execution_context(torch)==original
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE','0')
    assert torch_execution_context(torch)!=original


@pytest.mark.skipif(os.environ.get('KT_RUN_GPU_TESTS')!='1',reason='explicit real CUDA opt-in required')
def test_cuda_reference_ignores_outer_autocast_and_restores_caller_precision():
    import torch
    from kernel_tuning_lab.operator_library import _reference_gemm
    from kernel_tuning_lab.models import OperatorSpec
    if not torch.cuda.is_available():pytest.skip('no CUDA device')
    previous=torch.backends.cuda.matmul.fp32_precision if hasattr(torch.backends.cuda.matmul,'fp32_precision') else torch.get_float32_matmul_precision()
    modern=hasattr(torch.backends.cuda.matmul,'fp32_precision')
    try:
        if modern:torch.backends.cuda.matmul.fp32_precision='tf32'
        else:torch.set_float32_matmul_precision('medium')
        a=torch.randn((33,65),device='cuda',dtype=torch.float16)
        b=torch.randn((65,71),device='cuda',dtype=torch.float16)
        with strict_fp32_reference(torch):expected=(a.float()@b.float()).half()
        ctx=SimpleNamespace(runtime='cuda',torch=torch,inputs={'a':a,'b':b},operator=OperatorSpec('gemm',dtype='float16'))
        with torch.autocast('cuda',dtype=torch.bfloat16):actual=_reference_gemm(ctx)['out']
        assert torch.equal(actual,expected)
        assert (torch.backends.cuda.matmul.fp32_precision if modern else torch.get_float32_matmul_precision())==('tf32' if modern else 'medium')
    finally:
        if modern:torch.backends.cuda.matmul.fp32_precision=previous
        else:torch.set_float32_matmul_precision(previous)
