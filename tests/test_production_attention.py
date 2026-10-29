"""Native attention source/options/representation contracts, CPU only."""
import json
import pytest
from kernel_tuning_lab.production import flex_attention,sglang_decode
from kernel_tuning_lab.models import ShapeSpec


def test_flex_semantic_promises_are_not_search_knobs_or_backend_agnostic():
    shape=ShapeSpec({'Q_LEN':129,'KV_LEN':129})
    options=flex_attention.kernel_options(shape,'causal',backend='TRITON',parameters={'BLOCK_M':64,'BLOCK_N':32,'num_warps':4,'num_stages':2})
    assert options['PRESCALE_QK'] is False and options['ROWS_GUARANTEED_SAFE'] is True
    with pytest.raises(ValueError):flex_attention.kernel_options(shape,'causal',backend='FLASH',parameters={'BLOCK_M':64})
    with pytest.raises(ValueError):flex_attention.kernel_options(shape,'causal',backend='TRITON',parameters={'ROWS_GUARANTEED_SAFE':False})
    with pytest.raises(ValueError):flex_attention.kernel_options(ShapeSpec({'Q_LEN':1,'KV_LEN':129}),'causal',backend='AUTO',parameters={})


def test_generated_config_requires_actual_emitted_constants_not_requested_kwargs():
    code='''@triton_heuristics.template(num_stages=2,num_warps=4)
def triton_flex(Q):
    BLOCK_M: tl.constexpr = 64
    BLOCK_N: tl.constexpr = 32
'''
    assert flex_attention.generated_configuration([code])==[{'BLOCK_M':64,'BLOCK_N':32,'num_warps':4,'num_stages':2}]
    assert flex_attention.generated_configuration(['requested = {"BLOCK_M":64}'])==[]


def test_compatibility_path_exposes_actual_source_identity_and_same_documented_knobs():
    import ast
    from kernel_tuning_lab.production import native_sources as sources
    for profile in ('pytorch','pytorch213'):
        tree=ast.parse(sources.source(profile,'torch/nn/attention/flex_attention.py').read_text())
        options=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlexKernelOptions')
        fields={n.target.id for n in options.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
        assert set(flex_attention.KNOBS)|{'BACKEND','PRESCALE_QK','ROWS_GUARANTEED_SAFE','BLOCKS_ARE_CONTIGUOUS'}<=fields
    assert sources.pin('pytorch213')['revision']=='cf30153c4c131c8164ee7798e5022d810682e2cb'


def test_real_page_indices_preserve_ragged_lengths_and_do_not_form_contiguous_kv():
    layout=sglang_decode.paged_layout(4,129,16,'ragged',7)
    assert layout['seq_lens']==[1,15,17,129]
    assert layout['kv_indptr'][-1]==162
    indices=layout['kv_indices'];assert len(indices)==len(set(indices))
    assert max(indices)<layout['physical_slots']
    assert indices!=list(range(len(indices)))
    other=sglang_decode.paged_layout(4,129,16,'ragged',7)
    assert layout==other


def test_pure_python_page_counter_matches_original_numpy_without_changing_inputs():
    np=pytest.importorskip('numpy')
    from kernel_tuning_lab.production.inputs import _counter,counter_scalar
    for seed in (0,17,2**32-1):
        expected=_counter(64,seed).tolist()
        assert [counter_scalar(i,seed) for i in range(64)]==expected


def test_native_jit_loading_without_a_device_is_not_gpu_execution():
    import importlib.util,subprocess,sys,os
    if not importlib.util.find_spec('torch') or not importlib.util.find_spec('triton'):pytest.skip('optional native source-loader dependencies')
    script='''import torch
assert not torch.cuda.is_initialized()
from kernel_tuning_lab.production.sglang_decode import native_api
decode,scheduler=native_api()
assert hasattr(decode,'_fwd_kernel_stage1') and hasattr(scheduler,'get_num_kv_splits_triton')
assert not torch.cuda.is_initialized()
print('native JIT definitions loaded on CPU, no CUDA device initialized')
'''
    result=subprocess.run([sys.executable,'-c',script],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stderr
