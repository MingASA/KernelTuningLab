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


def test_legacy26_maps_only_native_options_and_rejects_unsupported_path_or_stage():
    shape=ShapeSpec({'Q_LEN':129,'KV_LEN':129});requested={'BLOCK_M':64,'BLOCK_N':32,'num_warps':4,'num_stages':3}
    options=flex_attention.kernel_options(shape,'causal',backend='TRITON',parameters=requested,source_version='2.6.0')
    assert options['FORCE_USE_FLEX_ATTENTION'] is True
    assert options['BLOCK_M']==64 and options['BLOCK_N']==32
    assert not set(options)&{'BACKEND','num_warps','num_stages'}
    assert flex_attention.kernel_options(shape,'dense',backend='AUTO',parameters={},source_version='2.6.0')=={
        'PRESCALE_QK':False,'ROWS_GUARANTEED_SAFE':True,'BLOCKS_ARE_CONTIGUOUS':False}
    with pytest.raises(ValueError,match='skips num_stages=2'):
        flex_attention.kernel_options(shape,'dense',backend='TRITON',parameters={**requested,'num_stages':2},source_version='2.6.0')
    for backend in ('FLASH','TRITON_DECODE'):
        with pytest.raises(ValueError,match='unavailable'):
            flex_attention.kernel_options(shape,'dense',backend=backend,parameters={},source_version='2.6.0')


def test_legacy26_stage_restriction_is_software_scoped_not_global():
    from kernel_tuning_lab.assets import SpaceContext
    from kernel_tuning_lab.models import HardwareSpec,OperatorSpec
    hardware=HardwareSpec('CPU fixture');operator=OperatorSpec('fixture');shape=ShapeSpec({'N':1})
    old=SpaceContext(hardware,operator,shape,environment={'torch':'2.6.0+cu124'})
    newer=SpaceContext(hardware,operator,shape,environment={'torch':'2.13.0+cu132'})
    assert flex_attention.resolve_launch_domains(old)=={'num_stages':(1,3)}
    assert flex_attention.resolve_launch_domains(newer)=={}


def test_legacy26_native_selector_body_force_flag_and_hook_restore():
    from types import SimpleNamespace
    from kernel_tuning_lab.production import native_sources as sources
    namespace={'sympy':SimpleNamespace(Lt=lambda a,b:a<b),
        'V':SimpleNamespace(graph=SimpleNamespace(sizevars=SimpleNamespace(evaluate_expr=bool)))}
    sources.host_functions('pytorch26','torch/_inductor/kernel/flex_attention.py',{'_use_flex_decoding'},namespace)
    query=SimpleNamespace(get_size=lambda:(1,4,32,64))
    selector=namespace['_use_flex_decoding']
    assert selector(query,{}) is True
    assert selector(query,{'FORCE_USE_FLEX_ATTENTION':True}) is False
    original=lambda query:(128,128,4,3);module=SimpleNamespace(_get_default_config_fwd=original)
    request={'BLOCK_M':64,'BLOCK_N':32,'num_warps':8,'num_stages':1}
    with flex_attention.legacy_fwd_configuration(None,module=module):
        assert module._get_default_config_fwd is original # AUTO never overridden
    with pytest.raises(RuntimeError,match='fixture compilation failure'):
        with flex_attention.legacy_fwd_configuration(request,module=module):
            assert module._get_default_config_fwd(query)==(64,32,8,1)
            raise RuntimeError('fixture compilation failure')
    assert module._get_default_config_fwd is original


def test_legacy26_installed_source_profile_is_exact_and_keeps_native_stage2_workaround():
    import ast
    from kernel_tuning_lab.production import native_sources as sources
    pin=sources.pin('pytorch26');assert pin['revision']=='2236df1770800ffea5697b11b0bb0d910b2e59e1'
    tree=ast.parse(sources.source('pytorch26','torch/_inductor/kernel/flex_attention.py').read_text())
    create=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='flex_attention')
    guards=[n for n in ast.walk(create) if isinstance(n,ast.If) and ast.unparse(n.test)=='num_stages == 2']
    assert guards and any(isinstance(child,ast.Continue) for child in guards[0].body)


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


def test_constexpr_disabled_pdl_dependency_hashing_keeps_original_body_and_no_global_patch(monkeypatch):
    pytest.importorskip('torch');triton=pytest.importorskip('triton')
    import triton.language as language
    for name in ('gdc_launch_dependents','gdc_wait'):monkeypatch.delattr(language.extra.cuda,name,raising=False)
    decode,_=sglang_decode.native_api()
    assert decode.__ktlab_missing_pdl_intrinsics__==('gdc_launch_dependents','gdc_wait')
    assert not hasattr(language.extra.cuda,'gdc_wait') # no installed/global mutation
    affected=[value for value in vars(decode).values() if hasattr(value,'src') and 'tl.extra.cuda.gdc_' in value.src]
    assert len(affected)>=2
    for function in affected:assert function.cache_key
