"""Fixed upstream/native CPU contracts; GPU use is separately admitted/opt-in."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import os
from types import SimpleNamespace
import pytest

from kernel_tuning_lab.production import vllm_moe_compat as compat,vllm_moe,native_sources as sources
from kernel_tuning_lab.catalog import AssetCatalog
from kernel_tuning_lab.models import ShapeSpec


def test_compat_plugin_preserves_shared_operator_and_real_numeric_knobs():
    base=AssetCatalog();vllm_moe.register_assets(base)
    catalog=AssetCatalog();compat.register_assets(catalog)
    assert catalog.resolve('operator','production/routed_moe_silu@1.0.0').fingerprint==base.resolve('operator','production/routed_moe_silu@1.0.0').fingerprint
    assert {kernel.id for kernel in catalog.list('kernel')}=={'production/vllm085_moe_native','production/vllm085_moe_explicit'}
    assert 'SPLIT_K' not in compat.KNOBS
    assert compat.REVISION=='ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d'


def test_original_rule_default_override_nearest_table_order_and_native_table_roundtrip(tmp_path):
    columns={'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':8}
    assert compat.native_rule(1,4,256,128,2)=={'BLOCK_SIZE_M':16,'BLOCK_SIZE_N':32,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':1}
    assert compat.native_rule(17,4,256,128,2)=={'BLOCK_SIZE_M':64,**columns}
    table={64:{'BLOCK_SIZE_M':64,**columns},16:{'BLOCK_SIZE_M':16,**columns}}
    path=tmp_path/'native.json';restored=compat.write_native_table(path,table)
    assert list(restored)==[64,16] and compat.native_rule(40,4,256,128,2,table=restored)['BLOCK_SIZE_M']==64
    assert compat.native_rule(40,4,256,128,2,table=table,override={'BLOCK_SIZE_M':32,**columns})['BLOCK_SIZE_M']==32
    with pytest.raises(ValueError,match='SPLIT_K'):compat.write_native_table(path,{1:{**columns,'BLOCK_SIZE_M':16,'SPLIT_K':1}})


def test_declared_domains_match_original_cuda_benchmark_and_native_default_scalars():
    assert compat.source_benchmark_domains()==compat.KNOBS


def test_environment_uses_original_getter_defaults_and_explicit_changes(monkeypatch):
    monkeypatch.delenv('VLLM_FUSED_MOE_CHUNK_SIZE',raising=False)
    assert compat.native_environment().VLLM_FUSED_MOE_CHUNK_SIZE==32768
    monkeypatch.setenv('VLLM_FUSED_MOE_CHUNK_SIZE','16')
    assert compat.native_environment().VLLM_FUSED_MOE_CHUNK_SIZE==16
    monkeypatch.setenv('VLLM_FUSED_MOE_CHUNK_SIZE','0')
    with pytest.raises(ValueError,match='positive'):compat.native_environment()


def test_component_extraction_refuses_unpinned_wheel_without_install_or_gpu(tmp_path):
    wheel=tmp_path/'wrong.whl';wheel.write_bytes(b'untrusted')
    with pytest.raises(ValueError,match='fixed official'):compat.prepare_components(wheel,tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


def test_component_identity_rejects_changed_binary_before_any_runtime_load(tmp_path,monkeypatch):
    profile={'kind':'official_wheel_moe_subset','version':'0.8.5','source_revision':compat.REVISION,
        'wheel_sha256':compat.WHEEL_SHA256,'binaries':compat.BINARY_SHA256}
    (tmp_path/'component-manifest.json').write_text(json.dumps(profile))
    for name in compat.BINARY_SHA256:
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'not the native binary')
    monkeypatch.setenv('KT_VLLM085_MOE_COMPONENTS',str(tmp_path))
    with pytest.raises(RuntimeError,match='binary differs'):compat.execution_identity()


def test_native_workspace_estimate_is_source_derived_and_not_claimed_as_total_peak():
    shape=ShapeSpec({'M':17,'H':128,'I':256,'E':4,'TOP_K':2})
    config=compat.native_rule(17,4,256,128,2)
    result=compat.native_temporary_bytes(shape,'float16',config,32768)
    assert result['cache13_bytes']==17*2*512*2
    assert result['activation_cache_bytes']==17*2*256*2
    assert result['total_peak_is_known'] is False


def test_plan_invoke_dependency_wiring_is_checked_without_a_gpu(monkeypatch):
    # Host orchestration fixture only: native math is verified separately on GPU.
    from contextlib import contextmanager
    from kernel_tuning_lab.models import HardwareSpec,OperatorSpec
    shape=ShapeSpec({'M':17,'H':128,'I':256,'E':4,'TOP_K':2});calls=[];copied=[];profile={'BLOCK_SIZE_M':16,'BLOCK_SIZE_N':32,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':1,'num_warps':4,'num_stages':2}
    package=SimpleNamespace(_config=None)
    package.get_config=lambda:package._config
    @contextmanager
    def override(value):
        old=package._config;package._config=value
        try:yield
        finally:package._config=old
    package.override_config=override
    class Kernel:
        def __getitem__(self,grid):return lambda *args,**kwargs:SimpleNamespace(metadata=SimpleNamespace(num_warps=4,num_stages=2))
    module=SimpleNamespace(fused_moe_kernel=Kernel(),try_get_optimal_moe_config=lambda *args:profile)
    def native(*args,**kwargs):
        for index in range(2):module.fused_moe_kernel[(1,)](BLOCK_SIZE_M=16,num_warps=4,num_stages=2)
        return 'native-result-fixture'
    module.fused_experts_impl=native
    monkeypatch.setattr(compat,'native_api',lambda:(package,module,{'fixture':True}))
    monkeypatch.setattr(compat,'COORDINATOR',SimpleNamespace(check_known_allocation=lambda *args:calls.append('resource-check')))
    inputs={name:SimpleNamespace(shape=dims) for name,dims in {'a':(17,128),'w1':(4,512,128),'w2':(4,128,256),'topk_weights':(17,2),'topk_ids':(17,2)}.items()}
    ctx=SimpleNamespace(torch=SimpleNamespace(),shape=shape,hardware=HardwareSpec('CPU fixture',memory_bytes=1024**3),
        operator=OperatorSpec('fixture',dtype='float16'),inputs=inputs,outputs={'out':SimpleNamespace(copy_=copied.append)},
        metadata={},workspace_memory_limit_bytes=1024**2,workspace_bytes=0,record_compiled_kernel=lambda *args:calls.append(args[0]))
    plan=compat.build(ctx,mode='explicit',**profile);plan.steps[0].launch()
    assert calls==['resource-check','vllm_moe_gemm0','vllm_moe_gemm1']
    assert copied==['native-result-fixture'] and package.get_config() is None
    assert ctx.metadata['experiment_annotations']['native_selected_configuration']==profile


def test_original_jit_and_host_definitions_load_without_serving_import_or_cuda(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('triton')
    code='''import torch,triton,triton.language as tl,sys
from pathlib import Path
from types import SimpleNamespace
from kernel_tuning_lab.production.vllm_moe_compat import _source_api
assert not torch.cuda.is_initialized()
package,module=_source_api(torch,triton,tl,Path('.'),SimpleNamespace(),SimpleNamespace(get_device_name=lambda:'CPU fixture',is_rocm=lambda:False,is_cuda=lambda:True))
assert module.fused_moe_kernel.cache_key
assert 'vllm' not in sys.modules
assert not torch.cuda.is_initialized()
with package.override_config({'BLOCK_SIZE_M':16}):assert package.get_config()=={'BLOCK_SIZE_M':16}
assert package.get_config() is None
print('original native JIT/host definitions loaded CPU; no serving imports or CUDA initialization')
'''
    result=subprocess.run([sys.executable,'-c',code],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stderr
