"""Pinned host/template evidence only. No CUDA or native extension is loaded."""
from kernel_tuning_lab.production import flute,vllm_moe,gemlite
from kernel_tuning_lab.catalog import AssetCatalog
from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec,KernelConfig
from kernel_tuning_lab.assets import SpaceContext
import pytest


def test_real_flute_codegen_yields_structured_templates_with_packing_identity():
    rows=flute.candidates(3)
    assert len(rows)==36 and {r.parameters['Threads'] for r in rows}=={128,256}
    assert {r.parameters['TileP'] for r in rows}=={32,64}
    assert all(r.compiled is None for r in rows)
    assert len(flute.candidates(4))==144
    assert flute.workspace_bytes(128)>256*1024**2
    assert all(flute.template_id(r.parameters)==int(r.native_id) for r in rows)


def test_native_candidate_identity_does_not_change_when_compilation_is_learned():
    from dataclasses import replace
    from kernel_tuning_lab.production.contracts import NativeCandidate
    params={'tile':32};candidate=NativeCandidate('template0',params,'fixture',preparation={'layout':['a']})
    identity=candidate.id;params['tile']=64
    assert candidate.id==identity and replace(candidate,compiled=True).id==identity
    with pytest.raises(TypeError):candidate.parameters['tile']=64


def test_int3_packing_constraints_are_not_just_template_ids():
    h=HardwareSpec('planning fixture',sm_count=128)
    o=OperatorSpec('fixture',parameters={'bits':3,'group_size':128})
    row=next(r for r in flute.candidates() if r.parameters['TileP']==64)
    ctx=SpaceContext(h,o,ShapeSpec({'M':128,'N':512,'K':2048}))
    assert 'packing' in flute.hard_support(row.config,ctx)
    ctx=SpaceContext(h,o,ShapeSpec({'M':128,'N':2048,'K':2048}))
    assert 'TileP=32' in flute.hard_support(row.config,ctx)
    packed=next(r for r in flute.candidates() if r.parameters['TileP']==32)
    assert flute.hard_support(packed.config,ctx) is None


def test_original_int3_packer_support_is_separate_from_source_and_compiled_templates():
    from kernel_tuning_lab.production import native_sources
    namespace={}
    native_sources.host_functions('flute','flute/utils.py',{'_pack_3bit'},namespace)
    with pytest.raises(NotImplementedError):namespace['_pack_3bit'](None,tile_P=64)
    h=HardwareSpec('CPU planning',sm_count=132);op=OperatorSpec('fixture',parameters={'bits':3,'group_size':128})
    ctx=SpaceContext(h,op,ShapeSpec({'M':128,'N':2048,'K':2048}))
    assert len(flute.candidates())==36 and any(c.parameters['TileP']==64 for c in flute.candidates())
    for row in flute.candidates():
        reason=flute.hard_support(row.config,ctx)
        if row.parameters['TileP']==64:assert reason and 'TileP=32' in reason


def test_native_moe_rule_nearest_table_override_and_order_survive_export(tmp_path):
    table={64:{'BLOCK_SIZE_M':64},16:{'BLOCK_SIZE_M':16}}
    before=vllm_moe.native_rule(40,8,256,128,2,table=table)
    assert before['BLOCK_SIZE_M']==64 # equal-distance native insertion-order tie
    file=tmp_path/'native.json';restored=vllm_moe.write_native_table(file,table)
    assert vllm_moe.native_rule(40,8,256,128,2,table=restored)==before
    assert vllm_moe.native_rule(40,8,256,128,2,table=table,override={'BLOCK_SIZE_M':32})=={'BLOCK_SIZE_M':32}
    with pytest.raises(ValueError):vllm_moe.write_native_table(file,{16:{'SPLIT_K':2}})


def test_all_native_assets_can_be_discovered_without_production_installations():
    c=AssetCatalog()
    for provider in (gemlite,flute,vllm_moe):provider.register_assets(c)
    assert len(c.list('operator'))==2 and len(c.list('kernel'))==9
    for asset in c.list():
        if hasattr(asset,'source_files'):
            names={__import__('pathlib').Path(name).name for name in asset.source_files}
            assert {'inputs.py','torch_context.py'}<=names


def test_moe_correctness_inputs_are_not_hidden_below_absolute_tolerance():
    # CPU reference/input-strength test, no native GPU execution claimed.
    import types
    torch=pytest.importorskip('torch');pytest.importorskip('numpy')
    shape=ShapeSpec({'M':17,'H':128,'I':256,'E':4,'TOP_K':2})
    op=OperatorSpec('fixture',dtype='float16',parameters={'route_distribution':'balanced'})
    arrays=vllm_moe.logical_arrays(shape,op.parameters,17)
    inputs={k:torch.from_numpy(v).to(dtype=torch.int32 if k=='topk_ids' else torch.float32 if k=='topk_weights' else torch.float16) for k,v in arrays.items()}
    result=vllm_moe.reference(types.SimpleNamespace(torch=torch,inputs=inputs,shape=shape,operator=op))['out']
    assert torch.isfinite(result).all()
    assert not torch.allclose(torch.zeros_like(result),result,rtol=.01,atol=.01)
    # First-generation /64 weights/activations indeed made the same small
    # workload insensitive. Retain this explicit negative control.
    old={k:(v/8 if k=='a' else v*(shape.get('H' if k=='w1' else 'I')**.5)/16 if k in ('w1','w2') else v) for k,v in inputs.items()}
    weak=vllm_moe.reference(types.SimpleNamespace(torch=torch,inputs=old,shape=shape,operator=op))['out']
    assert torch.allclose(torch.zeros_like(weak),weak,rtol=.01,atol=.01)


def test_flute_high_profile_changes_reduction_identity_without_changing_geometry_or_packing():
    original=flute.candidates();high=flute.candidates(accumulation_mode='kHigh')
    assert len(original)==len(high)==36
    for mixed,fp32 in zip(original,high):
        assert mixed.native_id==fp32.native_id and mixed.packing_key==fp32.packing_key
        assert mixed.id!=fp32.id
        assert {k:v for k,v in mixed.parameters.items() if k!='AccumulationMode'}=={k:v for k,v in fp32.parameters.items() if k!='AccumulationMode'}
        assert fp32.parameters['AccumulationMode']=='kHigh'
        assert flute.template_id(fp32.parameters)==int(fp32.native_id)


def test_flute_high_build_only_selects_original_high_dispatch_branch(tmp_path):
    import importlib.util
    from pathlib import Path
    from kernel_tuning_lab.production import native_sources
    script=Path(__file__).parents[1]/'scripts/build_flute_subset.py'
    spec=importlib.util.spec_from_file_location('flute_builder',script);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    source=tmp_path/'source';cutlass=tmp_path/'cutlass';(cutlass/'include/cute').mkdir(parents=True)
    (cutlass/'include/cute/tensor.hpp').write_text('// CPU build-contract fixture')
    for name in ['setup.py','flute/codegen_utils.py','flute/csrc/qgemm.cpp','flute/csrc/qgemm_kernel_raw_generated.cu','flute/utils.py','flute/csrc/config.hpp']:
        file=source/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(native_sources.source('flute',name).read_bytes())
    plan=builder.prepare(source,cutlass,tmp_path/'high',[6,8,31],accumulation='high')
    generated=(tmp_path/'high/flute/csrc/qgemm_kernel_raw_generated.cu').read_text()
    selected=[line for line in generated.splitlines() if line.lstrip().startswith('RUN_QGEMM(T,')]
    assert len(selected)==3 and all('kHigh, kStreamK' in line for line in selected)
    assert plan['accumulation_mode']=='kHigh' and plan['native_kernel_headers_modified'] is False
    assert (tmp_path/'high/flute/csrc/config.hpp').read_bytes()==(source/'flute/csrc/config.hpp').read_bytes()


def test_flute_resource_rejection_is_scoped_to_exact_template_requirement_and_device_limit():
    # Exact sizeof from the original CPU probe, not an empirical SMEM estimate.
    assert '119072' in flute.resource_support_reason(119072,101376)
    assert flute.resource_support_reason(119072,232448) is None
    assert flute.resource_support_reason(49440,101376) is None
    assert flute.resource_support_reason(119072,0) is None # unknown is not zero capacity
