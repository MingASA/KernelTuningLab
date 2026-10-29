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
    assert flute.hard_support(row.config,ctx) is None


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
