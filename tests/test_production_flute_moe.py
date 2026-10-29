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


def test_int3_packing_constraints_are_not_just_template_ids():
    h=HardwareSpec('planning fixture',sm_count=128)
    o=OperatorSpec('fixture',parameters={'bits':3,'group_size':128})
    row=next(r for r in flute.candidates() if r.parameters['TileP']==64)
    ctx=SpaceContext(h,o,ShapeSpec({'M':128,'N':512,'K':2048}))
    assert 'packing' in flute.hard_support(KernelConfig(row.parameters),ctx)
    ctx=SpaceContext(h,o,ShapeSpec({'M':128,'N':2048,'K':2048}))
    assert flute.hard_support(KernelConfig(row.parameters),ctx) is None


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
