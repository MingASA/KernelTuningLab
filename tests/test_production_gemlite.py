"""Original host rules only: these are not CUDA compilation/GPU claims."""
from kernel_tuning_lab.production import gemlite
from kernel_tuning_lab.catalog import AssetCatalog


def test_cpu_declarations_cover_native_families_and_two_bit_semantics_are_separate_inputs():
    catalog=AssetCatalog();gemlite.register_assets(catalog)
    assert len(catalog.list('kernel'))==5
    assert len(gemlite.declared_configs('GEMM'))==2000
    assert len(gemlite.declared_configs('GEMM_SPLITK'))==7500
    assert gemlite.native_family(1,2)=='GEMV_REVSPLITK'
    assert gemlite.native_family(32,4)=='GEMM_SPLITK'
    assert gemlite.native_family(64,4)=='GEMM'


def test_original_pruner_rewrites_and_deduplicates_instead_of_only_filtering():
    requested=[{'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':256,'GROUP_SIZE_M':8,'A_load_order':0,'num_warps':4,'num_stages':5},
        {'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':512,'GROUP_SIZE_M':8,'A_load_order':0,'num_warps':4,'num_stages':5}]
    result=gemlite.audit_pruner('GEMM',requested,m=1,n=128,k=256,bits=4,group_size=128,shared_memory_bytes=99*1024)
    assert len(result['effective'])==1
    assert result['effective'][0]['BLOCK_SIZE_M']==16 and result['effective'][0]['BLOCK_SIZE_K']==128
    assert result['requested']==requested and result['effective'][0]['num_stages']<5
    assert len(result['request_mappings'])==2
    assert result['request_mappings'][0]['normalized']==result['request_mappings'][1]['normalized']


def test_unknown_resource_cannot_be_treated_as_zero_capacity():
    import pytest
    with pytest.raises(ValueError,match='known resource'):
        gemlite.audit_pruner('GEMM',[],m=1,n=128,k=256,bits=2,group_size=128,shared_memory_bytes=0)
