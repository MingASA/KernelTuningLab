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


def test_runtime_source_verification_accepts_a_wheel_without_source_setup_py(tmp_path,monkeypatch):
    # Real pinned Python file bytes in an installed-wheel layout; optional
    # module execution is a sentinel, so this is not native import/GPU proof.
    from types import SimpleNamespace
    from kernel_tuning_lab.production import native_sources
    assert 'setup.py' in native_sources.manifest()['gemlite']['files']
    for name in native_sources.manifest()['gemlite']['files']:
        if name.startswith('gemlite/') and name.endswith('.py'):
            path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(native_sources.source('gemlite',name).read_bytes())
    assert not (tmp_path/'setup.py').exists()
    module=object();monkeypatch.setattr(gemlite.importlib.util,'find_spec',lambda name:SimpleNamespace(origin=str(tmp_path/'gemlite/__init__.py')))
    monkeypatch.setattr(gemlite.importlib,'import_module',lambda name:module)
    gemlite.native_api.cache_clear()
    try:
        result,identity=gemlite.native_api()
        assert result is module and 'setup.py' not in identity['source_git_blobs']
    finally:gemlite.native_api.cache_clear()


def test_optional_e8m0_registration_keeps_a16_maps_without_fabricating_a_dtype():
    import ast
    from enum import Enum
    from types import SimpleNamespace
    import pytest
    from kernel_tuning_lab.production import native_sources,gemlite_compat
    original=native_sources.source('gemlite','gemlite/dtypes.py').read_text()
    def execute(source):
        tree=ast.parse(source);torch_names={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='torch'}
        tl_names={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='tl'}
        torch=SimpleNamespace(**{name:object() for name in torch_names if name!='float8_e8m0fnu'})
        tl=SimpleNamespace(**{name:object() for name in tl_names});namespace={'torch':torch,'tl':tl,'Enum':Enum}
        nodes=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned-dtypes-host-only','exec'),namespace)
        return namespace,torch
    with pytest.raises(AttributeError,match='float8_e8m0fnu'):execute(original)
    result,torch=execute(gemlite_compat.optional_e8m0_source().decode())
    assert not hasattr(torch,'float8_e8m0fnu') and 19 not in result['DTYPE_TO_TORCH']
    assert result['DTYPE_TO_TORCH'][1] is torch.float16 and result['DTYPE_TO_TORCH'][2] is torch.bfloat16
    assert result['TORCH_TO_DTYPE'][torch.float16]==result['DType'].FP16


def test_raw_a16_profile_changes_only_optional_interop_decorators_not_function_bodies():
    import ast
    from kernel_tuning_lab.production import native_sources,gemlite_compat
    for relative in gemlite_compat.RAW_FILES:
        before=ast.parse(native_sources.source('gemlite',relative).read_text())
        after=ast.parse(gemlite_compat.raw_a16_source(relative).decode())
        functions=lambda tree:[(n.name,ast.dump(n.args),ast.dump(ast.Module(body=n.body,type_ignores=[]),include_attributes=False))
            for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
        assert functions(before)==functions(after)
        assert not any(ast.unparse(d.func if isinstance(d,ast.Call) else d) in ('torch.compile','torch.library.custom_op','torch.library.register_fake')
            for n in ast.walk(after) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for d in n.decorator_list)
