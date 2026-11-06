"""CPU provenance/layout/selector checks, never production GPU performance claims."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from kernel_tuning_lab.catalog import AssetCatalog,get_catalog
from kernel_tuning_lab.production import marlin


def test_production_plugin_is_cpu_discoverable_and_keeps_distinct_spaces():
    from kernel_tuning_lab.plugins import load_plugin
    catalog=AssetCatalog();catalog.register_many(get_catalog().list())
    load_plugin('kernel_tuning_lab.production.marlin',catalog)
    fixed=catalog.resolve('kernel','production/marlin_default@1.0.0')
    native=catalog.resolve('kernel','production/marlin_explicit@1.0.0')
    controlled=catalog.resolve('kernel','controlled/w4a16_fused@1.0.0')
    assert not fixed.space.parameters and len(native.space.parameters)==2
    assert len(controlled.space.parameters)==7
    assert fixed.operator==native.operator=='w4a16_gemm@2.0.0'
    assert catalog.resolve('operator','w4a16_gemm@1.0.0').parameters['group_size']==32
    assert fixed.metadata['native_binary_sha256']==marlin.BINARY_SHA256


def test_pinned_extractor_rejects_unverified_wheel_without_writing(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_marlin_components.py'
    spec=importlib.util.spec_from_file_location('marlin_component_fixture',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    bad=tmp_path/'wrong.whl';bad.write_bytes(b'unit fixture only')
    with pytest.raises(ValueError,match='pinned official'):module.extract(bad,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_portable_packing_matches_independent_bitstream_and_pinned_upstream():
    np=pytest.importorskip('numpy');torch=pytest.importorskip('torch')
    from kernel_tuning_lab.production.inputs import portable_arrays,upstream_cpu
    from kernel_tuning_lab.quantization import pack_bits,unpack_bits
    a,codes,scales,packed=portable_arrays(3,64,128,7)
    assert packed.tolist()==[[int(value) if int(value)<2**31 else int(value)-2**32 for value in row] for row in pack_bits(codes.tolist(),4)]
    assert unpack_bits(packed.tolist(),4,128)==codes.tolist()
    assert all(np.array_equal(x,y) for x,y in zip(portable_arrays(3,64,128,7),(a,codes,scales,packed)))
    helpers=upstream_cpu();native=helpers['marlin_weights'](torch.from_numpy(codes),128,64,4,helpers['get_weight_perm'](4))
    assert native.shape==(8,128) and native.dtype==torch.int32
    assert not torch.cuda.is_initialized()


def test_native_selector_executes_pinned_function_without_a_serving_import(monkeypatch):
    torch=pytest.importorskip('torch')
    from kernel_tuning_lab.models import ShapeSpec
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(9,0))
    shape=lambda n,k:ShapeSpec({'M':4,'N':n,'K':k})
    for n,k,expected in ((512,2048,True),(2048,2048,False),(512,1024,False)):
        assert marlin.native_rule(shape(n,k),torch.device('cuda'),torch.float16,atomic_opt_in=True)['USE_ATOMIC_ADD']==expected
        assert marlin.native_rule(shape(n,k),torch.device('cuda'),torch.float16)['USE_ATOMIC_ADD'] is False
    namespace,_=marlin.upstream_api()
    assert namespace['should_use_atomic_add_reduce'].__code__.co_filename.endswith('production/upstream/marlin_utils.py')
    assert not namespace['envs'].VLLM_MARLIN_USE_ATOMIC_ADD


def test_native_source_profile_cache_identity_is_scoped_to_verified_sources_and_actual_binary(tmp_path,monkeypatch):
    # These fixture bytes are never loaded as a GPU library.
    pins=json.loads((marlin.UPSTREAM/'NATIVE_SOURCES.json').read_text());root=tmp_path/'components';library=root/'vllm/_C.abi3.so';library.parent.mkdir(parents=True);library.write_bytes(b'CPU identity fixture only')
    manifest={'version':'0.8.5','kind':'native_source_subset','source_archive_sha256':pins['source_archive_sha256'],
        'source_files':pins['files'],'binary_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
        'required_torch':'2.6.0','compiled_architectures':['sm_89'],'compiler':'fixture, not a real build'}
    (root/'component-manifest.json').write_text(json.dumps(manifest));monkeypatch.setenv('KT_VLLM_COMPONENTS',str(root));monkeypatch.setattr(marlin,'REGISTERED_PROFILE',None)
    profile=marlin.component_profile();assert profile['binary_sha256']==manifest['binary_sha256'] and profile['compiled_architectures']==['sm_89']
    monkeypatch.setattr(marlin,'REGISTERED_PROFILE',profile);assert marlin.binary_path()==library
    library.write_bytes(b'changed CPU fixture')
    with pytest.raises(RuntimeError,match='stale'):marlin.binary_path()
    manifest['source_files']=[{**item,'sha256':'bad'} for item in pins['files']];(root/'component-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='numerical sources'):marlin.component_profile()
