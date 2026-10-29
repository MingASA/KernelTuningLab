"""CPU-safe installed source/binary identity before any measurement-cache lookup."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
from ..models import stable_id
from . import native_sources as sources


@lru_cache(maxsize=2048)
def _digest(filename,size,mtime_ns,ctime_ns):
    result=hashlib.sha256()
    with Path(filename).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):result.update(block)
    return result.hexdigest()


def file_identity(path):
    path=Path(path)
    if not path.is_file():return None
    stat=path.stat();return _digest(str(path.resolve()),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def _snapshot(value):
    if isinstance(value,dict):return {str(k):_snapshot(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_snapshot(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if hasattr(value,'kwargs') and hasattr(value,'num_warps'):
        return {'kwargs':_snapshot(value.kwargs),'num_warps':value.num_warps,'num_stages':value.num_stages}
    raise TypeError('opaque native selector state requires an explicit identity hook, not object repr/address')


def installed_identity(backend,package,*,native_default=False):
    distribution={'flute':'flute-kernel','pytorch':'torch'}.get(backend,package)
    try:version=importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:return {'backend':backend,'availability':'missing optional dependency'}
    spec=importlib.util.find_spec(package) # top-level spec discovery does not import the GPU backend
    if spec is None or spec.origin is None:return {'backend':backend,'version':version,'availability':'no importable package root'}
    root=Path(spec.origin).parent.parent;profile=backend
    if backend=='pytorch' and version.split('+')[0]=='2.13.0':profile='pytorch213'
    pin=sources.manifest()[profile];files={}
    for relative in pin['files']:
        if relative.endswith(('.py','.pth','.json')):files[relative]=file_identity(root/relative)
    directory=root/package
    # Only our selected backend's immediate binary artifacts, not unrelated
    # model weights or other users' files. Paths never enter public identities.
    patterns={'flute':('_C*.so','data/*.pth'),
        'vllm':('_C_stable_libtorch*.so','_moe_C_stable_libtorch*.so','model_executor/layers/fused_moe/configs/*.json'),
        'gemlite':('*.json','triton_kernels/*.json')}.get(backend,())
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):files[str(path.relative_to(root))]=file_identity(path)
    extras={}
    if backend=='flute' and os.environ.get('KT_FLUTE_BUILD_MANIFEST'):
        extras['compiled_subset_manifest_sha256']=file_identity(os.environ['KT_FLUTE_BUILD_MANIFEST'])
    if native_default:
        module=sys.modules.get({'gemlite':'gemlite.core','flute':'flute.tune','vllm':'vllm.model_executor.layers.fused_moe'}.get(backend,''))
        if module is None:extras['native_selector_state']='not loaded'
        else:
            field={'gemlite':'GEMLITE_TRITON_CONFIG_CACHE','flute':'_TEMPLATES','vllm':'_config'}.get(backend)
            if field:extras['native_selector_state_sha256']=stable_id(_snapshot(getattr(module,field,None)))
    if backend=='vllm' and os.environ.get('VLLM_TUNED_CONFIG_FOLDER'):
        # Authorized user-owned configured tables, read-only. No directory
        # traversal outside the selected table folder or hidden recursive scan.
        folder=Path(os.environ['VLLM_TUNED_CONFIG_FOLDER'])
        extras['configured_table_sha256']={p.name:file_identity(p) for p in sorted(folder.glob('*.json'))}
    prefixes={'gemlite':('GEMLITE_',),'vllm':('VLLM_BATCH_INVARIANT','VLLM_TRITON_USE_TD','VLLM_FUSED_MOE_CHUNK_SIZE'),
        'flute':('FLUTE_ABLATIONS',),'pytorch':('TORCHINDUCTOR_',)}.get(backend,())
    # Include behavior-changing options without logging private paths/values.
    extras['runtime_option_sha256']=stable_id({k:v for k,v in os.environ.items() if any(k.startswith(p) for p in prefixes)})
    versions={}
    for name in ('torch','triton'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'backend':backend,'installed_version':version,'source_revision':pin['revision'],
        'installed_artifacts_sha256':files,'compiler_versions':versions,**extras}


def vendored_identity(backend):
    pin=sources.manifest()[backend];versions={}
    for name in ('torch','triton'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'source_revision':pin['revision'],'verified_sources':{name:file_identity(sources.source(backend,name)) for name in pin['files']},
        'compiler_versions':versions}
