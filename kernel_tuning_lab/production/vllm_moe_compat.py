"""Pinned vLLM0.8.5 production MoE for the Torch2.6/CUDA12 deployment profile.

The numerical Triton definitions, complete expert wrapper, native selector and
CUDA alignment/activation/reduction remain upstream. Only imports/registration
unrelated to this unquantized path are omitted. No serving engine is imported.
"""
from __future__ import annotations
import ast
from contextlib import contextmanager
from functools import lru_cache,partial
import functools
import hashlib
import importlib.metadata
import json
from itertools import product
import logging
import os
from pathlib import Path
import sys
import threading
from types import ModuleType,SimpleNamespace
from typing import Any,Callable,Dict,List,Optional,Tuple
import zipfile

from ..sdk import KernelDefinition,ParameterSpec,SpaceDefinition,ExecutionPlan,ExecutionStep
from ..catalog import AssetCatalog
from ..inference_resources import ResourceContextLimit,COORDINATOR
from . import native_sources as sources,environment,vllm_moe as shared

VERSION='0.8.5'
REVISION='ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d'
WHEEL_SHA256='74bfe92953bee1269c1e1c27827bc156777751cdd6a3457ee8e27dd8ebf1e247'
BINARY_SHA256={
    'vllm/_C.abi3.so':'bf47ca452c7cf18cab47b651b29d9c1a703d32149e20139d92a472726aba6ccb',
    'vllm/_moe_C.abi3.so':'062e7390e0a71394333243475a1483980b282cc689c96088587135fccea6ff68'}
MAIN='vllm/model_executor/layers/fused_moe/fused_moe.py'
ALIGN='vllm/model_executor/layers/fused_moe/moe_align_block_size.py'
PACKAGE='vllm/model_executor/layers/fused_moe/__init__.py'
KNOBS={'BLOCK_SIZE_M':(16,32,64,128,256),'BLOCK_SIZE_N':(32,64,128,256),'BLOCK_SIZE_K':(64,128,256,32),
    'GROUP_SIZE_M':(1,16,32,64,8),'num_warps':(4,8),'num_stages':(2,3,4,5)}
_LOCK=threading.RLock()


def _sha(path):
    value=environment.file_identity(path)
    if value is None:raise FileNotFoundError('required native MoE component is absent: '+Path(path).name)
    return value


def prepare_components(wheel,output):
    """CPU-only extraction into a NEW private component directory; no install.

    This deliberately does not use the two-schema custom Marlin source build:
    that build has neither the original activation nor the MoE CUDA schemas.
    """
    wheel=Path(wheel);output=Path(output)
    if _sha(wheel)!=WHEEL_SHA256:raise ValueError('not the fixed official vLLM0.8.5 wheel')
    if output.exists():raise ValueError('preserve existing components; use a new private extraction directory')
    output.mkdir(parents=True,mode=0o700)
    manifest=sources.manifest()['vllm085'];records=[]
    with zipfile.ZipFile(wheel) as archive:
        names=[*BINARY_SHA256,*[name for name in manifest['files'] if name.endswith('.py') and name.startswith('vllm/')],
            *[name for name in archive.namelist() if name.startswith('vllm/model_executor/layers/fused_moe/configs/') and name.endswith('.json')]]
        for name in dict.fromkeys(names):
            target=output/name;target.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
            with archive.open(name) as stream,target.open('wb') as out:
                for block in iter(lambda:stream.read(4*1024**2),b''):out.write(block)
            target.chmod(0o600);digest=_sha(target)
            expected=BINARY_SHA256.get(name,manifest['files'].get(name,{}).get('sha256'))
            if expected is not None and digest!=expected:raise ValueError('fixed native component differs: '+name)
            records.append({'file':name,'bytes':target.stat().st_size,'sha256':digest})
    (output/'LICENSE').write_bytes(sources.source('vllm085','LICENSE').read_bytes())
    result={'kind':'official_wheel_moe_subset','version':VERSION,'source_revision':REVISION,
        'required_torch':'2.6.0','required_torch_cuda_major':12,'required_cxx11_abi':False,
        'wheel_sha256':WHEEL_SHA256,'binaries':dict(BINARY_SHA256),'files':records,
        'scope':'original unquantized two-expert linear/SiLU/linear + alignment and weighted sum; router network excluded',
        'GPU_status':'not_executed_by_this_CPU_command','compiled_architectures':None,
        'identity_granularity':'extension file SHAs plus observed Triton compiled identities; physical CUDA helper kernel identities unknown'}
    (output/'component-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def component_root():
    value=os.environ.get('KT_VLLM085_MOE_COMPONENTS')
    if not value:raise RuntimeError('set KT_VLLM085_MOE_COMPONENTS to a private prepare_components output; do not use the Marlin-only source subset')
    return Path(value)


def execution_identity():
    root=component_root();profile=json.loads((root/'component-manifest.json').read_text())
    if (profile.get('kind')!='official_wheel_moe_subset' or profile.get('version')!=VERSION
            or profile.get('source_revision')!=REVISION or profile.get('wheel_sha256')!=WHEEL_SHA256
            or profile.get('binaries')!=BINARY_SHA256):raise RuntimeError('native MoE component provenance mismatch')
    files={}
    for name,expected in BINARY_SHA256.items():
        digest=_sha(root/name)
        if digest!=expected:raise RuntimeError('native MoE binary differs from the fixed official wheel: '+name)
        files[name]=digest
    for name,pin in sources.manifest()['vllm085']['files'].items():
        if name.endswith('.py') and name.startswith('vllm/'):
            digest=_sha(root/name)
            if digest!=pin['sha256']:raise RuntimeError('native MoE source differs: '+name)
            files[name]=digest
    tables={str(path.relative_to(root)):_sha(path) for path in sorted((root/Path(MAIN).parent/'configs').glob('*.json'))}
    versions={}
    for package in ('torch','triton','numpy'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    return {'backend':'vLLM0.8.5 unquantized production MoE compatibility profile','version':VERSION,'source_revision':REVISION,
        'wheel_sha256':WHEEL_SHA256,'installed_artifacts_sha256':files,'native_config_tables_sha256':tables,
        'versions':versions,'component_manifest_sha256':_sha(root/'component-manifest.json'),
        'native_environment':{name:os.environ.get(name) for name in ('VLLM_FUSED_MOE_CHUNK_SIZE','VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON')}}


def support(hardware,operator,shape):
    try:
        if importlib.metadata.version('torch').split('+')[0]!='2.6.0':return 'fixed vLLM0.8.5 CUDA ABI requires the private Torch2.6.0 profile'
        execution_identity()
    except (RuntimeError,FileNotFoundError,importlib.metadata.PackageNotFoundError) as error:return str(error)
    if not hardware.architecture.startswith('sm_'):return 'this fixed native component is NVIDIA CUDA, not a cross-platform profile'
    if operator.dtype=='bfloat16' and hardware.architecture[3:].isdigit() and int(hardware.architecture[3:])<80:
        return 'BF16 Tensor Core path requires NVIDIA sm80+'
    return None


def _module(relative,names,namespace,name,*,strip_config_import=False):
    path=sources.source('vllm085',relative);tree=ast.parse(path.read_text())
    chosen=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in chosen}!=set(names):raise RuntimeError('expected original native definitions are absent')
    if strip_config_import:
        class Imports(ast.NodeTransformer):
            def visit_ImportFrom(self,node):
                if node.module=='vllm.model_executor.layers.fused_moe':return None
                return node
        chosen=[Imports().visit(node) for node in chosen]
    module=ModuleType(name);module.__file__=str(path);module.__dict__.update(namespace);sys.modules[name]=module
    exec(compile(ast.fix_missing_locations(ast.Module(body=chosen,type_ignores=[])),str(path),'exec'),module.__dict__)
    return module


def native_environment():
    """Execute the exact two upstream environment getter expressions only."""
    path=sources.source('vllm085','vllm/envs.py');tree=ast.parse(path.read_text());wanted={'VLLM_FUSED_MOE_CHUNK_SIZE','VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON'}
    entries={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Dict):
            for key,value in zip(node.keys,node.values):
                if isinstance(key,ast.Constant) and key.value in wanted:entries[key.value]=value
    if set(entries)!=wanted:raise RuntimeError('native environment getter definitions changed')
    result={key:eval(compile(ast.Expression(body=value),str(path),'eval'),{'os':os})() for key,value in entries.items()}
    if type(result['VLLM_FUSED_MOE_CHUNK_SIZE']) is not int or result['VLLM_FUSED_MOE_CHUNK_SIZE']<1:
        raise ValueError('native MoE chunk size must be positive')
    return SimpleNamespace(**result)


def _source_api(torch,triton,tl,root,ops,platform):
    typing={'Any':Any,'Callable':Callable,'Dict':Dict,'List':List,'Optional':Optional,'Tuple':Tuple}
    env=native_environment();namespace={**typing,'torch':torch,'triton':triton,'tl':tl,'functools':functools,'os':os,'json':json,
        'envs':env,'current_platform':platform,'ops':ops,'logger':logging.getLogger(__name__)}
    package=_module(PACKAGE,{'get_config','override_config'},{**namespace,'_config':None,'contextmanager':contextmanager},'_ktlab_vllm085_config')
    align_path=sources.source('vllm085',ALIGN);align_names={node.name for node in ast.parse(align_path.read_text()).body if isinstance(node,ast.FunctionDef)}
    align=_module(ALIGN,align_names,{**namespace,'round_up':lambda value,multiple:((value+multiple-1)//multiple)*multiple},'_ktlab_vllm085_align')
    names={'write_zeros_to_output','fused_moe_kernel','invoke_fused_moe_kernel','get_config_file_name','get_moe_configs',
        'get_default_config','try_get_optimal_moe_config','get_config_dtype_str','moe_kernel_prepare_input','fused_experts_impl'}
    module=_module(MAIN,names,{**namespace,'get_config':package.get_config,'moe_align_block_size':align.moe_align_block_size},
        '_ktlab_vllm085_fused_moe',strip_config_import=True)
    # Original get_moe_configs derives its private configuration folder from
    # the original file location. This is a verified relocated source file.
    module.__file__=str(root/MAIN)
    return package,module


@lru_cache(maxsize=1)
def native_api():
    import torch,triton,triton.language as tl
    if torch.__version__.split('+')[0]!='2.6.0' or torch.version.cuda is None or torch.version.cuda.split('.')[0]!='12':
        raise RuntimeError('fixed official native MoE component needs Torch2.6/CUDA12 ABI; create a private compatible environment')
    if bool(torch._C._GLIBCXX_USE_CXX11_ABI):raise RuntimeError('fixed official native MoE component requires the original CXX11_ABI=0 profile')
    identity=execution_identity();root=component_root()
    names={Path(name).name:digest for name,digest in BINARY_SHA256.items()}
    for library in torch.ops.loaded_libraries:
        path=Path(library)
        if path.name in names and _sha(path)!=names[path.name]:raise RuntimeError('conflicting native vLLM version already loaded; use a fresh isolated worker')
    for name in BINARY_SHA256:torch.ops.load_library(str(root/name))
    schema_names={'activation':('_C','silu_and_mul'),'alignment':('_moe_C','moe_align_block_size'),'reduction':('_moe_C','moe_sum')}
    schemas={key:str(getattr(getattr(torch.ops,namespace),symbol).default._schema) for key,(namespace,symbol) in schema_names.items()}
    # These original wrappers call the real _moe_C registrations, never Torch
    # arithmetic substitutes. Activation is called directly by upstream impl.
    namespace={'torch':torch}
    ops=_module('vllm/_custom_ops.py',{'moe_sum','moe_align_block_size','sgl_moe_align_block_size'},namespace,'_ktlab_vllm085_ops')
    platform=SimpleNamespace(get_device_name=lambda:torch.cuda.get_device_name(0),is_rocm=lambda:False,is_cuda=lambda:True)
    package,module=_source_api(torch,triton,tl,root,ops,platform)
    identity.update(extraction='unchanged selected definitions; only original local get_config import supplied from its original private module',
        CUDA_helpers={'activation':'torch.ops._C.silu_and_mul','alignment':'torch.ops._moe_C.moe_align_block_size','reduction':'torch.ops._moe_C.moe_sum'},
        native_registered_schemas=schemas)
    return package,module,identity


def native_rule(m,e,n,k,topk,*,table=None,override=None):
    namespace={'get_config':lambda:override,'get_moe_configs':lambda *a:table}
    sources.host_functions('vllm085',MAIN,{'get_default_config','try_get_optimal_moe_config'},namespace,
        remove_imports=('vllm.model_executor.layers.fused_moe',))
    return namespace['try_get_optimal_moe_config']((e,2*n,k),(e,k,n),topk,None,m,False,None)


def source_benchmark_domains():
    namespace={'current_platform':SimpleNamespace(is_rocm=lambda:False),'product':product}
    sources.host_functions('vllm085','benchmarks/kernels/benchmark_moe.py',{'get_configs_compute_bound'},namespace)
    rows=namespace['get_configs_compute_bound'](True,None)
    rows.extend((native_rule(1,4,256,128,2),native_rule(17,4,256,128,2)))
    return {name:tuple(dict.fromkeys(row[name] for row in rows if name in row)) for name in KNOBS}


def read_native_table(path):
    table=json.loads(Path(path).read_text());result={}
    for key,configuration in table.items():
        if str(int(key))!=key or int(key)<1:raise ValueError('native MoE table requires positive integer token-count keys')
        if not isinstance(configuration,dict) or set(configuration)-set(KNOBS):raise ValueError('configuration contains fields outside this unquantized vLLM0.8.5 path; SPLIT_K is not a launch parameter here')
        if not {'BLOCK_SIZE_M','BLOCK_SIZE_N','BLOCK_SIZE_K','GROUP_SIZE_M'}<=set(configuration):raise ValueError('native unquantized table lacks required block/group fields')
        result[int(key)]=dict(configuration)
    if not result:raise ValueError('native table is empty')
    return result


def write_native_table(path,table):
    Path(path).write_text(json.dumps({str(key):dict(value) for key,value in table.items()},indent=2)+'\n')
    return read_native_table(path)


def native_temporary_bytes(shape,dtype,configuration,chunk_size):
    m=min(shape.get('M'),chunk_size);h=shape.get('H');i=shape.get('I');topk=shape.get('TOP_K');experts=shape.get('E');width=2
    cache13=m*topk*max(2*i,h)*width;cache2=m*topk*i*width;out=shape.get('M')*h*width
    padded=m*topk+experts*(configuration['BLOCK_SIZE_M']-1)
    alignment=(padded+(padded+configuration['BLOCK_SIZE_M']-1)//configuration['BLOCK_SIZE_M']+1)*4
    return {'cache13_bytes':cache13,'activation_cache_bytes':cache2,'native_output_bytes':out,'alignment_bytes':alignment,
        'known_buffer_bytes_per_full_chunk':cache13+cache2+out+alignment,'total_peak_is_known':False,
        'scope':'source-derived first/full-chunk buffers; native dynamic allocations, not ctx-owned workspace; transient alignment, allocator and Graph capture may retain more'}


class _ActualLaunchTrace(shared._KernelTrace):
    def __getitem__(self,grid):
        original=super().__getitem__(grid)
        def launch(*args,**kwargs):
            compiled=original(*args,**kwargs);slot=str(self.index-1)
            actual=self.ctx.metadata['experiment_annotations']['actual_launch_parameters'][slot]
            metadata=getattr(compiled,'metadata',None)
            for name in ('num_warps','num_stages'):
                value=getattr(metadata,name,None)
                actual[name]=value
            actual['launch_option_evidence']='returned compiled.metadata; missing values remain unknown'
            return compiled
        return launch


def build(ctx,*,mode,**configuration):
    package,module,identity=native_api();torch=ctx.torch;trace=_ActualLaunchTrace(module.fused_moe_kernel,ctx);inputs=ctx.inputs
    env=native_environment();chunk=env.VLLM_FUSED_MOE_CHUNK_SIZE
    expected_launches=2*((ctx.shape.get('M')+chunk-1)//chunk)
    annotations=ctx.metadata.setdefault('experiment_annotations',{})
    annotations.update(backend='vLLM',version=VERSION,source_identity=identity,mode=mode,
        requested_configuration=dict(configuration) if mode=='explicit' else {'selection':'original private config table / original default'},
        scope='original fused_experts_impl: config-dependent token alignment, gate/up GEMM, native SiLU/multiply, down GEMM, weighted sum, output copy; router network excluded',
        stage_parameter_contract='one original shared configuration for both GEMMs, no independent-stage extension',
        native_chunk_size=chunk,expected_triton_gemm_launches=expected_launches)
    def invoke():
        with _LOCK:
            old_config=package.get_config();old_kernel=module.fused_moe_kernel
            try:
                with package.override_config(configuration if mode=='explicit' else old_config):
                    selected=module.try_get_optimal_moe_config(tuple(inputs['w1'].shape),tuple(inputs['w2'].shape),ctx.shape.get('TOP_K'),None,min(ctx.shape.get('M'),chunk))
                    annotations['native_selected_configuration']=dict(selected)
                    memory=native_temporary_bytes(ctx.shape,ctx.operator.dtype,selected,chunk);annotations['native_temporary_allocations']=memory
                    if memory['known_buffer_bytes_per_full_chunk']>ctx.workspace_memory_limit_bytes:
                        raise ResourceContextLimit('known native MoE temporary buffers exceed this experiment workspace context')
                    COORDINATOR.check_known_allocation(ctx.hardware,memory['known_buffer_bytes_per_full_chunk'],ctx.workspace_bytes)
                    module.fused_moe_kernel=trace;trace.index=0
                    result=module.fused_experts_impl(inputs['a'],inputs['w1'],inputs['w2'],inputs['topk_weights'],inputs['topk_ids'],
                        inplace=False,activation='silu',apply_router_weight_on_input=False)
                    if trace.index!=expected_launches:raise RuntimeError('native path did not expose the expected two-GEMM-per-chunk trace')
                    ctx.outputs['out'].copy_(result)
            finally:
                module.fused_moe_kernel=old_kernel;package._config=old_config
    return ExecutionPlan((ExecutionStep('native_vllm085_full_expert_then_output_copy',invoke),))


def register_assets(catalog):
    # Reuse the exact same mathematical/reference asset as0.30. A separate
    # Kernel identity preserves the production-version/ABI distinction.
    temporary=AssetCatalog();shared.register_assets(temporary)
    operator=temporary.resolve('operator','production/routed_moe_silu@1.0.0');catalog.register(operator)
    src=sources.declaration_sources(__file__,shared.__file__,*[str(sources.source('vllm085',name)) for name in sources.manifest()['vllm085']['files']])
    for mode in ('native','explicit'):
        space=SpaceDefinition({key:ParameterSpec(values) for key,values in KNOBS.items()},name='original vLLM0.8.5 shared two-stage launch knobs') if mode=='explicit' else SpaceDefinition({},name='original vLLM0.8.5 table / default selector')
        catalog.register(KernelDefinition('production/vllm085_moe_'+mode,'1.0.0',operator.ref,partial(build,mode=mode),space,
            runtime='cuda',supports=support,source_files=src,execution_identity=execution_identity,
            metadata={'pin':sources.pin('vllm085'),'compatibility':'fixed Torch2.6/CUDA12 ABI; no global/shared upgrades',
                'native_path':'unchanged selected vLLM0.8.5 definitions + official _C/_moe_C binaries',
                'unsupported':'quantized expert weights, router network, distributed/EP, production serving backend dispatch',
                'space_scope':'factorized union of original CUDA benchmark domains and native default scalars; 1920 source benchmark proposals, 3200 declared combinations; not a claim all integer launch parameters are represented',
                'allocation_scope':'original per-call allocations/initialization/reduction retained, known sizes separately reported'}))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wheel',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=prepare_components(args.wheel,args.output)
    print(json.dumps({'status':'components_prepared_CPU_only','version':result['version'],'files':len(result['files']),'GPU_execution':False}))
