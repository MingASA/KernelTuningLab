"""Fixed vLLM0.8.5 production Marlin, plus same-semantics controlled kernels.

Explicit plugin: --plugin kernel_tuning_lab.production.marlin, or plugins in a
research spec. No model download, serving imports, GPU init or global install.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

from ..sdk import (KernelDefinition,OperatorDefinition,ParameterSpec,SpaceDefinition,
                   ShapeSpec,ExecutionPlan,ExecutionStep)
from ..builtin_assets import quantized as controlled
from . import inputs

BINARY_SHA256='bf47ca452c7cf18cab47b651b29d9c1a703d32149e20139d92a472726aba6ccb'
WHEEL_SHA256='74bfe92953bee1269c1e1c27827bc156777751cdd6a3457ee8e27dd8ebf1e247'
UPSTREAM=Path(__file__).with_name('upstream')
DEFAULT_ATOMIC_OPT_IN=bool(int(os.environ.get('VLLM_MARLIN_USE_ATOMIC_ADD','0')))
REGISTERED_PROFILE=None


def component_profile():
    root=os.environ.get('KT_VLLM_COMPONENTS')
    if not root:return {'kind':'official_wheel','version':'0.8.5','binary_sha256':BINARY_SHA256,'wheel_sha256':WHEEL_SHA256}
    manifest=json.loads((Path(root)/'component-manifest.json').read_text())
    if manifest.get('version')!='0.8.5':raise RuntimeError('different native production version')
    if manifest.get('kind')=='native_source_subset':
        pins=json.loads((UPSTREAM/'NATIVE_SOURCES.json').read_text())
        actual={row['file']:row['sha256'] for row in manifest['source_files']}
        if manifest.get('source_archive_sha256')!=pins['source_archive_sha256'] or any(actual.get(row['file'])!=row['sha256'] for row in pins['files']):
            raise RuntimeError('native numerical sources differ from pinned upstream vLLM0.8.5')
        return {key:manifest.get(key) for key in ('kind','version','binary_sha256','required_torch',
            'source_archive_sha256','compiled_architectures','compiler','cuda_toolkit','registration')}
    if manifest.get('wheel_sha256')!=WHEEL_SHA256 or manifest.get('binary_sha256')!=BINARY_SHA256:
        raise RuntimeError('production component provenance differs from fixed official wheel')
    return {'kind':'official_wheel','version':'0.8.5','binary_sha256':BINARY_SHA256,'wheel_sha256':WHEEL_SHA256}


@functools.lru_cache(maxsize=4)
def _digest_file(path,size,mtime):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def binary_path():
    root=os.environ.get('KT_VLLM_COMPONENTS')
    if not root:raise RuntimeError('Set KT_VLLM_COMPONENTS to the private verified vLLM0.8.5 component directory')
    root=Path(root);profile=component_profile()
    if REGISTERED_PROFILE is not None and profile!=REGISTERED_PROFILE:raise RuntimeError('native component identity changed since asset registration')
    library=root/'vllm/_C.abi3.so';stat=library.stat()
    if _digest_file(str(library),stat.st_size,stat.st_mtime_ns)!=profile['binary_sha256']:
        raise RuntimeError('native binary changed; refuse stale measurement-cache identity')
    return library


def native_supports(hardware,operator,shape):
    if operator.dtype!='float16':return 'initial production cohort uses FP16'
    if not hardware.architecture.startswith('sm_') or not hardware.architecture[3:].isdigit() or int(hardware.architecture[3:])<80:return 'Marlin needs NVIDIA sm80+'
    if bool(int(os.environ.get('VLLM_MARLIN_USE_ATOMIC_ADD','0')))!=DEFAULT_ATOMIC_OPT_IN:return 'native selector environment changed since asset registration'
    profile=component_profile()
    if profile.get('compiled_architectures') and hardware.architecture not in profile['compiled_architectures']:return 'source build does not contain this architecture; retain other profiles for their supported GPUs'
    if shape.get('K')%128 or shape.get('N')%64:return 'no implicit production padding: K%128=N%64=0 required'
    try:
        if not importlib.metadata.version('torch').startswith('2.6.0'):return 'fixed vLLM0.8.5 binary requires the tested Torch2.6.0 ABI'
        binary_path()
    except (RuntimeError,FileNotFoundError,importlib.metadata.PackageNotFoundError) as exc:return str(exc)
    return None


@functools.lru_cache(maxsize=1)
def upstream_api():
    import torch
    spec=importlib.util.spec_from_file_location('ktlab_vllm085_scalar',UPSTREAM/'scalar_type.py')
    module=importlib.util.module_from_spec(spec)
    # dataclasses consult the defining module when resolving string annotations.
    import sys
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    namespace={'torch':torch,'USE_FP32_REDUCE_DEFAULT':True,
        'envs':SimpleNamespace(VLLM_MARLIN_USE_ATOMIC_ADD=False)}
    inputs.source_functions('custom_ops.py',{'gptq_marlin_gemm'},namespace)
    op=namespace['gptq_marlin_gemm']
    namespace['ops']=SimpleNamespace(gptq_marlin_gemm=op)
    inputs.source_functions('marlin_utils.py',{'should_use_atomic_add_reduce','apply_gptq_marlin_linear'},namespace)
    return namespace,module.scalar_types.uint4b8


def native_rule(shape,device,dtype,*,atomic_opt_in=False):
    namespace,_=upstream_api();prior=namespace['envs'].VLLM_MARLIN_USE_ATOMIC_ADD
    try:
        namespace['envs'].VLLM_MARLIN_USE_ATOMIC_ADD=atomic_opt_in
        use=namespace['should_use_atomic_add_reduce'](shape.get('M'),shape.get('N'),shape.get('K'),device,dtype)
    finally:namespace['envs'].VLLM_MARLIN_USE_ATOMIC_ADD=prior
    return {'USE_ATOMIC_ADD':bool(use),'USE_FP32_REDUCE':True}


def _build(ctx,*,mode,USE_ATOMIC_ADD=None,USE_FP32_REDUCE=True):
    torch=ctx.torch;library=binary_path();torch.ops.load_library(str(library));namespace,wtype=upstream_api()
    m,n,k=(ctx.shape.get(key) for key in ('M','N','K'));start=time.perf_counter()
    # A native, independent repacker verifies the host/upstream packed layout.
    repacked=torch.ops._C.gptq_marlin_repack(ctx.inputs['packed'],ctx.inputs['empty_perm'],k,n,4)
    if not torch.equal(repacked,ctx.inputs['marlin_packed']):raise ValueError('native repack differs from the independently prepared upstream Marlin layout')
    del repacked
    scratch=ctx.allocate_workspace('native_marlin_locks',(n//64*16,),'int32')
    trace={};native_call=namespace['gptq_marlin_gemm']
    def traced_call(*args,**kwargs):
        trace.update({key:kwargs.get(key,False) for key in ('use_atomic_add','use_fp32_reduce','has_zp','is_zp_float')})
        return native_call(*args,**kwargs)
    # Functions share this scope. Trace the actual upstream wrapper invocation,
    # without changing its selector or numerical kernel.
    namespace['ops']=SimpleNamespace(gptq_marlin_gemm=traced_call)
    atomic_opt_in=mode=='native_atomic_opt_in' or (mode=='native_default' and DEFAULT_ATOMIC_OPT_IN)
    def call():
        a=ctx.inputs['a'];b=ctx.inputs['marlin_packed'];scales=ctx.inputs['marlin_scales']
        empty=ctx.inputs['empty_g_idx'];perm=ctx.inputs['empty_perm']
        if mode!='explicit':
            prior=namespace['envs'].VLLM_MARLIN_USE_ATOMIC_ADD
            try:
                namespace['envs'].VLLM_MARLIN_USE_ATOMIC_ADD=atomic_opt_in
                result=namespace['apply_gptq_marlin_linear'](a,b,scales,empty,empty,perm,scratch,wtype,n,k,False,True)
            finally:namespace['envs'].VLLM_MARLIN_USE_ATOMIC_ADD=prior
        else:
            result=traced_call(a,b,scales,empty,empty,perm,scratch,wtype,m,n,k,True,
                has_zp=False,use_atomic_add=USE_ATOMIC_ADD,use_fp32_reduce=USE_FP32_REDUCE,is_zp_float=False)
        # The native ABI allocates C. Keep the platform's stable declared output;
        # the copy is timed for native default AND all explicit variants.
        ctx.outputs['out'].copy_(result)
    call();torch.cuda.synchronize()
    symbols=[];profile_status='unknown';profile_error=None;cpu_ops=[]
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
            call();torch.cuda.synchronize()
        symbols=sorted({event.name for event in profiler.events() if 'CUDA' in str(event.device_type)})[:64]
        cpu_ops=sorted({event.name for event in profiler.events() if any(x in event.name for x in ('zeros','empty','gptq_marlin'))})[:32]
        profile_status='observed' if any('marlin' in symbol.lower() for symbol in symbols) else 'no_marlin_symbol_observed'
    except Exception as exc:profile_error=type(exc).__name__+': '+str(exc)
    annotation={'backend':'vllm0.8.5/_C.gptq_marlin_gemm','mode':mode,
        'host_library_sha256':component_profile()['binary_sha256'],'native_profile':component_profile(),
        'host_library_container_bytes':library.stat().st_size,
        'logical_packing_verification':'native repack equals unchanged upstream CPU packing',
        'actual_native_call_flags':dict(trace),'profiler_status':profile_status,'kernel_symbols':symbols,
        'profile_error':profile_error,'observed_allocation_ops':cpu_ops,
        'effective_atomic_branch':'unknown; request flags alone do not prove internal per-partition branch',
        'compiler_identity':'library SHA + observed symbol is trace identity; per-kernel CUBIN and byte attribution unknown',
        'selection_scope':'upstream Python/C++ host selection per invocation/capture; not re-executed by Graph replay',
        'timed_scope':'whole native operator including mandatory initialization/reductions, then adapter output copy',
        'prepack_scope':'host packing in preparation; native repack cross-check and profiler outside latency',
        'build_prepack_and_trace_wall_ms':(time.perf_counter()-start)*1000}
    ctx.metadata.setdefault('experiment_annotations',{})['production']=annotation
    # Avoid replacing global native objects between different plans. The scope
    # itself is process-local; invocation is serial in the supervised v2 worker.
    def launch():
        namespace['ops']=SimpleNamespace(gptq_marlin_gemm=traced_call)
        call()
    return ExecutionPlan((ExecutionStep('native_marlin_then_output_copy',launch),))


def build_default(ctx):return _build(ctx,mode='native_default')
def build_opt_in(ctx):return _build(ctx,mode='native_atomic_opt_in')
def build_explicit(ctx,USE_ATOMIC_ADD,USE_FP32_REDUCE):
    return _build(ctx,mode='explicit',USE_ATOMIC_ADD=USE_ATOMIC_ADD,USE_FP32_REDUCE=USE_FP32_REDUCE)


def register_assets(catalog):
    global REGISTERED_PROFILE
    REGISTERED_PROFILE=component_profile()
    sources=(str(Path(__file__)),str(Path(__file__).with_name('inputs.py')),str(UPSTREAM/'NATIVE_SOURCES.json'),str(UPSTREAM/'SOURCES.json'),str(UPSTREAM/'marlin_utils.py'),str(UPSTREAM/'custom_ops.py'),
             str(UPSTREAM/'scalar_type.py'),str(UPSTREAM/'marlin_utils_test.py'),str(UPSTREAM/'quant_utils.py'))
    operator=OperatorDefinition('w4a16_gemm','2.0.0',('M','N','K'),('float16',),inputs.generate,controlled._reference,
        tolerances={'float16':(.02,.02)},parameters={'group_size':128},validate=inputs.validate,
        validation_shapes=(ShapeSpec({'M':1,'N':64,'K':128}),ShapeSpec({'M':3,'N':128,'K':256})),
        source_files=(*sources,str(Path(controlled.__file__))),
        metadata={'semantics':'C=A@((u4codes-8)*group128 scales), FP16 inputs/output, FP32 reference',
            'input_protocol':'host counter_u32_v1; cross-device actual bytes checked, not CUDA RNG seed equivalence',
            'layout_scope':'restricted canonical W4 K-words verified against GPTQ/Marlin packing; not arbitrary checkpoint import'})
    catalog.register(operator)
    for original,identifier in (('triton/w4a16_gemm','controlled/w4a16_fused'),('triton_pipeline/w4a16_gemm','controlled/w4a16_pipeline')):
        kernel=catalog.resolve('kernel',original+'@1.0.0')
        catalog.register(KernelDefinition(identifier,'1.0.0',operator.ref,kernel.build,kernel.space,runtime='cuda',
            dtypes=('float16',),supports=kernel.supports,source_files=(*kernel.source_files,*sources),
            metadata={**dict(kernel.metadata),'common_input_operator':operator.ref}))
    metadata={'backend':'vLLM','version':'0.8.5','native_profile':REGISTERED_PROFILE,
        'native_binary_sha256':REGISTERED_PROFILE['binary_sha256'],
        'license':'Apache-2.0','native_default_atomic_opt_in':DEFAULT_ATOMIC_OPT_IN,
        'native_candidates':'operator runtime reduction parameters; not tile, compiled-binary or Graph identities'}
    for identifier,build,space in (
        ('production/marlin_default',build_default,SpaceDefinition({},name='native_production_default')),
        ('production/marlin_atomic_opt_in',build_opt_in,SpaceDefinition({},name='native_production_atomic_opt_in')),
        ('production/marlin_explicit',build_explicit,SpaceDefinition({'USE_ATOMIC_ADD':ParameterSpec((False,True)),
            'USE_FP32_REDUCE':ParameterSpec((True,False))},name='native_marlin_reduction_options',
            candidates=tuple({'USE_ATOMIC_ADD':a,'USE_FP32_REDUCE':r} for a in (False,True) for r in (True,False))))):
        catalog.register(KernelDefinition(identifier,'1.0.0',operator.ref,build,space,runtime='cuda',dtypes=('float16',),
            supports=native_supports,source_files=sources,metadata=metadata))
