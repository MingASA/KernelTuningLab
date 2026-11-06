"""Pinned real GemLite W4/W2, native expert choice and raw explicit launches."""
from __future__ import annotations
import copy
from functools import lru_cache,partial
import importlib,importlib.metadata,importlib.util
import math
from pathlib import Path
from types import SimpleNamespace,FunctionType
import time

from ..sdk import OperatorDefinition,KernelDefinition,ParameterSpec,SpaceDefinition,ShapeSpec,ExecutionPlan,ExecutionStep,UnsupportedPreparationError
from ..inference_resources import ResourceContextLimit
from ..models import stable_id
from . import native_sources as sources,suite_inputs,environment,gemlite_compat
from .contracts import ConfigurationMapping

VERSION='0.6.0.post2'
FAMILIES={'GEMM':'gemm_kernels','GEMM_SPLITK':'gemm_splitK_kernels',
    'GEMV':'gemv_kernels','GEMV_REVSPLITK':'gemv_revsplitK_kernels'}


class HostConfig:
    """Only upstream host-rule tests use this CPU declaration object."""
    def __init__(self,kwargs,num_warps=4,num_stages=2,pre_hook=None,**extra):
        self.kwargs=dict(kwargs);self.num_warps=num_warps;self.num_stages=num_stages;self.pre_hook=pre_hook
    def to_dict(self):return {**self.kwargs,'num_warps':self.num_warps,'num_stages':self.num_stages}


@lru_cache(maxsize=8)
def host_rules(family):
    module=FAMILIES[family];ns={'math':math,'copy':copy,'triton':SimpleNamespace(Config=HostConfig),
        'MATMUL_TYPE':family,'IS_HIP':False,'GEMLITE_TRITON_CONFIG_CACHE':{},
        'init_to_zero':lambda name:lambda args:args[name].zero_()}
    sources.host_functions('gemlite','gemlite/core.py',{'get_default_gemv','get_matmul_type'},ns)
    sources.host_functions('gemlite','gemlite/triton_kernels/utils.py',
        {'next_power_of_2','generate_autotune_lookup_v2','get_closest_m','estimate_shared_memory_per_block'},ns)
    ns.update(M_MAXVAL=4096,M_MAPPING=ns['generate_autotune_lookup_v2'](4096,mode='next'),BLOCK_QUANT_SIZE=128)
    sources.host_functions('gemlite','gemlite/triton_kernels/'+module+'.py',
        {'get_max_autotune_config_nvidia','kernel_config_pruner'},ns,remove_imports=('..core',))
    return ns


def declared_configs(family):return tuple(c.to_dict() for c in host_rules(family)['get_max_autotune_config_nvidia']())


def native_family(m,bits):return host_rules('GEMM')['get_matmul_type'](m,bits,False)


def audit_pruner(family,requested,*,m,n,k,bits,group_size,shared_memory_bytes,cache=None):
    if not shared_memory_bytes:raise ValueError('native pruner needs a known resource fact; unknown SMEM is not zero capacity')
    ns=dict(host_rules(family));ns['GEMLITE_TRITON_CONFIG_CACHE']=copy.deepcopy(cache or {})
    # Rebind functions to this per-call scope; no shared upstream config edits.
    for key,value in list(ns.items()):
        if isinstance(value,FunctionType):ns[key]=FunctionType(value.__code__,ns,key,value.__defaults__,value.__closure__)
    ns['get_gpu_shared_memory']=lambda:shared_memory_bytes
    configs=[HostConfig({k:v for k,v in p.items() if k not in ('num_warps','num_stages')},p['num_warps'],p['num_stages']) for p in requested]
    args={'M':m,'N':n,'K':k,'group_size':group_size,'elements_per_sample':32//bits,'type_id':bits,'a_sizeof':2,'b_sizeof':4}
    effective=list(ns['kernel_config_pruner'](configs,args,load_scales_as_block=False,channel_scale_mode=0))
    mappings=[]
    for index,config in enumerate(configs):
        single=list(ns['kernel_config_pruner']([config],args,load_scales_as_block=False,channel_scale_mode=0))
        mappings.append({'request_index':index,'requested':requested[index],
            'normalized':[p.to_dict() for p in single],
            'classification':'native pruning/canonicalization; not proof of GPU illegality'})
    return {'requested':requested,'effective':[p.to_dict() for p in effective],
        'request_mappings':mappings,
        'scope':'exact native pruner including canonicalization, soft performance preferences and scoped resource estimates; NOT platform hard exclusions',
        'native_cache_hit':family in (cache or {}),'input_resource_bytes':shared_memory_bytes}


def _support(h,o,s):
    if o.parameters.get('bits') not in (2,4):return 'this GemLite production path is W2/W4; W3 uses FLUTE with its own packing'
    try:
        if importlib.metadata.version('gemlite')!=VERSION:return 'install pinned GemLite'+VERSION+' in a private environment'
        parts=tuple(int(x) for x in importlib.metadata.version('triton').split('.')[:2])
        if parts<(3,6):return 'GemLite0.6.0.post2 requires Triton>=3.6; original shared Torch/Triton env must not be upgraded'
    except importlib.metadata.PackageNotFoundError:return 'optional GemLite/Triton dependency is not installed'
    return None


@lru_cache(maxsize=1)
def native_api():
    spec=importlib.util.find_spec('gemlite')
    if spec is None:raise RuntimeError('optional GemLite package is not installed')
    root=Path(spec.origin).parent.parent
    files=[name for name in sources.manifest()['gemlite']['files'] if name.startswith('gemlite/') and name.endswith('.py') and name!='gemlite/dtypes.py' and name not in gemlite_compat.RAW_FILES]
    identity=sources.native_component(root,'gemlite',files)
    identity['optional_dtype_registration']=gemlite_compat.verify_dtype_source(root)
    identity['native_A16_host_interop']={name:gemlite_compat.verify_runtime_source(root,name) for name in gemlite_compat.RAW_FILES}
    module=importlib.import_module('gemlite.core')
    return module,identity


def hard_config(config,ctx,*,family):
    if family in ('GEMM','GEMM_SPLITK') and config.get('BLOCK_SIZE_K',32)>ctx.operator.parameters['group_size']:
        return 'INT non-block-scales kernel requires BLOCK_SIZE_K<=semantic group_size'
    return None


def split_metadata_shape(n,k,group,configuration):
    """Bounds of the upstream INT kernel's unmasked group metadata loads.

    A/B tails are masked, but every split executes ceil(K/(BK*split))
    iterations and loads its scale, including wholly inactive K partitions.
    """
    bk=configuration['BLOCK_SIZE_K'];bn=configuration['BLOCK_SIZE_N'];split=configuration['SPLIT_K']
    padded_k=((k+bk*split-1)//(bk*split))*bk*split
    return ((padded_k+group-1)//group,((n+bn-1)//bn)*bn)


def pad_split_metadata(torch,scales,*,n,k,group,configuration):
    if tuple(scales.shape)!=(k//group,n):raise ValueError('unexpected native group-scale layout; refuse reinterpretation')
    shape=split_metadata_shape(n,k,group,configuration)
    if shape==tuple(scales.shape):return scales,{'padded':False,'shape':list(shape)}
    padded=torch.zeros(shape,dtype=scales.dtype,device=scales.device)
    padded[:k//group,:n].copy_(scales)
    return padded,{'padded':True,'logical_shape':list(scales.shape),'shape':list(shape),
        'padding_value':0,'scope':'metadata only; logical K/N, packed weights, launch knobs and masked A/B loads unchanged',
        'reason':'upstream INT Split-K scale loads are unmasked in K and N'}


def native_split_metadata_bounds(native,*,m,n,k,bits,group):
    # The native autotuner may select any declared launch. Include its exact
    # shape-cache override too, without changing the selector or its order.
    declared=declared_configs('GEMM_SPLITK')
    bounds={key:max(c[key] for c in declared) for key in ('BLOCK_SIZE_K','BLOCK_SIZE_N','SPLIT_K')}
    signature=str((native.utils.get_closest_m(m),n,k,group,32//bits,native.DType.FP16.value*100+bits))
    cached=native.GEMLITE_TRITON_CONFIG_CACHE.get('GEMM_SPLITK',{}).get(signature,{})
    if cached and cached.get('BLOCK_SIZE_K',group)>group:
        raise UnsupportedPreparationError('native cached INT tile spans multiple quantization groups',details={'configuration':cached,'group_size':group})
    for key in bounds:
        value=cached.get(key,bounds[key])
        if type(value) is not int or value<1:raise UnsupportedPreparationError('invalid native cached launch extent')
        bounds[key]=max(bounds[key],value)
    return bounds


class _OutputAllocator:
    def __init__(self,torch,ctx):self.torch=torch;self.ctx=ctx
    def __getattr__(self,name):return getattr(self.torch,name)
    def empty(self,shape,**kwargs):
        out=self.ctx.outputs['out']
        if tuple(shape)==tuple(out.shape) and kwargs.get('dtype',out.dtype)==out.dtype:return out
        # BF16 without native atomics may require a timed FP32 reduction/cast;
        # this is the real wrapper, never a Torch numerical replacement.
        return self.torch.empty(shape,**kwargs)
    def zeros(self,shape,**kwargs):
        value=self.empty(shape,**kwargs);value.zero_();return value


class _Launch:
    def __init__(self,kernel,ctx,parameters,explicit):
        self.original=kernel;self.ctx=ctx;self.parameters=parameters;self.explicit=explicit
        self.fn=kernel.fn if explicit else kernel # Strip only Autotuner, preserve native heuristics/JIT.
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            cache_before=len(getattr(self.original,'cache',{}))
            if self.explicit:
                p=dict(self.parameters);p['NUM_STAGES']=p['num_stages'];kwargs.update(p)
                if p.get('SPLIT_K',1)>1:args[2].zero_()
            compiled=self.fn[grid](*args,**kwargs)
            self.ctx.record_compiled_kernel('gemlite',compiled)
            best=getattr(self.original,'best_config',None)
            chosen=self.parameters if self.explicit else (best.all_kwargs() if best is not None else
                {'tile':'unknown native best_config not exposed','num_warps':getattr(compiled.metadata,'num_warps',None),
                 'num_stages':getattr(compiled.metadata,'num_stages',None)})
            self.ctx.metadata['experiment_annotations']['configuration_mapping']=ConfigurationMapping(
                self.parameters if self.explicit else {'selection':'native'},dict(chosen),
                'raw explicit JIT with native derived heuristics' if self.explicit else 'native cache/pruner/autotuner actual best_config',
                'fixed GemLite'+VERSION).to_dict()
            if not self.explicit:
                self.ctx.metadata['experiment_annotations']['native_internal_tuning']={
                    'upstream_benchmark_wall_s':getattr(self.original,'bench_time',None),
                    'declared_autotune_proposals':len(self.original.configs),
                    'cache_entries_before':cache_before,'cache_entries_after':len(self.original.cache),
                    'actual_native_benchmark_evaluations':'unknown unless upstream exports them; never counted as platform research observations'}
            return compiled
        return call


def build(ctx,*,family,mode,**params):
    native,identity=native_api();torch=ctx.torch;bits=ctx.operator.parameters['bits'];group=ctx.operator.parameters['group_size']
    runtime_config=importlib.import_module('gemlite.triton_kernels.config')
    if runtime_config.GEMLITE_FAST_GEMV_ACC:raise ValueError('fast input-dtype GEMV accumulation violates the fixed FP32 accumulation contract')
    m,n,k=(ctx.shape.get(x) for x in ('M','N','K'));start=time.perf_counter()
    dtype=native.DType.FP16 if ctx.operator.dtype=='float16' else native.DType.BF16
    layer=native.GemLiteLinearTriton(bits,group,k,n,input_dtype=dtype,output_dtype=dtype,acc_dtype=native.DType.FP32)
    layer.pack(ctx.inputs['codes'],ctx.inputs['scales'],1<<(bits-1),packing_bitwidth=32,fma_mode=True)
    actual_family=native.get_matmul_type(m,bits,False) if mode=='native' else family
    metadata_padding=None;prepared_scales=layer.scales
    if actual_family=='GEMM_SPLITK':
        if layer.W_group_mode!=3 or layer.zeros.numel()!=1:raise UnsupportedPreparationError('audited Split-K padding requires group scales and scalar zero')
        bounds=params if mode=='explicit' else native_split_metadata_bounds(native,m=m,n=n,k=k,bits=bits,group=group)
        required_shape=split_metadata_shape(n,k,group,bounds)
        if math.prod(required_shape)*layer.scales.element_size()>ctx.workspace_memory_limit_bytes:
            raise ResourceContextLimit('Split-K prepared metadata exceeds this preparation memory context')
        prepared_scales,metadata_padding=pad_split_metadata(torch,layer.scales,n=n,k=k,group=group,configuration=bounds)
        metadata_padding['bound_source']='explicit launch' if mode=='explicit' else 'complete native proposal extents plus exact shape-cache override'
    for name,value in [('weights',layer.W_q),('scales',prepared_scales),('zeros',layer.zeros)]:
        ctx.register_prepared_input('gemlite_'+name,value)
    if actual_family not in FAMILIES:raise RuntimeError('this adapter has no audited launch for native family '+actual_family)
    backend=importlib.import_module('gemlite.triton_kernels.'+FAMILIES[actual_family])
    original=native.GEMLITE_TRITON_MAPPING[actual_family].forward
    scope=dict(original.__globals__);scope['torch']=_OutputAllocator(torch,ctx)
    used=[]
    for name,value in list(scope.items()):
        if name.endswith('_INT_kernel'):
            scope[name]=_Launch(value,ctx,params,mode=='explicit');used.append(name)
    forward=FunctionType(original.__code__,scope,original.__name__,original.__defaults__,original.__closure__)
    meta=layer.get_meta_args();type_id=native.DType.FP16.value*100+bits
    tensor_args=layer.get_tensor_args();tensor_args[1]=prepared_scales;torch.cuda.synchronize()
    annotations=ctx.metadata.setdefault('experiment_annotations',{});annotations.update(
        backend='GemLite',version=VERSION,source_identity=identity,family=actual_family,
        mode=mode,hidden_autotune_allowed=mode=='native',packing_wall_ms=(time.perf_counter()-start)*1000,
        packing_identity={'logical':annotations.get('logical_input_identity'),
            'packed_shape':list(layer.W_q.shape),'packed_dtype':str(layer.W_q.dtype),'elements_per_sample':layer.elements_per_sample},
        numeric_contract='FP16/BF16 A; uniform signed int'+str(bits)+' group scales; FP32 accumulator',
        scope='native complete forward including atomic initialization and required cast/output stabilization; no routing/service')
    if metadata_padding is not None:annotations['split_metadata_padding']=metadata_padding
    def invoke():
        result=forward(ctx.inputs['a'],*tensor_args[:3],None,*meta[1:-1],bool(layer.data_contiguous),type_id,meta_scale=tensor_args[3])
        if result.data_ptr()!=ctx.outputs['out'].data_ptr():ctx.outputs['out'].copy_(result)
    return ExecutionPlan((ExecutionStep('native_'+actual_family,invoke),))


def register_assets(catalog):
    source_files=sources.declaration_sources(__file__,suite_inputs.__file__,gemlite_compat.__file__)
    operator=suite_inputs.operator_definition()
    catalog.register(operator)
    catalog.register(KernelDefinition('production/gemlite_default','1.0.0',operator.ref,partial(build,family=None,mode='native'),
        SpaceDefinition({},name='native GemLite family/cache/autotune expert'),runtime='cuda',supports=_support,
        source_files=source_files,execution_identity=partial(environment.installed_identity,'gemlite','gemlite',native_default=True),
        metadata={'pin':sources.pin('gemlite'),'cost':'native internal tuning separate from research observations'}))
    for family in FAMILIES:
        configs=declared_configs(family);keys=configs[0].keys()
        domains={key:tuple(dict.fromkeys(c[key] for c in configs)) for key in keys}
        # The native expert's default stage2 is omitted from its "max" tuner
        # proposal list, but is an actual valid implementation parameter.
        domains['num_stages']=tuple(sorted(set(domains['num_stages'])|{2}))
        space=SpaceDefinition({k:ParameterSpec(v) for k,v in domains.items()},constraints=(partial(hard_config,family=family),),
            name='GemLite '+family+' explicit declared launch knobs')
        catalog.register(KernelDefinition('production/gemlite_'+family.lower(),'1.0.0',operator.ref,
            partial(build,family=family,mode='explicit'),space,runtime='cuda',supports=_support,source_files=source_files,
            execution_identity=partial(environment.installed_identity,'gemlite','gemlite'),
            metadata={'pin':sources.pin('gemlite'),'native_default_pruner':'not applied to explicit path; hard semantic constraints only',
                'native_family':family,'source_declared_candidates':len(configs),'effective_parameters':list(keys)}))

