"""Original vLLM0.30 Triton routed-expert computation, not a grouped-GEMM toy."""
from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache,partial
import importlib,importlib.metadata,importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import threading

from ..sdk import OperatorDefinition,KernelDefinition,ParameterSpec,SpaceDefinition,ShapeSpec,ExecutionPlan,ExecutionStep
from ..torch_context import strict_fp32_reference
from . import native_sources as sources,environment
from .inputs import _counter

VERSION='0.30.0';_NATIVE_LOCK=threading.RLock()
KNOBS={'BLOCK_SIZE_M':(16,32,64,128,256),'BLOCK_SIZE_N':(32,64,128,256),
    'BLOCK_SIZE_K':(32,64,128),'GROUP_SIZE_M':(1,4,8,16,32),
    'num_warps':(4,8),'num_stages':(2,3,4,5)}


def native_rule(m,e,n,k,topk,*,table=None,override=None,batch_invariant=False,rocm=False):
    ns={'envs':SimpleNamespace(VLLM_BATCH_INVARIANT=batch_invariant),
        'current_platform':SimpleNamespace(is_rocm=lambda:rocm),'next_power_of_2':lambda v:1<<(v-1).bit_length(),
        'get_config':lambda:override,'get_moe_configs':lambda *a:table}
    sources.host_functions('vllm','vllm/model_executor/layers/fused_moe/fused_moe.py',
        {'get_default_config','try_get_optimal_moe_config'},ns,
        remove_imports=('vllm.model_executor.layers.fused_moe',))
    return ns['try_get_optimal_moe_config']((e,2*n,k),(e,k,n),topk,None,m,None)


def read_native_table(path):
    raw=json.loads(Path(path).read_text());result={}
    for batch,cfg in raw.items():
        m=int(batch)
        if m<1 or str(m)!=batch or not isinstance(cfg,dict):raise ValueError('native MoE table requires positive integer M keys')
        if set(cfg)-set(KNOBS)-{'SPLIT_K'}:raise ValueError('unknown native MoE configuration field')
        if cfg.get('SPLIT_K',1)!=1:raise ValueError('this unquantized fused-MoE path fixes SPLIT_K=1')
        result[m]=dict(cfg)
    if not result:raise ValueError('empty native MoE table')
    return result


def write_native_table(path,table):
    # Native nearest-M tie-breaking follows dict insertion order. Sorting this
    # payload would silently change the original production selector.
    data={str(k):dict(v) for k,v in table.items()}
    Path(path).write_text(json.dumps(data,indent=2)+'\n')
    return read_native_table(path)


def logical_arrays(shape,parameters,seed):
    import numpy as np
    m,h,i,e,t=(shape.get(k) for k in ('M','H','I','E','TOP_K'))
    def values(dims,salt):
        a=((_counter(int(np.prod(dims)),seed^salt)%17).astype(np.int16)-8).reshape(dims).astype(np.float32)/8
        # Keep variance meaningful after two expert projections. The previous
        # /64 for every tensor produced near-zero outputs where absolute
        # tolerance could accept an all-zero implementation on tiny cases.
        if len(dims)==3:a*=2/(dims[-1]**.5)
        return a
    ids=(np.arange(m)[:,None]+np.arange(t)[None,:])%e
    if parameters['route_distribution']=='skewed':ids[np.arange(m)%5!=0]=np.arange(t)[None,:]
    weights=(1+(_counter(m*t,seed^25)&7)).reshape(m,t).astype(np.float32);weights/=weights.sum(1,keepdims=True)
    return {'a':values((m,h),13),'w1':values((e,2*i,h),17),'w2':values((e,h,i),19),
        'topk_ids':ids.astype(np.int32),'topk_weights':weights}


def generate(ctx,seed):
    torch=ctx.torch;m,h=(ctx.shape.get(k) for k in ('M','H'));device=f'cuda:{ctx.device}';dtype=getattr(torch,ctx.operator.dtype)
    arrays=logical_arrays(ctx.shape,ctx.operator.parameters,seed)
    ctx.allocate_output('out',(m,h))
    return {key:torch.from_numpy(value).to(device,dtype=torch.int32 if key=='topk_ids' else torch.float32 if key=='topk_weights' else dtype)
        for key,value in arrays.items()}


def reference(ctx):
    torch=ctx.torch;a=ctx.inputs['a'];w1=ctx.inputs['w1'];w2=ctx.inputs['w2'];ids=ctx.inputs['topk_ids'];rw=ctx.inputs['topk_weights'];dtype=a.dtype;i=ctx.shape.get('I')
    with strict_fp32_reference(torch):
        out=torch.zeros_like(a,dtype=torch.float32)
        for expert in range(ctx.shape.get('E')):
            for slot in range(ctx.shape.get('TOP_K')):
                rows=torch.where(ids[:,slot]==expert)[0]
                if not rows.numel():continue
                first=(a[rows].float()@w1[expert].float().T).to(dtype)
                gate=first[:,:i].float();up=first[:,i:].float()
                act=(torch.nn.functional.silu(gate)*up).to(dtype)
                second=act.float()@w2[expert].float().T
                out[rows]+=(second*rw[rows,slot,None]).to(dtype).float()
        return {'out':out.to(dtype)}


def validate(ctx):
    s=ctx.shape
    if s.get('TOP_K')>s.get('E'):return 'Top-K must not exceed expert count; no duplicate expert per token'
    if ctx.operator.parameters['route_distribution'] not in ('balanced','skewed'):return 'supported controlled route distributions are balanced/skewed'
    if s.get('H')%32 or s.get('I')%32:return 'initial native cohort requires hidden/intermediate dimensions divisible by32'
    if s.get('M')>4096 or s.get('E')>64:return 'bounded acceptance cohort, not full-service capacity claim'
    return None


def support(h,o,s):
    try:
        if importlib.metadata.version('vllm')!=VERSION:return 'vLLM0.30 private ABI-compatible environment required; no global/shared upgrade'
    except importlib.metadata.PackageNotFoundError:return 'optional vLLM0.30 production dependency is unavailable'
    return None


@lru_cache(maxsize=1)
def native_api():
    spec=importlib.util.find_spec('vllm');root=Path(spec.origin).parent.parent
    identity=sources.native_component(root,'vllm',[
        'vllm/model_executor/layers/fused_moe/fused_moe.py','vllm/model_executor/layers/fused_moe/config.py',
        'vllm/model_executor/layers/fused_moe/activation.py','vllm/model_executor/layers/fused_moe/moe_align_block_size.py'])
    package=importlib.import_module('vllm.model_executor.layers.fused_moe')
    module=importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
    return package,module,identity


class _KernelTrace:
    def __init__(self,original,ctx):self.original=original;self.ctx=ctx;self.index=0
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            compiled=self.original[grid](*args,**kwargs);slot=self.index;self.index+=1
            self.ctx.record_compiled_kernel('vllm_moe_gemm'+str(slot),compiled)
            self.ctx.metadata['experiment_annotations'].setdefault('actual_launch_parameters',{})[str(slot)]={
                k:v for k,v in kwargs.items() if k in (*KNOBS,'SPLIT_K','top_k','MUL_ROUTED_WEIGHT')}
            return compiled
        return launch


def build(ctx,*,mode,**configuration):
    package,module,identity=native_api();torch=ctx.torch;trace=_KernelTrace(module.fused_moe_kernel,ctx)
    kwargs=dict(configuration);kwargs['SPLIT_K']=1
    inputs=ctx.inputs
    ctx.metadata.setdefault('experiment_annotations',{}).update(
        backend='vLLM',version=VERSION,source_identity=identity,scope='two expert GEMMs + native SiLU/multiply + config-dependent token alignment/reorder + weighted reduction + output copy; routing network excluded',
        requested_configuration=configuration if mode=='explicit' else {'selection':'native_table_or_default'},
        alignment_policy='recomputed by original fused_experts_impl on EVERY invocation with the actual BLOCK_SIZE_M',
        stage_parameter_contract='the upstream single shared config is used by both expert GEMMs; no independent-stage Cartesian extension')
    def invoke():
        with _NATIVE_LOCK:
            old_config=package.get_config();old_kernel=module.fused_moe_kernel
            try:
                with package.override_config(kwargs if mode=='explicit' else old_config):
                    selected=module.try_get_optimal_moe_config(tuple(inputs['w1'].shape),tuple(inputs['w2'].shape),ctx.shape.get('TOP_K'),None,ctx.shape.get('M'),None)
                    ctx.metadata['experiment_annotations']['native_selected_configuration']=dict(selected)
                    module.fused_moe_kernel=trace;trace.index=0
                    result=module.fused_experts_impl(inputs['a'],inputs['w1'],inputs['w2'],inputs['topk_weights'],inputs['topk_ids'],
                        activation='silu',apply_router_weight_on_input=False)
                    if trace.index!=2:raise RuntimeError('native execution did not expose the two audited Triton GEMM launches; backend/path is unsupported here')
                    ctx.outputs['out'].copy_(result)
            finally:
                module.fused_moe_kernel=old_kernel
                # Upstream override_config lacks finally around yield; restore
                # our own worker's previous override even after compilation errors.
                package._config=old_config
    return ExecutionPlan((ExecutionStep('native_routed_expert_full_then_output_copy',invoke),))


def register_assets(catalog):
    src=(__file__,sources.__file__,str(sources.ROOT/'manifest.json'))
    op=OperatorDefinition('production/routed_moe_silu','1.0.0',('M','H','I','E','TOP_K'),('float16','bfloat16'),generate,reference,
        tolerances={'float16':(.01,.01),'bfloat16':(.02,.02)},parameters={'route_distribution':'balanced'},validate=validate,
        validation_shapes=(ShapeSpec({'M':17,'H':128,'I':256,'E':4,'TOP_K':2}),),source_files=src,
        metadata={'semantics':'pre-routed Top-K expert linear(gate/up), SiLU*up, expert down projection, routed weighted sum; no router network'})
    catalog.register(op)
    for mode in ('native','explicit'):
        space=SpaceDefinition({k:ParameterSpec(v) for k,v in KNOBS.items()},name='native shared two-stage MoE launch config') if mode=='explicit' else SpaceDefinition({},name='original vLLM native config table and fallback')
        catalog.register(KernelDefinition('production/vllm_moe_'+mode,'1.0.0',op.ref,partial(build,mode=mode),space,
            runtime='cuda',supports=support,source_files=src,execution_identity=partial(environment.installed_identity,'vllm','vllm',native_default=mode=='native'),
            metadata={'pin':sources.pin('vllm'),'native_path':'fused_experts_impl + original override_config and try_get_optimal_moe_config',
            'scope':'full routed expert computation with shared two-stage config, not single Grouped GEMM','fallback':'none; missing/incompatible native package is explicit'}))
