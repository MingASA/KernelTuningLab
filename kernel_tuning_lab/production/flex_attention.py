"""Pinned native FlexAttention inference and scoped fixed-version launch hooks."""
from __future__ import annotations
import ast
from contextlib import contextmanager,nullcontext
from functools import partial
import hashlib
import importlib,importlib.metadata
import math
import re
import threading
from pathlib import Path

from ..sdk import OperatorDefinition,KernelDefinition,ParameterSpec,SpaceDefinition,ShapeSpec,ExecutionPlan,ExecutionStep
from ..torch_context import strict_fp32_reference
from . import native_sources as sources,environment

KNOBS={'BLOCK_M':(32,64,128),'BLOCK_N':(32,64,128),'num_warps':(4,8),'num_stages':(1,2,3)}
SOURCE_PROFILES={'2.14.0':'pytorch','2.13.0':'pytorch213','2.6.0':'pytorch26'}
_LEGACY_CONFIG_LOCK=threading.RLock()


def kernel_options(shape,mask,*,backend,parameters,source_version='2.14.0'):
    source_version=source_version.split('+')[0]
    if source_version not in SOURCE_PROFILES:raise ValueError('unsupported fixed FlexAttention source profile')
    if backend not in ('AUTO','TRITON','TRITON_DECODE','FLASH'):raise ValueError('unknown native FlexAttention backend')
    if backend!='TRITON' and parameters:raise ValueError('Triton forward tile knobs are not applied to another native backend')
    if mask not in ('dense','causal'):raise ValueError('controlled forward mask must be dense/causal')
    if mask=='causal' and shape.get('Q_LEN')!=shape.get('KV_LEN'):raise ValueError('prefill causal contract requires equal Q/KV lengths; no implicit decode position offset')
    if set(parameters)-set(KNOBS):raise ValueError('semantic promises and numerical options are not tunable launch parameters')
    options={'PRESCALE_QK':False,'ROWS_GUARANTEED_SAFE':True,'BLOCKS_ARE_CONTIGUOUS':False}
    if source_version=='2.6.0':
        if backend not in ('AUTO','TRITON'):
            raise ValueError('fixed Torch2.6 compatibility supports native AUTO or forced Triton forward only; FLASH/TRITON_DECODE selection is unavailable')
        if parameters:
            if set(parameters)!=set(KNOBS):raise ValueError('Torch2.6 explicit launch hook requires all four scheduling parameters')
            if parameters['num_stages']==2:
                raise ValueError('fixed Torch2.6 forward source explicitly skips num_stages=2 (upstream issue129625); no silent stage rewrite')
        # 2.6 passes launch arguments separately from **kernel_options. The
        # native kwargs would conflict, so only its real tile keys go here.
        if backend=='TRITON':options['FORCE_USE_FLEX_ATTENTION']=True
        options.update({k:v for k,v in parameters.items() if k in ('BLOCK_M','BLOCK_N')})
    else:options.update(BACKEND=backend,**parameters)
    return options


def resolve_launch_domains(context):
    version=context.environment.get('torch')
    if version is None:
        try:version=importlib.metadata.version('torch')
        except importlib.metadata.PackageNotFoundError:return {}
    if version.split('+')[0]=='2.6.0':return {'num_stages':(1,3)}
    return {}


@contextmanager
def legacy_fwd_configuration(parameters=None,*,module=None):
    """One process-local host selection hook; original CUDA/Triton bodies stay
    untouched. AUTO shares the lock but never replaces its native selector.
    """
    with _LEGACY_CONFIG_LOCK:
        if not parameters:
            yield
            return
        if module is None:
            # Initialize native lowering first, avoiding its standalone-module
            # circular import. This runs only inside an admitted CUDA build.
            importlib.import_module('torch._inductor.lowering')
            module=importlib.import_module('torch._inductor.kernel.flex_attention')
        original=module._get_default_config_fwd
        requested=tuple(parameters[k] for k in ('BLOCK_M','BLOCK_N','num_warps','num_stages'))
        def selected(query):return requested
        module._get_default_config_fwd=selected
        try:yield
        finally:module._get_default_config_fwd=original


def generated_configuration(codes):
    """Read actual Inductor-emitted template metadata/constants; no inference
    from requested parameters. Missing evidence remains unknown and rejects
    the explicit path rather than claiming a knob was honored.
    """
    results=[]
    for code in codes:
        # Generated Triton source is usually embedded in triple-quoted Python
        # strings. Parse both the wrapper and every literal source string.
        queue=[code];seen=set()
        while queue:
            text=queue.pop()
            if text in seen:continue
            seen.add(text)
            try:tree=ast.parse(text)
            except SyntaxError:continue
            values={}
            for node in ast.walk(tree):
                if isinstance(node,ast.Constant) and isinstance(node.value,str) and 'def ' in node.value:queue.append(node.value)
                if isinstance(node,(ast.Assign,ast.AnnAssign)):
                    targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target,ast.Name) and target.id in ('BLOCK_M','BLOCK_N'):
                            try:values[target.id]=ast.literal_eval(node.value)
                            except (ValueError,TypeError):pass
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='template':
                    for kw in node.keywords:
                        if kw.arg in ('num_warps','num_stages'):
                            try:values[kw.arg]=ast.literal_eval(kw.value)
                            except (ValueError,TypeError):pass
            if set(KNOBS)<=set(values):results.append({k:values[k] for k in KNOBS})
    unique={tuple(sorted(r.items())):r for r in results}
    return list(unique.values())


def generate(ctx,seed):
    torch=ctx.torch;s=ctx.shape;g=torch.Generator(device='cpu');g.manual_seed(seed)
    b,hq,hkv,d,ql,kl=(s.get(k) for k in ('B','H_Q','H_KV','D','Q_LEN','KV_LEN'));dtype=getattr(torch,ctx.operator.dtype);device=f'cuda:{ctx.device}'
    ctx.allocate_output('out',(b,hq,ql,d))
    return {'q':(torch.randn((b,hq,ql,d),generator=g)*.125).to(device,dtype=dtype),
        'k':(torch.randn((b,hkv,kl,d),generator=g)*.125).to(device,dtype=dtype),
        'v':(torch.randn((b,hkv,kl,d),generator=g)*.125).to(device,dtype=dtype)}


def reference(ctx):
    torch=ctx.torch;q,k,v=(ctx.inputs[x].float() for x in ('q','k','v'));ratio=q.shape[1]//k.shape[1]
    k=k.repeat_interleave(ratio,dim=1);v=v.repeat_interleave(ratio,dim=1)
    with strict_fp32_reference(torch):
        scores=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])
        if ctx.operator.parameters['mask']=='causal':
            mask=torch.arange(k.shape[-2],device=k.device)[None,:]<=torch.arange(q.shape[-2],device=q.device)[:,None]
            scores=scores.masked_fill(~mask,float('-inf'))
        return {'out':(scores.softmax(dim=-1)@v).to(ctx.inputs['q'].dtype)}


def validate(ctx):
    s=ctx.shape
    if s.get('H_Q')%s.get('H_KV'):return 'GQA requires H_Q divisible by H_KV'
    if s.get('D') not in (32,64,128):return 'initial controlled native forward head dimensions32/64/128'
    try:kernel_options(s,ctx.operator.parameters['mask'],backend='AUTO',parameters={})
    except ValueError as e:return str(e)
    if s.get('Q_LEN')*s.get('KV_LEN')*s.get('B')*s.get('H_Q')>16_777_216:return 'bounded independent dense reference limit'
    return None


def support(h,o,s,*,backend=None):
    try:
        version=importlib.metadata.version('torch').split('+')[0]
        if version not in SOURCE_PROFILES:return 'fixed PyTorch2.14 or source-verified2.13/2.6 compatibility environment required; no global/shared upgrade'
        if version=='2.6.0' and backend not in (None,'AUTO','TRITON'):
            return 'Torch2.6 profile cannot select FLASH/TRITON_DECODE; newer source profiles keep their native backend interfaces'
    except importlib.metadata.PackageNotFoundError:return 'optional Torch2.14 dependency unavailable'
    return None


def build(ctx,*,backend,**parameters):
    torch=ctx.torch;native=importlib.import_module('torch.nn.attention.flex_attention')
    root=Path(torch.__file__).parent.parent
    version=torch.__version__.split('+')[0]
    profile=SOURCE_PROFILES[version]
    identity=sources.native_component(root,profile,[f for f in sources.manifest()[profile]['files'] if f.startswith('torch/')])
    legacy=version=='2.6.0'
    options=kernel_options(ctx.shape,ctx.operator.parameters['mask'],backend=backend,parameters=parameters,source_version=version)
    block_mask=None
    if ctx.operator.parameters['mask']=='causal':
        def causal(b,h,q,k):return q>=k
        block_mask=native.create_block_mask(causal,B=ctx.shape.get('B'),H=ctx.shape.get('H_Q'),
            Q_LEN=ctx.shape.get('Q_LEN'),KV_LEN=ctx.shape.get('KV_LEN'),device=f'cuda:{ctx.device}')
        for name in ('kv_num_blocks','kv_indices','full_kv_num_blocks','full_kv_indices','q_num_blocks','q_indices'):
            value=getattr(block_mask,name,None)
            if isinstance(value,torch.Tensor):ctx.register_prepared_input('flex_mask_'+name,value)
    def forward(q,k,v):
        if legacy and parameters:
            # These genuine contract checks also put both launch scalars into
            # Dynamo's closure guards. Otherwise a cached graph with identical
            # tile keys could bypass the host hook for another warp/stage pair.
            if parameters['num_warps'] not in (4,8) or parameters['num_stages'] not in (1,3):
                raise ValueError('invalid fixed Torch2.6 launch contract')
        return native.flex_attention(q,k,v,block_mask=block_mask,enable_gqa=ctx.shape.get('H_Q')!=ctx.shape.get('H_KV'),kernel_options=options)
    compile_options={'triton.cudagraphs':False}
    if parameters:compile_options.update(max_autotune=False,max_autotune_gemm=False)
    if version=='2.6.0' and parameters:
        # Launch tuple lives in the scoped legacy host selector, outside FX
        # kwargs. Do not let an old same-tile graph hide a changed warp/stage.
        compile_options.update(fx_graph_cache=False,fx_graph_remote_cache=False)
    compiled=torch.compile(forward,fullgraph=True,dynamic=False,options=compile_options)
    from torch._inductor.utils import run_and_get_code
    with legacy_fwd_configuration(parameters or None) if legacy else nullcontext():
        with torch.inference_mode():_,codes=run_and_get_code(compiled,ctx.inputs['q'],ctx.inputs['k'],ctx.inputs['v'])
    actual=generated_configuration(codes)
    if parameters and actual!=[parameters]:
        raise RuntimeError('cannot confirm one actual Flex Triton configuration equals request; generated metadata='+str(actual))
    ctx.metadata.setdefault('experiment_annotations',{}).update(backend='PyTorch FlexAttention',version=torch.__version__,source_identity=identity,
        requested_configuration=dict(parameters),actual_generated_configurations=actual,
        generated_source_sha256=[hashlib.sha256(c.encode()).hexdigest() for c in codes],
        backend_request=backend,actual_backend='Triton template observed' if actual else 'unknown native AUTO/FLASH path; not inferred from request',
        compatibility_profile='Torch2.6 native forward host configuration hook; no numeric/template rewrite' if legacy else 'native public kernel_options',
        launch_parameter_path='locked temporary _get_default_config_fwd tuple; generated configuration independently checked' if legacy and parameters else 'native default selector' if legacy else 'native kernel_options',
        fixed_semantic_options={k:v for k,v in options.items() if k not in KNOBS},
        row_safety_evidence='dense nonempty KV or equal-length causal prefill has at least key0 for each valid row',
        scope='native compiled forward + output copy; mask construction/compile outside timed call',
        hidden_max_autotune=False if parameters else 'native default setting preserved',
        compile_options=dict(compile_options))
    def invoke():
        with torch.inference_mode():ctx.outputs['out'].copy_(compiled(ctx.inputs['q'],ctx.inputs['k'],ctx.inputs['v']))
    return ExecutionPlan((ExecutionStep('native_flex_forward_then_output_copy',invoke),))


def register_assets(catalog):
    src=sources.declaration_sources(__file__)
    op=OperatorDefinition('production/attention_prefill','1.0.0',('B','H_Q','H_KV','D','Q_LEN','KV_LEN'),('float16','bfloat16'),
        generate,reference,parameters={'mask':'causal'},tolerances={'float16':(.005,.005),'bfloat16':(.02,.02)},validate=validate,
        validation_shapes=(ShapeSpec({'B':1,'H_Q':4,'H_KV':2,'D':64,'Q_LEN':129,'KV_LEN':129}),),source_files=src,
        metadata={'semantics':'forward inference scaled dot-product attention; dense/equal-length causal; GQA; no backward or paged cache'})
    catalog.register(op)
    for backend in ('AUTO','TRITON','TRITON_DECODE','FLASH'):
        explicit=backend=='TRITON'
        space=SpaceDefinition({k:ParameterSpec(v) for k,v in KNOBS.items()} if explicit else {},
            resolver=resolve_launch_domains if explicit else None,name='native Flex '+backend)
        catalog.register(KernelDefinition('production/flex_'+backend.lower(),'1.0.0',op.ref,partial(build,backend=backend),space,
            runtime='cuda',supports=partial(support,backend=backend),source_files=src,execution_identity=partial(environment.installed_identity,'pytorch','torch'),
            metadata={'pin':sources.pin('pytorch'),'compatible_source_pin':sources.pin('pytorch213'),
            'legacy_compatibility_source_pin':sources.pin('pytorch26'),
            'legacy_launch_hook':'Torch2.6 only: locked temporary host default-config tuple, never original numeric body; AUTO untouched; stage2 source workaround preserved',
            'native_backend':backend,'explicit_forward_launch':explicit,'unsupported_option_policy':'reject, never treat passed kwargs as proof'}))
