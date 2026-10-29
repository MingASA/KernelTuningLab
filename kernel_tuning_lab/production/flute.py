"""Real FLUTE0.4.2 W3 qgemm, structured templates and packing dependencies."""
from __future__ import annotations
import ast
from functools import lru_cache,partial
import hashlib
import importlib,importlib.metadata,importlib.util
import json
import math
import os
from pathlib import Path
import time

from ..sdk import KernelDefinition,ParameterSpec,SpaceDefinition,ExecutionPlan,ExecutionStep
from . import native_sources as sources,suite_inputs
from .contracts import NativeCandidate,preparation_identity

VERSION='0.4.2'


@lru_cache(maxsize=1)
def template_configs():
    """Execute only the original codegen enumeration, before its file writes."""
    path=sources.source('flute','flute/codegen_utils.py');tree=ast.parse(path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='codegen_raw')
    body=[]
    for n in function.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='names' for t in n.targets):break
        body.append(n)
    fn=ast.FunctionDef(name='_enumerate',args=function.args,body=[*body,ast.Return(ast.Name(id='configs',ctx=ast.Load()))],decorator_list=[])
    ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),ns)
    return ns['_enumerate']()


def candidates(bits=3,compiled_ids=None):
    compiled=None if compiled_ids is None else set(compiled_ids)
    return tuple(NativeCandidate(str(tid),dict(c),'FLUTE StreamK',packing_key='flute-int'+str(bits)+'-tileP'+str(c['TileP']),
        preparation={'tile_P':c['TileP'],'native_template_id':tid,'bits':bits},
        compiled=None if compiled is None else tid in compiled)
        for (b,tid),c in template_configs().items() if b==bits)


@lru_cache(maxsize=1)
def host_rules():
    ns={'math':math,'TEMPLATE_CONFIGS':template_configs()}
    sources.host_functions('flute','flute/utils.py',{'get_template_config','get_template_ids','is_template_supported'},ns)
    return ns


def template_id(parameters,bits=3):
    match=[int(c.native_id) for c in candidates(bits) if dict(c.parameters)==dict(parameters)]
    if len(match)!=1:raise ValueError('structural parameters do not name one original FLUTE template')
    return match[0]


def hard_support(config,ctx):
    n=ctx.shape.get('N');k=ctx.shape.get('K');m=ctx.shape.get('M');bits=ctx.operator.parameters['bits']
    if bits!=3:return 'this production FLUTE cohort is explicitly uniform signed int3'
    if n%16 or (n//16)%config.get('TileP'):return 'original int3 packing requires N/16 divisible by template TileP'
    tid=template_id(config.params,bits)
    if ctx.hardware.sm_count and not host_rules()['is_template_supported'](m,n,k,bits,tid,ctx.hardware.sm_count):
        return 'original FLUTE0.4.2 is_template_supported predicate rejects this input/template/SM count'
    return None


def _support(h,o,s):
    if o.parameters.get('bits')!=3:return 'W3 FLUTE uses bits=3 task semantics; W2/W4 are distinct supplementary cohorts'
    try:
        version=importlib.metadata.version('flute-kernel').split('+')[0]
        if version!=VERSION:return 'FLUTE0.4.2 private build required'
    except importlib.metadata.PackageNotFoundError:return 'optional FLUTE0.4.2 CUDA extension has not been built/installed'
    return None


@lru_cache(maxsize=1)
def native_api():
    spec=importlib.util.find_spec('flute')
    if spec is None:raise RuntimeError('optional FLUTE0.4.2 package is unavailable')
    root=Path(spec.origin).parent.parent
    identity=sources.native_component(root,'flute',['flute/utils.py','flute/tune.py','flute/ops.py','flute/codegen_utils.py'])
    module=importlib.import_module('flute');utils=importlib.import_module('flute.utils');tune=importlib.import_module('flute.tune')
    ext=importlib.import_module('flute._C');binary=Path(ext.__file__)
    identity['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    build_path=os.environ.get('KT_FLUTE_BUILD_MANIFEST');build=None
    if build_path:
        build=json.loads(Path(build_path).read_text())
        if build['binary_sha256']!=identity['binary_sha256']:raise ValueError('FLUTE compiled-set manifest does not match the loaded binary')
        identity['compiled_templates']=build['compiled_templates'];identity['compiled_architectures']=build['compiled_architectures']
    else:identity['compiled_templates']=None
    return module,utils,tune,identity


def workspace_bytes(sm_count):return sm_count*4*256*(4*64*8)+4*(sm_count*4)


def build(ctx,*,mode,**parameters):
    module,utils,tune,identity=native_api();torch=ctx.torch;bits=ctx.operator.parameters['bits'];group=ctx.operator.parameters['group_size']
    m,n,k=(ctx.shape.get(x) for x in ('M','N','K'));sms=ctx.hardware.sm_count;start=time.perf_counter()
    if not sms:raise ValueError('actual GPU SM count must be probed before FLUTE execution')
    if mode=='native':
        # Real upstream selector/tuner, not a copied nearest-config heuristic.
        before=len(tune._TEMPLATES)
        tid=tune._tune(M=m,N=n,K=k,num_bits=bits,group_size=group,num_sms=sms,
            dtype=getattr(torch,ctx.operator.dtype),device=ctx.inputs['a'].device,num_seeds=3,legacy=False)
        selection={'native_template_cache_hit':len(tune._TEMPLATES)==before,'native_internal_tuning_wall_ms':(time.perf_counter()-start)*1000}
    else:tid=template_id(parameters,bits);selection={'native_internal_tuning_wall_ms':0.,'hidden_tuning_allowed':False}
    candidate=next(c for c in candidates(bits) if int(c.native_id)==tid)
    if identity.get('compiled_templates') is not None and tid not in identity['compiled_templates'].get(str(bits),[]):
        raise RuntimeError('template is source-expressible but not in this fixed compiled subset')
    if not utils.is_template_supported(m,n,k,bits,tid,sms):raise ValueError('actual upstream template support check failed')
    codes=ctx.inputs['codes'].T.contiguous()
    pack_started=time.perf_counter();packed=utils.pack(codes,num_bits=bits,template_ids=[tid],num_sms=sms)
    scales=ctx.inputs['scales'].contiguous();qmap=torch.arange(1<<bits,dtype=getattr(torch,ctx.operator.dtype),device=ctx.inputs['a'].device)-(1<<(bits-1))
    qmap2=utils.make_qmap2_from_qmap(qmap)
    for name,value in [('packed',packed),('qmap',qmap),('qmap2',qmap2),('scales',scales)]:ctx.register_prepared_input('flute_'+name,value)
    scratch=ctx.allocate_workspace('flute_streamk',(workspace_bytes(sms),),'uint8',fill=0)
    torch.cuda.synchronize();annotations=ctx.metadata.setdefault('experiment_annotations',{})
    annotations.update(backend='FLUTE',version=VERSION,source_identity=identity,mode=mode,
        requested_configuration=dict(parameters) if mode=='explicit' else {'selection':'native'},
        effective_configuration=dict(candidate.parameters),actual_template_id=tid,
        packing_key=preparation_identity('FLUTE0.4.2',annotations['logical_input_identity'],candidate,{'bits':bits,'tile_P':candidate.parameters['TileP']}),
        packing_wall_ms=(time.perf_counter()-pack_started)*1000,packing_reuse_scope='only equal logical input and equal TileP; never arbitrary templates',
        accumulation='original kMixed StreamK, independent FP32 reference; non-deterministic reduction order possible',
        selection=selection,scope='native qgemm including required StreamK workspace/reduction and output copy; packing and initial workspace allocation outside timing')
    def invoke():
        result=module.qgemm(ctx.inputs['a'],packed,scales,qmap,qmap2,scratch,bits,group,tid,sms)
        ctx.outputs['out'].copy_(result)
    return ExecutionPlan((ExecutionStep('native_flute_qgemm_then_output_copy',invoke),))


def register_assets(catalog):
    op=suite_inputs.operator_definition();catalog.register(op)
    src=(__file__,suite_inputs.__file__,sources.__file__,str(sources.ROOT/'manifest.json'))
    rows=tuple(dict(c.parameters) for c in candidates(3));domains={k:tuple(dict.fromkeys(c[k] for c in rows)) for k in rows[0]}
    space=SpaceDefinition({k:ParameterSpec(v,kind='categorical' if isinstance(v[0],str) else 'integer') for k,v in domains.items()},
        constraints=(hard_support,),candidates=rows,name='FLUTE actual structured int3 template set')
    metadata={'pin':sources.pin('flute'),'source_template_count':len(rows),'compiled_template_set':'unknown until validated build manifest or actual execution',
        'packing_dependency':'TileP + bits + logical weights','ID_semantics':'native template_id resolved from structural parameters, never a continuous coordinate'}
    catalog.register(KernelDefinition('production/flute_w3_explicit','1.0.0',op.ref,partial(build,mode='explicit'),space,
        runtime='cuda',supports=_support,source_files=src,metadata=metadata))
    catalog.register(KernelDefinition('production/flute_w3_default','1.0.0',op.ref,partial(build,mode='native'),
        SpaceDefinition({},name='native FLUTE template tuning/cache'),runtime='cuda',supports=_support,source_files=src,
        metadata={**metadata,'cost':'original upstream internal tuning is paid separately, never hidden inside platform search observations'}))
