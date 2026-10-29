"""Pinned SGLang real paged KV decode + original GPU split scheduler."""
from __future__ import annotations
from functools import partial
import importlib.metadata
import logging
import math
from types import SimpleNamespace
from typing import NamedTuple,Optional,Tuple

from ..sdk import OperatorDefinition,KernelDefinition,ParameterSpec,SpaceDefinition,ShapeSpec,ExecutionPlan,ExecutionStep
from ..torch_context import strict_fp32_reference
from . import native_sources as sources,environment
from .inputs import _counter,counter_scalar

KNOBS={'BLOCK_N':(32,64,128),'num_warps':(2,4,8),'num_stages':(1,2,3),'MAX_KV_SPLITS':(1,2,4,8,16)}


def paged_layout(batch,max_seq,page_size,pattern,seed):
    if pattern not in ('equal','ragged'):raise ValueError('length pattern must be equal/ragged')
    lengths=[max_seq]*batch
    if pattern=='ragged':
        boundaries=[1,max(1,page_size-1),min(max_seq,page_size+1),max_seq]
        lengths=[min(max_seq,boundaries[i%4]) for i in range(batch)]
    pages_per_sequence=(max_seq+page_size-1)//page_size;pages=batch*pages_per_sequence
    physical=sorted(range(pages),key=lambda i:(counter_scalar(i,seed^0x4251),i))
    indices=[];indptr=[0]
    for b,length in enumerate(lengths):
        indices.extend(int(physical[b*pages_per_sequence+t//page_size])*page_size+t%page_size for t in range(length))
        indptr.append(len(indices))
    return {'seq_lens':lengths,'kv_indptr':indptr,'kv_indices':indices,'physical_slots':pages*page_size,
        'layout':'actual shuffled physical pages, flattened native slot buffer; per-sequence indptr/indices'}


def generate(ctx,seed):
    import numpy as np
    torch=ctx.torch;s=ctx.shape;b,hq,hkv,d,maxseq,page=(s.get(k) for k in ('B','H_Q','H_KV','D','MAX_SEQ','PAGE_SIZE'))
    layout=paged_layout(b,maxseq,page,ctx.operator.parameters['length_pattern'],seed);slots=layout['physical_slots'];dtype=getattr(torch,ctx.operator.dtype);device=f'cuda:{ctx.device}'
    def values(shape,salt):
        a=((_counter(int(np.prod(shape)),seed^salt)%33).astype(np.int16)-16).reshape(shape).astype(np.float32)/16
        return torch.from_numpy(a).to(device,dtype=dtype)
    ctx.allocate_output('out',(b,hq,d));ctx.metadata['experiment_annotations']={'paged_layout':layout['layout'],'lengths':layout['seq_lens'],
        'KV_contract':'FP16/BF16 flattened[max_slots,H_KV,D], real physical page indices; scales1/logit_cap0; no paged-to-contiguous replacement'}
    return {'q':values((b,hq,d),19),'k':values((slots,hkv,d),23),'v':values((slots,hkv,d),29),
        'kv_indptr':torch.tensor(layout['kv_indptr'],dtype=torch.int32,device=device),
        'kv_indices':torch.tensor(layout['kv_indices'],dtype=torch.int64,device=device),
        'seq_lens':torch.tensor(layout['seq_lens'],dtype=torch.int32,device=device)}


def reference(ctx):
    torch=ctx.torch;inputs=ctx.inputs;q=inputs['q'];ratio=ctx.shape.get('H_Q')//ctx.shape.get('H_KV');out=torch.empty_like(q)
    with strict_fp32_reference(torch):
        for b in range(ctx.shape.get('B')):
            begin,end=(int(inputs['kv_indptr'][b]),int(inputs['kv_indptr'][b+1]))
            positions=inputs['kv_indices'][begin:end]
            k=inputs['k'][positions].repeat_interleave(ratio,dim=1).permute(1,0,2).float()
            v=inputs['v'][positions].repeat_interleave(ratio,dim=1).permute(1,0,2).float()
            scores=(q[b].float()[:,None,:]*k).sum(-1)/math.sqrt(q.shape[-1])
            out[b]=(scores.softmax(-1)[:,:,None]*v).sum(1).to(q.dtype)
    return {'out':out}


def validate(ctx):
    s=ctx.shape
    if s.get('H_Q')%s.get('H_KV'):return 'GQA/MQA requires H_Q divisible by H_KV'
    if s.get('D') not in (32,64,128):return 'controlled native non-MLA head dimensions32/64/128'
    if s.get('PAGE_SIZE') not in (1,16,32):return 'initial native physical page sizes1/16/32'
    if s.get('B')*s.get('MAX_SEQ')*s.get('H_KV')*s.get('D')>16_777_216:return 'bounded KV/reference allocation limit'
    if ctx.operator.parameters['length_pattern'] not in ('equal','ragged'):return 'equal/ragged sequence pattern required'
    return None


def support(h,o,s):
    try:
        importlib.metadata.version('torch');v=importlib.metadata.version('triton')
        if tuple(int(x) for x in v.split('.')[:2])<(3,2):return 'native extracted source requires an audited Triton3.2+ environment; compile support still separately verified'
    except importlib.metadata.PackageNotFoundError:return 'optional Torch/Triton packages unavailable'
    return None


def native_api():
    import torch,triton,triton.language as tl
    ns={'torch':torch,'triton':triton,'tl':tl,'math':math,'logging':logging,
        'NamedTuple':NamedTuple,'Optional':Optional,'Tuple':Tuple,
        'is_hip':lambda:False,'is_gfx95_supported':lambda:False,'is_gfx1250_supported':lambda:False,
        'get_device_core_count':lambda *a,**kw:torch.cuda.get_device_properties(0).multi_processor_count,
        'envs':SimpleNamespace()}
    sources.host_functions('sglang','python/sglang/kernels/ops/attention/score_mod.py',{'unpack_aux_tensors'},ns)
    decode=sources.jit_module('sglang','python/sglang/kernels/ops/attention/decode_attention.py',ns,name='_ktlab_sglang_native_decode')
    scheduler=sources.jit_module('sglang','python/sglang/kernels/ops/attention/metadata.py',
        {'torch':torch,'triton':triton,'tl':tl,'Optional':Optional},name='_ktlab_sglang_native_metadata')
    return decode,scheduler


class _Trace:
    def __init__(self,original,ctx,name,override=None):self.original=original;self.ctx=ctx;self.name=name;self.override=override or {}
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            kwargs.update(self.override);compiled=self.original[grid](*args,**kwargs)
            self.ctx.record_compiled_kernel(self.name,compiled)
            self.ctx.metadata['experiment_annotations'].setdefault('actual_raw_launch',{})[self.name]={k:kwargs.get(k) for k in
                ('BLOCK_N','MIN_BLOCK_KV','MAX_KV_SPLITS','num_warps','num_stages','PAGE_SIZE','FORCED_KV_SPLITS') if k in kwargs}
            return compiled
        return launch


def build(ctx,*,mode,BLOCK_N=None,num_warps=None,num_stages=None,MAX_KV_SPLITS=None):
    native,metadata=native_api();torch=ctx.torch;s=ctx.shape;b,hq,d=(s.get(k) for k in ('B','H_Q','D'))
    # The native caller supplies the cap. This fixed integration profile uses8;
    # the actual GPU scheduler below is upstream, not an approximate host rule.
    cap=8 if mode=='native' else MAX_KV_SPLITS
    partials=ctx.allocate_workspace('paged_partials',(b,hq,cap,d),'float32',fill=0)
    lse=ctx.allocate_workspace('paged_lse',(b,hq,cap),'float32',fill=float('-inf'))
    splits=ctx.allocate_workspace('paged_num_splits',(b,),'int32',fill=1)
    uneven=ctx.allocate_workspace('paged_uneven',(1,),'int32',fill=0)
    if not ctx.hardware.sm_count:raise ValueError('actual device core count required by native GPU split scheduler')
    override={} if mode=='native' else {'BLOCK_N':BLOCK_N,'num_warps':num_warps,'num_stages':num_stages}
    for name in ('_fwd_kernel_stage1','_fwd_grouped_kernel_stage1'):
        setattr(native,name,_Trace(getattr(native,name),ctx,name,override))
    native._fwd_kernel_stage2=_Trace(native._fwd_kernel_stage2,ctx,'_fwd_kernel_stage2')
    scheduler=_Trace(metadata.get_num_kv_splits_triton,ctx,'native_split_scheduler')
    def initialize():
        scheduler[(1,)](splits,ctx.inputs['seq_lens'],b,1,hq,s.get('H_KV'),cap,ctx.hardware.sm_count,
            MAX_NUM_SEQ=max(256,1<<(b-1).bit_length()),uneven_batch_ptr=uneven)
    annotations=ctx.metadata['experiment_annotations'];annotations.update(
        backend='SGLang native Triton',source_identity=sources.pin('sglang'),mode=mode,
        requested_configuration={'selection':'native','profile_max_split_cap':8} if mode=='native' else
            {'BLOCK_N':BLOCK_N,'num_warps':num_warps,'num_stages':num_stages,'MAX_KV_SPLITS':MAX_KV_SPLITS},
        native_policy_scope='original GPU per-sequence split scheduler, standard non-MLA CUDA wrapper; profile cap explicit, not inferred from server defaults',
        workspace_policy={'partials':[b,hq,cap,d],'lse':[b,hq,cap],'merge_cap':cap,'min_block_KV':32},
        scope='native GPU split scheduling + stage1 paged/gqa partials + stage2 merge, no hidden continuous-KV substitute')
    initialize();torch.cuda.synchronize();annotations['actual_per_sequence_splits']=splits.cpu().tolist()
    def invoke():
        native.decode_attention_fwd(ctx.inputs['q'],ctx.inputs['k'],ctx.inputs['v'],ctx.outputs['out'],ctx.inputs['kv_indptr'],ctx.inputs['kv_indices'],
            partials,lse,splits,cap,1/math.sqrt(d),1.,1.,page_size=s.get('PAGE_SIZE'),has_mla=False,use_pdl=False,uneven_batch=uneven)
    return ExecutionPlan((ExecutionStep('native_paged_stage1_and_stage2',invoke),),initialize=initialize)


def register_assets(catalog):
    src=(__file__,sources.__file__,str(sources.ROOT/'manifest.json'))
    op=OperatorDefinition('production/paged_decode','1.0.0',('B','H_Q','H_KV','D','MAX_SEQ','PAGE_SIZE'),('float16','bfloat16'),
        generate,reference,parameters={'length_pattern':'ragged'},validate=validate,
        tolerances={'float16':(.005,.005),'bfloat16':(.02,.02)},source_files=src,
        validation_shapes=(ShapeSpec({'B':4,'H_Q':8,'H_KV':2,'D':64,'MAX_SEQ':129,'PAGE_SIZE':16}),),
        metadata={'semantics':'one query per sequence, true paged KV indptr/physical-slot indices, nonuniform lengths, GQA/MQA, scales1; no paged-to-contiguous execution replacement'})
    catalog.register(op)
    for mode in ('native','explicit'):
        space=SpaceDefinition({k:ParameterSpec(v) for k,v in KNOBS.items()} if mode=='explicit' else {},name='SGLang native '+mode+' launch profile')
        catalog.register(KernelDefinition('production/sglang_paged_'+mode,'1.0.0',op.ref,partial(build,mode=mode),space,
            runtime='cuda',supports=support,source_files=src,execution_identity=partial(environment.vendored_identity,'sglang'),
            metadata={'pin':sources.pin('sglang'),'numerical_body':'unchanged pinned upstream JIT definitions',
            'launch_patch':'explicit wrapper proxy changes only actual stage1 tile/warp/stage; scheduler cap/grid/workspace/stage2 coherent'}))
