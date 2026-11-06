"""Portable logical inputs shared by real low-bit production backends."""
from __future__ import annotations
import hashlib
from .inputs import _counter
from ..torch_context import strict_fp32_reference


def generate_quant(ctx,seed):
    import numpy as np
    torch=ctx.torch;m,n,k=(ctx.shape.get(x) for x in ('M','N','K'))
    bits=ctx.operator.parameters['bits'];group=ctx.operator.parameters['group_size']
    codes=(_counter(n*k,seed^0x17a2c4b3)&((1<<bits)-1)).reshape(n,k).astype(np.uint8)
    scales=(1+(_counter(n*(k//group),seed^0x43a7b5c1)&3)).reshape(n,k//group).astype(np.float32)/64
    a=((_counter(m*k,seed^0x5b93aef1)%33).astype(np.int16)-16).reshape(m,k).astype(np.float32)/16
    dtype=getattr(torch,ctx.operator.dtype);device=f'cuda:{ctx.device}'
    ctx.allocate_output('out',(m,n));ctx.metadata['experiment_annotations']={
        'logical_format':'row-major N,K unsigned codes; signed uniform LUT; group scales along K',
        'logical_input_identity':{'codes':hashlib.sha256(codes.tobytes()).hexdigest(),
            'scales':hashlib.sha256(scales.tobytes()).hexdigest(),'a':hashlib.sha256(a.tobytes()).hexdigest(),
            'bits':bits,'zero':1<<(bits-1),'group_size':group},
        'layout_claim':'backend-specific packing, not arbitrary GPTQ/AWQ/EXL2 checkpoint import'}
    return {'a':torch.from_numpy(a).to(device,dtype=dtype),
        'codes':torch.from_numpy(codes).to(device),'scales':torch.from_numpy(scales).to(device,dtype=dtype)}


def reference_quant(ctx):
    torch=ctx.torch;group=ctx.operator.parameters['group_size'];zero=1<<(ctx.operator.parameters['bits']-1)
    with strict_fp32_reference(torch):
        weights=(ctx.inputs['codes'].float()-zero)*ctx.inputs['scales'].float().repeat_interleave(group,dim=1)
        return {'out':(ctx.inputs['a'].float()@weights.T).to(getattr(torch,ctx.operator.dtype))}


def validate_quant(ctx):
    bits=ctx.operator.parameters['bits'];group=ctx.operator.parameters['group_size']
    if bits not in (2,3,4) or group not in (64,128):return 'fixed uniform int2/int3/int4, group64/128 semantics only'
    if ctx.shape.get('K')%group:return 'K must be divisible by the semantic group size; no hidden padding'
    if ctx.shape.get('N')*ctx.shape.get('K')>16_777_216:return 'bounded preparation supports at most16M logical weights'
    return None


def operator_definition():
    from ..sdk import OperatorDefinition,ShapeSpec
    from pathlib import Path
    return OperatorDefinition('production/uniform_lowbit_gemm','1.0.0',('M','N','K'),('float16','bfloat16'),
        generate_quant,reference_quant,parameters={'bits':4,'group_size':128},
        tolerances={'float16':(.01,.01),'bfloat16':(.02,.02)},validate=validate_quant,
        # Large enough for original FLUTE StreamK support as well as GemLite;
        # a tinyN128 generic fixture cannot represent TileP32/64 packing.
        validation_shapes=(ShapeSpec({'M':128,'N':2048,'K':2048}),),
        source_files=(__file__,str(Path(__file__).with_name('inputs.py')),
            str(Path(__file__).parents[1]/'torch_context.py')),
        metadata={'semantics':'A@(((unsigned integer codes-zero)*group scales).T); logical weightsN,K; uniform LUT; not arbitrary checkpoint import'})
