"""Portable logical W4 inputs and audited conversion to two real packed layouts."""
from __future__ import annotations

import __future__
import ast
import functools
import hashlib
import json
from pathlib import Path
import time

UPSTREAM=Path(__file__).with_name('upstream')


def source_functions(filename,names,namespace):
    """Execute exact pinned upstream function bodies, without unrelated imports.

    This is trusted dependency loading, not a sandbox. No user-generated code
    or performance labels are accepted here.
    """
    path=UPSTREAM/filename;tree=ast.parse(path.read_text());selected=[]
    expected=json.loads((UPSTREAM/'SOURCES.json').read_text())['files'][filename]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('fixed upstream source changed: '+filename)
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:selected.append(node)
    if {n.name for n in selected}!=set(names):raise ValueError('pinned upstream functions missing')
    module=ast.Module(body=selected,type_ignores=[])
    exec(compile(module,str(path),'exec',flags=__future__.annotations.compiler_flag),namespace)
    return namespace


@functools.lru_cache(maxsize=1)
def upstream_cpu():
    import numpy as np
    import torch
    namespace={'np':np,'numpy':np,'torch':torch,'GPTQ_MARLIN_TILE':16}
    source_functions('quant_utils.py',{'get_pack_factor'},namespace)
    source_functions('marlin_utils.py',{'get_scale_perms','marlin_permute_scales'},namespace)
    source_functions('marlin_utils_test.py',{'marlin_permute_weights','marlin_weights','get_weight_perm'},namespace)
    return namespace


def _counter(count,seed):
    import numpy as np
    x=np.arange(count,dtype=np.uint32)^np.uint32(seed & 0xffffffff)
    x=(x^(x>>16))*np.uint32(0x7feb352d)
    x=(x^(x>>15))*np.uint32(0x846ca68b)
    return x^(x>>16)


def portable_arrays(m,n,k,seed):
    import numpy as np
    if k%128 or n%64 or min(m,n,k)<1:raise ValueError('common Marlin cohort requires K%128=N%64=0 and positive dimensions')
    codes=(_counter(k*n,seed^0x17a2c4b3)&15).reshape(k,n).astype(np.int32)
    scales=(1+(_counter(k//128*n,seed^0x43a7b5c1)&3)).reshape(k//128,n).astype(np.float16)*np.float16(1/64)
    a=((_counter(m*k,seed^0x5b93aef1)%33).astype(np.int16)-16).reshape(m,k).astype(np.float16)*np.float16(1/16)
    packed=np.zeros((k//8,n),dtype=np.uint32)
    for index in range(8):packed|=codes[index::8].astype(np.uint32)<<(4*index)
    return a,codes,scales,packed.view(np.int32)


def _digest(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def generate(ctx,seed):
    import numpy as np
    torch=ctx.torch;start=time.perf_counter();m,n,k=(ctx.shape.get(key) for key in ('M','N','K'))
    a,codes,scales,packed=portable_arrays(m,n,k,seed)
    helpers=upstream_cpu();codes_cpu=torch.from_numpy(codes);scales_cpu=torch.from_numpy(scales)
    marlin=helpers['marlin_weights'](codes_cpu,k,n,4,helpers['get_weight_perm'](4))
    permuted=helpers['marlin_permute_scales'](scales_cpu,k,n,128)
    # Independent bit-unpack of the canonical/GPTQ K-word representation.
    decoded=np.empty_like(codes)
    for index in range(8):decoded[index::8]=((packed.view(np.uint32)>>(index*4))&15).astype(np.int32)
    if not np.array_equal(decoded,codes):raise ValueError('logical weight conversion failed')
    device=f'cuda:{ctx.device}'
    inputs={'a':torch.from_numpy(a).to(device),'packed':torch.from_numpy(packed).to(device),
        'scales':scales_cpu.to(device),'zero_points':torch.full((k//128,n),8,device=device,dtype=torch.int32),
        'marlin_packed':marlin.to(device),'marlin_scales':permuted.to(device),
        'empty_g_idx':torch.empty(0,device=device,dtype=torch.int32),'empty_perm':torch.empty(0,device=device,dtype=torch.int32)}
    ctx.allocate_output('out',(m,n))
    ctx.metadata.update({'input_variant':'portable_counter_u32_v1','packing_format':'canonical_bitstream_v1 plus verified GPTQ4/Marlin conversion',
        'quantization_bits':4,'group_size':128,'logical_input_identity':{'a_sha256':_digest(a),'codes_sha256':_digest(codes),'scales_sha256':_digest(scales),'zero_point':8},
        'backend_packing_identity':{'canonical_gptq_k_words_sha256':_digest(packed),'marlin_packed_sha256':_digest(marlin.numpy()),'marlin_scales_sha256':_digest(permuted.numpy()),'logical_roundtrip':'passed independent bit unpack'},
        'host_input_and_prepack_wall_ms':(time.perf_counter()-start)*1000})
    ctx.metadata['experiment_annotations']={key:ctx.metadata[key] for key in
        ('input_variant','packing_format','logical_input_identity','backend_packing_identity','host_input_and_prepack_wall_ms')}
    return inputs


def validate(ctx):
    if ctx.runtime!='cuda':return 'This common production cohort uses CUDA implementations; CPU checks use portable_arrays directly'
    if ctx.operator.dtype!='float16':return 'Initial production cohort is FP16 only'
    if dict(ctx.operator.parameters)!={'group_size':128}:return 'Initial common semantics are group128 symmetric uint4 bias8'
    if set(ctx.shape.dims)!={'M','N','K'}:return 'M/N/K required'
    m,n,k=(ctx.shape.get(key) for key in ('M','N','K'))
    if k%128 or n%64:return 'Shared production cohort requires K%128=0 and N%64=0; no implicit padding'
    if max(m*k,k*n,m*n)>=2**31:return 'signed indexing limit'
    if k*n>16_777_216:return 'bounded host input/prepack cohort limited to16M weight elements'
    return None
