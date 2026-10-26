"""CPU-safe discovery of the explicitly runtime-shape T1 control."""
from pathlib import Path

from ..sdk import KernelDefinition, ParameterSpec, SpaceDefinition
from .t1_gemm import DOMAINS, DESCRIPTIONS, resource_constraint, supports


def build(ctx, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M):
    import inspect
    import os
    import triton
    if 'do_not_specialize_on_alignment' not in inspect.signature(triton.jit).parameters:
        raise RuntimeError('Shape-generic T1 requires triton.jit do_not_specialize_on_alignment; validated with Triton 3.7.1')
    from ..backends._triton_t1_generic import RUNTIME_SCALARS, t1_gemm_generic
    a,b,c=ctx.inputs['a'],ctx.inputs['b'],ctx.outputs['out']
    m,n,k=(ctx.shape.get(name) for name in ('M','N','K'))
    grid=(triton.cdiv(m,BLOCK_M)*triton.cdiv(n,BLOCK_N),)
    args=(a,b,c,m,n,k,*a.stride(),*b.stride(),*c.stride())
    config=dict(BLOCK_M=BLOCK_M,BLOCK_N=BLOCK_N,BLOCK_K=BLOCK_K,num_warps=num_warps,num_stages=num_stages,GROUP_M=GROUP_M)
    compiled=t1_gemm_generic.warmup(*args,grid=grid,**config)
    ctx.record_compiled_kernel('t1_gemm_generic',compiled)
    resource=ctx.metadata['execution_resources']['t1_gemm_generic']
    resource.update(runtime_scalar_names=list(RUNTIME_SCALARS),shape_arguments_constexpr=False,
        do_not_specialize=list(t1_gemm_generic.do_not_specialize),
        do_not_specialize_on_alignment=list(t1_gemm_generic.do_not_specialize_on_alignment),
        process_id=os.getpid(),compiled_object_instance=id(compiled),
        compiler_signature={str(key):str(value) for key,value in compiled.src.signature.items()},
        compiler_constants={str(key):repr(value) for key,value in compiled.src.constants.items()},
        grid=list(grid),shape=dict(ctx.shape.dims),strides={'a':list(a.stride()),'b':list(b.stride()),'out':list(c.stride())})
    def launch():
        actual=t1_gemm_generic[grid](*args,**config)
        if actual.hash!=compiled.hash:
            raise RuntimeError('Actual launch implicitly specialized differently from the recorded warmup compilation')
        resource['actual_launch_compiler_identity']=actual.hash
        resource['actual_launch_object_instance']=id(actual)
    return launch


def register_assets(catalog):
    source=str(Path(__file__).parents[1]/'backends'/'_triton_t1_generic.py')
    catalog.register(KernelDefinition('triton/t1_gemm_generic','1.0.0','gemm@1.0.0',build,
        SpaceDefinition({key:ParameterSpec(values,description=DESCRIPTIONS[key]) for key,values in DOMAINS.items()},
            constraints=(resource_constraint,),name='t1_gemm_generic_v1'),
        runtime='cuda',dtypes=('float16','bfloat16'),supports=supports,source_files=(source,),
        metadata={'requires_triton':True,'minimum_architecture':'sm_80',
            'description':'T1 runtime int32 M/N/K and strides, value/alignment specialization disabled',
            'shape_specialization':'runtime scalars; dtype, pointer alignment, tile/warps/stages and target still compile-time',
            'validated_triton_api':'3.7.1; requires do_not_specialize_on_alignment',
            'license':'MIT (distributed third_party/triton-tutorial-LICENSE)',
            'source':'https://triton-lang.org/main/python-api/generated/triton.jit.html'}))
