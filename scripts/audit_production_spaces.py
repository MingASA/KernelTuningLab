"""Bounded CPU audit of pinned production spaces (never compiles or uses CUDA)."""
import argparse
from dataclasses import replace
import json
import random
import time
import tracemalloc
from pathlib import Path

from kernel_tuning_lab.catalog import AssetCatalog
from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec
from kernel_tuning_lab.sdk import SpaceBuildBudget
from kernel_tuning_lab.production import gemlite,flute,flex_attention,sglang_decode,vllm_moe


def audit(*,max_nodes=100000,wall_s=3,sm_count=0):
    catalog=AssetCatalog()
    for provider in (gemlite,flute,flex_attention,sglang_decode,vllm_moe):provider.register_assets(catalog)
    hardware=HardwareSpec('CPU audit planning context; not a physical probe',sm_count=sm_count)
    shapes={'production/uniform_lowbit_gemm':{'M':128,'N':2048,'K':2048},
        'production/attention_prefill':{'B':1,'H_Q':4,'H_KV':2,'D':64,'Q_LEN':129,'KV_LEN':129},
        'production/paged_decode':{'B':4,'H_Q':8,'H_KV':2,'D':64,'MAX_SEQ':129,'PAGE_SIZE':16},
        'production/routed_moe_silu':{'M':17,'H':128,'I':256,'E':4,'TOP_K':2}}
    rows=[]
    for kernel in catalog.list('kernel'):
        definition=catalog.resolve('operator',kernel.operator)
        parameters=dict(definition.parameters)
        if definition.id=='production/uniform_lowbit_gemm':parameters.update(bits=3 if 'flute' in kernel.id else 4,group_size=128)
        operator=OperatorSpec(definition.id,dtype='float16',parameters=parameters)
        shape=ShapeSpec(shapes[definition.id]);started=time.perf_counter();tracemalloc.start()
        # Dependency availability is a runtime axis, not a source legality
        # theorem. Explicitly bypass ONLY that callback for this CPU audit.
        # Original structured constraints and opaque hard checks remain intact.
        dependency_status=kernel.supports(hardware,operator,shape) if kernel.supports else None
        space=build_effective_space(replace(kernel,supports=None),hardware,operator,shape)
        built=space.build_feasible(hardware,operator,shape,budget=SpaceBuildBudget(max_nodes=max_nodes,max_wall_time_s=wall_s,max_configs=50000,max_bytes=32*1024**2))
        construction=time.perf_counter()-started;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        start=time.perf_counter();sample=space.complete({},hardware,operator,shape,count=min(16,space.cardinality),rng=random.Random(7),
            budget=SpaceBuildBudget(max_nodes=20000,max_wall_time_s=1,max_configs=16))
        rows.append({'kernel_ref':kernel.ref,'kernel_fingerprint':kernel.fingerprint,'shape':dict(shape.dims),'parameters':parameters,
            'declared_cardinality':space.cardinality,'exact_static_feasible':len(built.configs) if built.complete else None,
            'static_found':len(built.configs),'build_status':built.status,'build_diagnostics':dict(built.diagnostics),
            'build_wall_s':construction,'python_peak_traced_bytes':peak,'peak_scope':'Python allocations only, not RSS/GPU memory',
            'sample_count':len(sample.configs),'sample_wall_s':time.perf_counter()-start,'sample_status':sample.status,
            'sample_diagnostics':dict(sample.diagnostics),'sample_distribution':'nonuniform bounded constructive restarts; not guaranteed uniform',
            'source_candidate_count':kernel.metadata.get('source_template_count',kernel.metadata.get('source_declared_candidates')),
            'compiled_candidate_set':kernel.metadata.get('compiled_template_set'),
            'GPU_compile_success_rate':None,'GPU_correctness_success_rate':None,
            'runtime_dependency_status':dependency_status or 'no dependency refusal; GPU not executed',
            'static_scope':'source-declared constraints in planning hardware context; runtime availability separately reported',
            'source_pin':kernel.metadata.get('pin')})
    return {'evidence':'CPU static space audit only; no GPU or native dependency import','planning_sm_count':sm_count,
        'limits':{'max_nodes':max_nodes,'build_wall_s_per_kernel':wall_s},'spaces':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--max-nodes',type=int,default=100000);parser.add_argument('--wall-seconds',type=float,default=3)
    parser.add_argument('--planning-sm-count',type=int,default=0)
    args=parser.parse_args();result=audit(max_nodes=args.max_nodes,wall_s=args.wall_seconds,sm_count=args.planning_sm_count)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'spaces':len(result['spaces']),'output':str(out),'GPU_execution':False}))


if __name__=='__main__':main()
