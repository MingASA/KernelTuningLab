"""Generate or explicitly execute a small target-GPU smoke via existing v2.

Default is static generation: no plugin/GPU imports or measurement. H20 has no
guessed profile. Target execution validates the actual device before searching.
"""
import argparse
import json
from pathlib import Path


def prepare(target,output,execute=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    profiles={'h20':None,'h200':'H200','h800':'H800','4090':'4090','5060ti':'5060ti16g'}
    spec={'schema_version':1,'kind':'research','name':'target-smoke-'+target,
        'algorithms':['random@1.0.0','heuristic@1.0.0'],'dtypes':['float16','bfloat16'],'seeds':[7],
        'budgets':[{'max_evaluations':2,'max_wall_time_s':60}],
        'defaults':{'hardware':profiles[target],'warmup':3,'repetitions':9,'timing_mode':'cuda_graph','cuda_graph_batch_size':512,
            'validation':{'max_configs':2,'max_cases':0,'max_wall_time_s':60,'trust_claims':False}},
        'cases':[{'name':'t1-smoke','operator':'gemm@1.0.0','kernels':[{'asset':'triton/t1_gemm@1.0.0',
            'space':{'BLOCK_M':[16,32],'BLOCK_N':[64],'BLOCK_K':[32],'num_warps':[4],'num_stages':[3],'GROUP_M':[1]}}],
            'shapes':[{'M':1,'N':128,'K':128},{'M':33,'N':129,'K':65}]}]}
    path=output/'target-smoke.json';text=json.dumps(spec,indent=2)+'\n'
    if path.exists() and path.read_text()!=text:raise ValueError('refuse to overwrite a different target experiment')
    path.write_text(text)
    if execute:
        from dataclasses import asdict
        import re
        from kernel_tuning_lab.backends import get_adapter
        adapter=get_adapter('triton');actual=adapter.probe_hardware()
        if target=='h20' and not re.search(r'\bH20\b',actual.name,re.IGNORECASE):
            raise ValueError('H20 target requires an actual H20 device name; profile remains unknown')
        (output/'actual-probe.json').write_text(json.dumps({'hardware':asdict(actual),'environment':adapter.environment()},indent=2)+'\n')
    from kernel_tuning_lab.research import run_research
    result=run_research(path,output/'study',write_only=not execute)
    print(json.dumps({'requested_target':target,'executed':execute,'experiment':str(path),
        'status':result['status'],'exit_code':result.get('exit_code',0),
        'notice':'Static generation is not target GPU validation. Profiles and old 5060 Ti binaries are not hardware evidence.'},indent=2))
    return result.get('exit_code',0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target',choices=('h20','h200','h800','4090','5060ti'),required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--execute',action='store_true',help='actually probe/compile/measure on this target GPU')
    args=parser.parse_args();raise SystemExit(prepare(args.target,args.output,args.execute))
