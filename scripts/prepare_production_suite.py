"""Write bounded native-default/explicit acceptance specs; initializes no GPU.

Use one backend/environment at a time. Execute generated specs through ktlab
research, which owns workers, validation, timeouts, storage and recovery.
"""
import argparse
import json
from pathlib import Path


def specification(backend,dtype='float16',bits=4):
    module='kernel_tuning_lab.production.'+backend
    if backend=='gemlite':
        if bits not in (2,4):raise ValueError('GemLite suite explicitly separates W2 and W4')
        op='production/uniform_lowbit_gemm@1.0.0';params={'bits':bits,'group_size':128}
        cases=[]
        for m,family in ((1,'gemv_revsplitk'),(33,'gemm_splitk'),(128,'gemm')):
            cases.append({'name':'w'+str(bits)+'-M'+str(m),'operator':op,'parameters':params,
                'kernels':['production/gemlite_default@1.0.0','production/gemlite_'+family+'@1.0.0'],
                'shapes':[{'M':m,'N':512,'K':512}]})
    elif backend=='flute':
        cases=[{'name':'int3-template-packing','operator':'production/uniform_lowbit_gemm@1.0.0',
            'parameters':{'bits':3,'group_size':128},'kernels':['production/flute_w3_default@1.0.0','production/flute_w3_explicit@1.0.0'],
            'shapes':[{'M':128,'N':2048,'K':2048}]}]
    elif backend=='flex_attention':
        cases=[{'name':mask,'operator':'production/attention_prefill@1.0.0','parameters':{'mask':mask},
            'kernels':['production/flex_auto@1.0.0','production/flex_triton@1.0.0'],
            'shapes':[{'B':1,'H_Q':4,'H_KV':2,'D':64,'Q_LEN':129,'KV_LEN':129}]} for mask in ('dense','causal')]
    elif backend=='sglang_decode':
        cases=[{'name':pattern,'operator':'production/paged_decode@1.0.0','parameters':{'length_pattern':pattern},
            'kernels':['production/sglang_paged_native@1.0.0','production/sglang_paged_explicit@1.0.0'],
            'shapes':[{'B':4,'H_Q':8,'H_KV':2,'D':64,'MAX_SEQ':129,'PAGE_SIZE':16}]} for pattern in ('equal','ragged')]
    elif backend=='vllm_moe':
        cases=[{'name':pattern,'operator':'production/routed_moe_silu@1.0.0','parameters':{'route_distribution':pattern},
            'kernels':['production/vllm_moe_native@1.0.0','production/vllm_moe_explicit@1.0.0'],
            'shapes':[{'M':17,'H':128,'I':256,'E':4,'TOP_K':2}]} for pattern in ('balanced','skewed')]
    else:raise ValueError('unknown backend')
    return {'schema_version':1,'kind':'research','name':'production-acceptance-'+backend+'-'+dtype+('-w'+str(bits) if backend=='gemlite' else ''),
        'plugins':[module],'dtypes':[dtype],'seeds':[17],
        'algorithms':[{'asset':'grid@1.0.0'}],'budgets':[{'max_evaluations':3,'max_wall_time_s':180}],
        'defaults':{'warmup':5,'repetitions':15,'timing_mode':'cuda_graph','cuda_graph_batch_size':64,
            'workspace_memory_limit_bytes':512*1024**2,'preflight_seconds':45,
            'validation':{'max_configs':3,'max_cases':0,'max_wall_time_s':240,'trust_claims':False}},
        'cases':cases,'max_jobs':16}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('gemlite','flute','flex_attention','sglang_decode','vllm_moe'),required=True)
    parser.add_argument('--dtype',choices=('float16','bfloat16'),default='float16');parser.add_argument('--bits',type=int,default=4)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    spec=specification(args.backend,args.dtype,args.bits);path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('do not overwrite an existing frozen acceptance spec; use a new output')
    path.write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps({'spec':str(path),'cases':len(spec['cases']),'max_research_observations':sum(len(c['kernels'])*len(c['shapes'])*3 for c in spec['cases']),
        'native_default_internal_tuning':'additional bounded worker time, not platform observation count','GPU_execution':False,
        'command':['ktlab','research','--spec',str(path),'--output','artifacts/production-acceptance/'+spec['name']]}))


if __name__=='__main__':main()
