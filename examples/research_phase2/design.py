"""Outcome-independent C48/readiness design, frozen before phase2 GPU study.

C48 is a controlled *non-nested* successor to C64, not the full 192-config SDK
space. GROUP/stage controls remain separate. Historical geometry is not blind.
"""
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SPACE={'BLOCK_M':[16,32,64,128],'BLOCK_N':[32,64,128],'BLOCK_K':[32,64],
       'num_warps':[4,8],'num_stages':[3],'GROUP_M':[1]}
SEEDS=[7,19]
HISTORICAL=json.loads((ROOT/'examples/research_phase1/shape-design.json').read_text())['cases']
SHAPES=[{'name':row['name'],'split':'train','shape':row['shape'],'role':'historically_explored',
         'hypothesis':row['hypothesis']} for row in HISTORICAL]+[
 {'name':name,'split':split,'shape':dict(zip(('M','N','K'),dims)),'role':'new_heldout', 'hypothesis':reason}
 for name,split,dims,reason in [
 ('low-2','test',(2,2048,2048),'M2 between prior decode and token batches; 8 MiB weight'),
 ('low-4','test',(4,2048,2048),'M4 low-M boundary, same 8 MiB B'),
 ('low-8','validation',(8,2048,2048),'M8 with small working set; prior M8 used N8192'),
 ('low-16','test',(16,2048,2048),'M16 exactly one smallest tile; prior M16 used N4096'),
 ('low-32','test',(32,2048,2048),'M32 tile transition at same N/K'),
 ('low-64','validation',(64,2048,2048),'M64 two/four/single M tiles depending on candidate'),
 ('edge-31','test',(31,2048,2048),'Just below M32 tile boundary'),
 ('edge-33','test',(33,2048,2048),'Just above M32 tile boundary'),
 ('edge-65','test',(65,2048,2048),'Just above M64 tile boundary'),
 ('edge-129','validation',(129,2048,2048),'Just above M128 tile boundary'),
 ('working-set-16m','test',(4,2048,4096),'M4 / 16 MiB B below L2, greater K'),
 ('working-set-64m','test',(4,8192,4096),'M4 / 64 MiB B beyond L2, paired with 16 MiB case'),
 ('new-tail','test',(63,1031,1025),'All-axis tails and stride/alignment variation'),
 ('new-compute','test',(768,1536,3072),'New large-M geometry, rectangular compute hypothesis')]]


def configs():return [dict(zip(SPACE,values)) for values in itertools.product(*SPACE.values())]


def case(entry,kernel='triton/t1_gemm@1.0.0',space=SPACE):
    return {'name':entry['name'],'split':entry['split'],'operator':'gemm@1.0.0',
        'kernels':[{'asset':kernel,'space':space}],'shapes':[entry['shape']]}


def spec(name,cases,count,seeds=SEEDS,selected=None,dtypes=('float16',),wall=600):
    algorithm='phase1/cohort@1.0.0' if selected is None else {'asset':'phase1/cohort@1.0.0','options':{'configs':selected}}
    return {'schema_version':1,'kind':'research','name':name,'plugins':['../research_phase1/cohort.py'],
        'algorithms':[algorithm],'dtypes':list(dtypes),'seeds':list(seeds),
        'budgets':[{'max_evaluations':count,'max_wall_time_s':wall}],
        'defaults':{'warmup':10,'repetitions':31,'timing_mode':'cuda_graph','cuda_graph_batch_size':512,
            'preflight_seconds':60,'validation':{'max_configs':4,'max_cases':0,'max_wall_time_s':90,'trust_claims':False}},
        'cases':cases}


def generate(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    primary=spec('phase2-t1-c48-v1',[case(s) for s in SHAPES],48)
    selected=[configs()[i] for i in (0,3,7,12,19,25,36,47)]
    pilot=spec('phase2-t1-pilot-v1',[case(SHAPES[i]) for i in (0,2,4,5)],8,selected=selected,wall=180)
    train=json.loads(json.dumps(primary));train['cases']=[c for c in train['cases'] if c['split']=='train']
    validation=json.loads(json.dumps(primary));validation['cases']=[c for c in validation['cases'] if c['split']=='validation']
    test=json.loads(json.dumps(primary));test['cases']=[c for c in test['cases'] if c['split']=='test']
    repeat=spec('phase2-repeat-v1',[case(SHAPES[i]) for i in (0,4,5,11,17,23)],48,seeds=(31,))
    control_space={**SPACE,'BLOCK_N':[64,128],'BLOCK_K':[32],'num_warps':[4],'num_stages':[2,3],'GROUP_M':[1,8]}
    base=[dict(BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=32,num_warps=4,num_stages=3,GROUP_M=1)
          for bm,bn in ((16,64),(32,128),(64,64),(128,128))]
    controls=[c for original in base for c in (original,{**original,'num_stages':2},{**original,'GROUP_M':8})]
    group_stage=spec('phase2-stage-group-control-v1',[case(SHAPES[i],space=control_space) for i in (0,2,5,13,17,23)],12,selected=controls)
    generic_configs=[dict(BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,num_warps=4,num_stages=3,GROUP_M=1)
        for bm,bn,bk in ((16,32,32),(16,64,64),(32,128,32),(64,64,32))]
    generic=spec('phase2-generic-poc-v1',[{'name':'generic-matched','split':'test','operator':'gemm@1.0.0',
        'kernels':[{'asset':ref,'space':SPACE} for ref in ('triton/t1_gemm@1.0.0','triton/t1_gemm_generic@1.0.0')],
        'shapes':[SHAPES[i]['shape'] for i in (0,5,8,10,17,22)]}],4,selected=generic_configs)
    tails=[{'M':m,'N':n,'K':k} for m,n,k in ((1,1,1),(15,17,31),(33,129,65),(63,1031,1025))]
    correctness=spec('phase2-generic-correctness-v1',[{'name':'generic-tails','operator':'gemm@1.0.0',
        'kernels':[{'asset':'triton/t1_gemm_generic@1.0.0','space':SPACE}],'shapes':tails}],4,
        seeds=(7,),selected=generic_configs,dtypes=('float16','bfloat16'))
    bf16=spec('phase2-bf16-v1',[case(SHAPES[i]) for i in (0,5,8,17)],8,selected=selected,dtypes=('bfloat16',))
    online=spec('phase2-online-v1',[case(SHAPES[i]) for i in (11,17,23)],8,seeds=(7,))
    online['algorithms']=['random@1.0.0','heuristic@1.0.0','surrogate@1.0.0']
    wspace={'BLOCK_M':[32,128],'BLOCK_N':[128],'BLOCK_K':[32],'num_warps':[4],
            'num_stages':[2],'GROUP_M':[1,8],'SPLIT_K':[1,2,4]}
    w4=spec('phase2-w4-unseen-v1',[{'name':'w4-unseen','operator':'w4a16_gemm@1.0.0','split':'test',
        'kernels':[{'asset':'triton/w4a16_gemm@1.0.0','space':wspace},
                   {'asset':'triton_pipeline/w4a16_gemm@1.0.0','space':{**wspace,'DEQUANT_BLOCK':[512]}}],
        'shapes':[dict(zip(('M','N','K'),dims)) for dims in ((4,2048,2048),(64,2048,2048),(257,1031,1025))]}],12)
    outputs={'matrix':primary,'train':train,'validation':validation,'test':test,'pilot':pilot,
        'repeat':repeat,'controls':group_stage,'generic':generic,'generic-correctness':correctness,'bf16':bf16,'online':online,'w4':w4}
    for name,value in outputs.items():(directory/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    manifest={'schema_version':1,'primary_cohort':'C48; fixed stage3/GROUP1, not a superset of C64',
        'space':SPACE,'configs':configs(),'shapes':SHAPES,'seeds':SEEDS,'physical_observations':len(SHAPES)*48*len(SEEDS),
        'validation_allowance':len(SHAPES)*len(SEEDS)*4,'train_shapes':10,'validation_shapes':3,'test_shapes':11,
        'metrics':['per-Shape finite-set regret','within 5%/10%','hitting time','seed/Shape/run variation',
            'parameter matched effects','Top overlap','compiler identities','source/new/cache/acceptance/job costs'],
        'search_budgets':[4,8,16,32],'search_seeds':list(range(5)),
        'prior_freeze':'train prefix only, before new validation/test acquisition; historical C64 labels stay historical',
        'physical_replication':'2 input seeds for all cells; third seed for 6 predeclared Shapes in separate study',
        'generic_configs':generic_configs,'controls':controls,
        'resource_bound':'largest primary A/B staging estimate 98304 B; exact compiled check remains authoritative',
        'wall_budget':'pilot 10 min cap; primary 90 min per invocation, resume if needed after pilot; serial GPU only'}
    for entry in manifest['shapes']:
        m,n,k=(entry['shape'][key] for key in ('M','N','K'))
        entry['fp16_input_output_bytes']=2*(m*k+k*n+m*n);entry['weight_bytes']=2*k*n
    (directory/'frozen-design.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':generate(Path(__file__).resolve().parent)
