"""Frozen small hardware-transfer cohort, selected before target measurements.

The six geometries and expert candidates are informed by 5060 Ti exploration;
they are not a blind test on that GPU. Remote outcome data must not tune them.
"""
import hashlib
import json
from pathlib import Path

RUNTIME_BASE='35ac36ed66becf19271c9ed3e97950fb67a03d87'
SPACE={'BLOCK_M':[16,32,64,128],'BLOCK_N':[32,64,128],'BLOCK_K':[32,64],
       'num_warps':[4,8],'num_stages':[3,2],'GROUP_M':[1,8]}
SHAPES=[
    {'name':'decode-small','dims':{'M':1,'N':2048,'K':2048},'hypothesis':'Low M, 8 MiB B; tile utilization and CTA count'},
    {'name':'decode-weight64m','dims':{'M':4,'N':8192,'K':4096},'hypothesis':'Low M, 64 MiB B exceeds measured 5060 Ti L2; compare with actual target cache sizes'},
    {'name':'tile-edge33','dims':{'M':33,'N':2048,'K':2048},'hypothesis':'Just above BM32, below BM64; additional M tile versus padding'},
    {'name':'medium','dims':{'M':512,'N':1024,'K':1024},'hypothesis':'Medium compute and multi-CTA scheduling'},
    {'name':'compute-large','dims':{'M':1024,'N':2048,'K':2048},'hypothesis':'Larger compute workload and pipeline/tile tradeoffs'},
    {'name':'tail','dims':{'M':63,'N':1031,'K':1025},'hypothesis':'Irregular strides and tails on all axes, near M64'}]

# BM, BN, BK, warps, stages, GROUP. Kept explicit: not the 192-point Cartesian domain.
PARAMETERS=[
    (16,32,32,4,3,1), (32,128,32,4,3,1), (16,64,64,8,3,1), (16,64,64,4,3,1),
    (32,64,64,4,3,1), (64,64,64,4,3,1), (32,64,64,8,3,1), (64,128,64,8,3,1),
    (128,64,32,4,3,1), (16,128,32,4,3,1), (16,128,64,4,3,1), (64,64,32,4,3,1),
    (32,128,32,4,2,1), (32,128,32,8,3,1), (64,64,32,4,3,8), (32,128,32,4,3,8)]
CONFIGS=[dict(zip(('BLOCK_M','BLOCK_N','BLOCK_K','num_warps','num_stages','GROUP_M'),values)) for values in PARAMETERS]
SEEDS=[7,19]
PROTOCOL={'dtype':'float16','warmup':10,'repetitions':31,'timing_mode':'cuda_graph','cuda_graph_batch_size':512,
          'input_seed_role':'independent physical input/run repeats; search RNG is separate in Replay',
          'layout':'contiguous row-major A[M,K], B[K,N], C[M,N]',
          'correctness':'platform reference, input integrity, repeated invocation and final timed output; no claims'}


def specification(name,shapes,configs):
    return {'schema_version':1,'kind':'research','name':name,'plugins':['../research_phase1/cohort.py'],
        'algorithms':[{'asset':'phase1/cohort@1.0.0','options':{'configs':configs}}],
        'dtypes':['float16'],'seeds':SEEDS,'budgets':[{'max_evaluations':len(configs),'max_wall_time_s':180}],
        'defaults':{'hardware':None,'warmup':10,'repetitions':31,'timing_mode':'cuda_graph','cuda_graph_batch_size':512,
            'preflight_seconds':45,'workspace_memory_limit_bytes':268435456,
            'validation':{'max_configs':4,'max_cases':0,'max_wall_time_s':60,'trust_claims':False}},
        'cases':[{'name':s['name'],'operator':'gemm@1.0.0','split':'test',
            'kernels':[{'asset':'triton/t1_gemm@1.0.0','space':SPACE}],'shapes':[s['dims']]} for s in shapes]}


def generate(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    anchor=specification('crossgpu-anchor-v1',SHAPES,CONFIGS)
    pilot=specification('crossgpu-anchor-pilot-v1',[SHAPES[i] for i in (1,4,5)],[CONFIGS[i] for i in (0,1,2,14)])
    identity=hashlib.sha256(json.dumps({'shapes':SHAPES,'configs':CONFIGS,'protocol':PROTOCOL,'seeds':SEEDS},sort_keys=True).encode()).hexdigest()
    manifest={'schema_version':1,'cohort_id':identity,'runtime_base_commit':RUNTIME_BASE,
        'operator':'gemm@1.0.0','kernel':'triton/t1_gemm@1.0.0','shapes':SHAPES,'configs':CONFIGS,'protocol':PROTOCOL,'seeds':SEEDS,
        'prior_information':'5060 Ti exploratory shapes/configs; frozen before any 4090/H800 latency observations',
        'candidate_count':16,'raw_cartesian_domain_count':192,
        'acquisition':'Existing cohort SDK plugin visits only the listed configs; independent acceptance can sample the broader declared SDK domain and is excluded from the matrix.',
        'fold_protocol':'Keep physical records unchanged. Hardware-holdout Replay derives explicit train/test views with original record IDs; no target labels in training.',
        'metrics':['target-relative regret','Spearman','Top-5/10/20% overlaps','source Top-K target coverage (oracle scoring only)',
            'B0 fixed source Top1','B1/2/4/8 observed search','failures/missing/common subset','source and target costs separately'],
        'search_rng_seeds':list(range(10)),'search_budgets':[0,1,2,4,8],
        'budget_per_gpu':{'pilot_observations':24,'pilot_acceptance_allowance':24,'pilot_wall_cap_s':600,
            'main_observations':192,'main_acceptance_allowance':48,'main_wall_cap_s':1200,
            'estimated_main_wall_s':[480,720],'gpu_memory_admission_budget_bytes':2*1024**3,
            'host_memory_admission_budget_bytes':2*1024**3,'compiler_workers':1,
            'main_timed_complete_plan_calls':192*31*512,
            'notice':'Wall estimates calibrated after local pilot. Runtime/allocator contexts vary; stop rather than exceed authorized server limits.'}}
    for name,data in [('anchor',anchor),('pilot',pilot),('frozen-design',manifest)]:
        (directory/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':generate(Path(__file__).resolve().parent)
