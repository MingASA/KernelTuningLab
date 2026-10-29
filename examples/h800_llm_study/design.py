"""Frozen experiment and prompt construction; never opens target latency data."""
import hashlib
import json
from pathlib import Path

from kernel_tuning_lab.models import stable_id,canonical_json
from kernel_tuning_lab.llm_search import PARAMETERS,FEATURES,FUNCTION_ARITY

HERE=Path(__file__).resolve().parent


def protocol():return json.loads((HERE/'protocol.json').read_text())


def task_cases(split):
    p=protocol();result=[]
    for entry in p['kernels']:
        for shape in p[split+'_shapes']:
            key=stable_id([entry,shape])
            result.append({'name':split+'-'+key,'operator':entry['operator'],'kernels':[entry['kernel']],
                'shapes':[shape],'split':split})
    return result


def research_spec(algorithms,*,splits=('test',),budget=32,seeds=None,name='H800-real-local-LLM-v1'):
    p=protocol();m=p['measurement']
    return {'schema_version':1,'kind':'research','name':name,
        'plugins':['kernel_tuning_lab.production.marlin','kernel_tuning_lab.optimizers',
                   'kernel_tuning_lab.llm_search','kernel_tuning_lab.live_llm'],
        'algorithms':algorithms,'dtypes':[p['dtype']],'seeds':p['tuning_seeds'] if seeds is None else seeds,
        'budgets':[{'max_evaluations':budget,'max_wall_time_s':900}],
        'max_jobs':1024,'defaults':{'hardware':None,'warmup':m['warmup'],'repetitions':m['repetitions'],
            'timing_mode':m['mode'],'cuda_graph_batch_size':m['batch'],
            'preflight_seconds':180,'workspace_memory_limit_bytes':m['workspace_memory_limit_bytes'],
            'validation':{'max_configs':1,'max_cases':0,'max_wall_time_s':90,'trust_claims':False}},
        'cases':[c for split in splits for c in task_cases(split)]}


def rule_messages(space_descriptions,*,previous=None,train_feedback=None,contract_error=None):
    # Only source semantics, domains and hardware. Held-out Shapes are not part
    # of rule-development prompts; direct decisions see their own input only.
    p=protocol()
    task={'objective':'Generate a numeric configuration-ranking heuristic for three GPU GEMM implementations. Lower scores rank first. A generic executor evaluates this exact expression over the full bounded static legal set, excludes revealed configs, and measures selected candidates with the existing Runner.',
        'hardware':{'model':'H800','architecture':'sm_90','measurement':'hot whole GPU execution plan via CUDA Graph; input preparation/compile outside latency but charged tuning wall'},
        'semantics':{'triton/t1_gemm@1.0.0':'FP16 contiguous A@B, FP32 TensorCore accumulation, grouped program ordering; no Triton autotuner',
            'controlled/w4a16_fused@1.0.0':'FP16 A@((u4-8)*group128scales); packed integers unpacked inside each K tile; Split-K adds FP32 workspace/reduction',
            'controlled/w4a16_pipeline@1.0.0':'Same logical W4 GEMM; each call materializes dequantized FP16 weights, then GEMM, then reduction when Split-K>1'},
        'domains':space_descriptions,'available_parameters':list(PARAMETERS),'available_features':list(FEATURES),
        'missing_parameter_conventions':{'SPLIT_K':1,'DEQUANT_BLOCK_SIZE':0},
        'history_features':'NEAREST_MS is latency of closest VERIFIED observed config in normalized parameter distance; NEAREST_DISTANCE/UNCERTAINTY and INCUMBENT_DISTANCE depend only on revealed points; before observations nearest_ms0, distances1. RANDOM is seeded per target step. BEST_MS is best verified revealed latency or0. No target oracle is accessible.',
        'functions_and_exact_arity':FUNCTION_ARITY,'expression_contract':'Only numeric arithmetic +,-,*,/,//,%,constant powers[-4,4], individual comparisons, and the listed functions. No imports/statements/attributes/subscripts/lists. Use where(condition,a,b) rather than Python if/and/or. <=4096 characters and256 AST nodes.',
        'output_contract':{'score_expression':'one expression string','rationale':'short explanation; no unmeasured latency or speedup claims'},
        'history_cost':'If development observations are supplied, they are paid historical measurements, not free target information.',
        'scope':{'kernel_refs':[k['kernel'] for k in p['kernels']],'architectures':['sm_90']}}
    if previous is not None:task['previous_model_program']=previous
    if train_feedback is not None:
        if train_feedback.get('test_labels_read') is not False or any(r['split']!='train' for r in train_feedback['observations']):
            raise ValueError('rule iteration accepts train feedback only')
        task['observed_train_feedback']=train_feedback
    if contract_error is not None:task['static_contract_error']=contract_error
    return [{'role':'system','content':'You generate a GPU tuning heuristic. The final answer must be a JSON object with score_expression and rationale. Use only supplied observations; no tools or files are available. Your exact expression will be frozen and executed without a coding agent rewriting it.'},
        {'role':'user','content':canonical_json(task)}]


def extract_rule(text):
    if '</think>' in text:text=text.rsplit('</think>',1)[1]
    decoder=json.JSONDecoder();objects=[]
    for index,char in enumerate(text):
        if char!='{':continue
        try:value,_=decoder.raw_decode(text[index:])
        except ValueError:continue
        if isinstance(value,dict) and 'score_expression' in value:objects.append(value)
    if not objects:raise ValueError('no final generated rule JSON found')
    value=objects[-1]
    if set(value)-{'score_expression','rationale'}:raise ValueError('unexpected generated-rule fields')
    from kernel_tuning_lab.llm_search import validate_expression
    validate_expression(value['score_expression'])
    return value


def rule_artifact(generation,model_run,*,version,development_feedback_sha256=None):
    if generation['identity']['evidence_level']!='real_local_model_inference' or model_run['status']!='passed':
        raise ValueError('real completed model generation required')
    if generation['finish_reason']!='eos':raise ValueError('truncated model program is not a valid frozen version')
    value=extract_rule(generation['text']);p=protocol()
    return {'schema_version':1,'kind':'llm_generated_rule','version':version,
        'score_expression':value['score_expression'],'model_rationale':value.get('rationale'),
        'scope':{'kernel_refs':[x['kernel'] for x in p['kernels']],'architectures':['sm_90']},
        'generation_evidence':{k:generation['identity'][k] for k in ('evidence_level','model_id','manifest_sha256','worker_source_sha256','query_bundle_sha256')},
        'source_response_sha256':hashlib.sha256(canonical_json(generation).encode()).hexdigest(),
        'source_prompt_sha256':generation['prompt_sha256'],'agent_rewritten':False,
        'development_feedback_sha256':development_feedback_sha256,'test_labels_used_for_development':False,
        'costs':{'model_run_id':model_run['identity']['query_bundle_sha256'],'model_load_wall_s':model_run['load_wall_s'],
            'generation_wall_s':generation['generation_wall_s'],'input_tokens':generation['input_tokens'],
            'output_tokens':generation['output_tokens'],'scope':'Offline historical generation cost, count model load once per run, not once per trial'}}
