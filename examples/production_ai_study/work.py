"""Bounded study composition over existing search_case/Runner/Store; no new engine."""
from __future__ import annotations
import argparse,ast,hashlib,json,random,time
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def catalog():
    from kernel_tuning_lab.catalog import get_catalog
    from kernel_tuning_lab.plugins import load_plugin
    c=get_catalog()
    for name in ('gemlite','flute','flex_attention','sglang_decode','vllm_moe_compat'):
        load_plugin('kernel_tuning_lab.production.'+name,c)
    load_plugin('kernel_tuning_lab.optimizers',c)
    return c

def bind(row,split,protocol,*,native=False):
    from kernel_tuning_lab.sdk import AssetKernelAdapter,OperatorSpec,ShapeSpec,SpaceContext
    from kernel_tuning_lab.asset_space import build_effective_space
    from kernel_tuning_lab.asset_context import AssetCase
    c=catalog();op=c.resolve('operator',row['operator']);kernel=c.resolve('kernel',row['native'] if native else row['kernel'])
    adapter=AssetKernelAdapter(kernel,op,force_correctness=True,accept_claims=False,timing_mode='cuda_graph',
        cuda_graph_batch_size=protocol['timing']['batch_size'],workspace_memory_limit_bytes=512*1024**2)
    hardware=adapter.probe_hardware();operator=OperatorSpec(op.id,kind=op.id,version=op.version,implementation=kernel.ref,
        dtype=protocol['dtype'],correctness_baseline=op.ref,parameters=row['parameters'],
        metadata={'operator_asset_fingerprint':op.fingerprint,'kernel_asset_fingerprint':kernel.fingerprint})
    shape=ShapeSpec(row[split]);context=SpaceContext(hardware,operator,shape,adapter.environment(),512*1024**2)
    space=build_effective_space(kernel,hardware,operator,shape,context=context)
    return AssetCase(kernel,op,hardware,operator,shape,space,adapter)

def public_task(protocol,version,feedback=None):
    c=catalog();cases=[]
    for row in protocol['cases']:
        kernel=c.resolve('kernel',row['kernel'])
        cases.append({'id':row['id'],'operator':row['operator'],'implementation':row['kernel'],'parameters':row['parameters'],
            'development_shape':row['train'],'declared_domains':{name:list(spec.values) for name,spec in kernel.space.parameters.items()}})
    interface='''Return a complete SDK Searcher program, not a scoring expression or hardcoded winning table.
Plain class __init__(seed=0); suggest(self,hardware,operator,shape,space,history,budget,batch_size=1) returns list[KernelConfig]. No sdk.Searcher base class exists.
Import ONLY random, math and KernelConfig, SearchAlgorithmDefinition, SpaceBuildBudget from kernel_tuning_lab.sdk.
ASSETS=(SearchAlgorithmDefinition("generated/production_policy","1.0.0",YourClass),) and algorithm_ref generated/production_policy@1.0.0 must match.
KernelConfig accepts ONE dict, never named launch keywords. SpaceView.domains maps parameter to tuple. space.complete(partial,hardware,operator,shape,count=16,rng=self.rng,exclude_ids=seen,budget=SpaceBuildBudget(max_nodes=20000,max_wall_time_s=1)) returns result.configs. Nonempty configs are success; result.status is sampled/exhausted/complete, NEVER success.
space.neighbors(config,max_changed=2,max_neighbors=64) returns neighbors; space.validate(config,hardware,operator,shape).valid is bool. Check feasibility and exclude seen config.id. Conditional/default/native-template constraints remain authoritative.
history is a sequence of records with config, status, correctness_verified and median_ms; history.attempted_config_ids includes prior failures. Only use verified median_ms as measured latency. budget.max_evaluations is remaining observations; budget.exhausted(0,0) means stop.
Use shape/resource-aware schedules plus revealed feedback where useful. Different backends have different parameter names; numeric template attrs are meaningful, arbitrary IDs are not continuous. You may sample a bounded number of completions and maintain state; do not enumerate the full Cartesian product or call the measurement engine. Do not read files, network, credentials, labels or global experiment state. Under100 lines; no unbounded while. Generic unknown domains should still return valid novel points, including SDK synthetic singleton spaces.
Return JSON with exactly program, algorithm_ref, rationale string fields. No Markdown. Greedy local inference, no hidden fallback or human code edits.'''
    task={'schema_version':1,'feature_version':'production-study-full-searcher-v1','public_search_interface':interface,
        'space':{'development_cases':cases,'authority':'runtime SpaceView, complete declared space; no sampled candidate pool'},
        'remaining_budget':{'observations_per_test_case':protocol['test_observations'],'program_version':version},
        'notice':'Test shapes/performance withheld. Generate reusable full Searcher. No test data or old studies supplied.'}
    if feedback:
        task['development_feedback']={'test_labels_read':False,'observations':feedback['observations']}
        task['previous_program']=feedback['previous_program']
    return task

def static_review(source):
    tree=ast.parse(source);allowed={'random','math','kernel_tuning_lab.sdk'};issues=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import) and any(a.name not in allowed for a in node.names):issues.append('unexpected import')
        if isinstance(node,ast.ImportFrom) and node.module not in allowed:issues.append('unexpected import')
        if isinstance(node,ast.Attribute) and node.attr.startswith('__'):issues.append('dunder access')
        if isinstance(node,ast.While):issues.append('unbounded while')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('open','eval','exec','compile','__import__','globals','locals','vars','input'):issues.append('forbidden capability')
    if not any(isinstance(n,ast.FunctionDef) and n.name=='suggest' for n in ast.walk(tree)):issues.append('missing suggest')
    if not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ASSETS' for t in n.targets) for n in tree.body):issues.append('missing ASSETS')
    if issues:raise ValueError(';'.join(sorted(set(issues))))

def generate(plan,protocol,output,version):
    from kernel_tuning_lab.local_inference import LocalModelClient
    from kernel_tuning_lab.program_artifacts import generate as model_generate
    previous=json.loads(Path(plan['feedback']).read_text()) if version else None
    task=public_task(protocol,version,previous);write(output/'visible-task.json',task)
    large=plan['models']['C'];state={'status':'generating','version':version,'actual_model_calls':0,'program_executed':False,'test_labels_read':False}
    write(output/'state.json',state)
    try:
        with LocalModelClient(model_manifest=large['manifest'],manifest_sha256=large['manifest_sha256'],resource_mode='single_card_resident',
                seed=23,log_path=output/'model-calls.jsonl') as client:
            for attempt in range(2):
                state['actual_model_calls']+=1;write(output/'state.json',state)
                row=model_generate(client,task,max_output_tokens=protocol['model_policy']['C_generation_max_output_tokens'])
                text=row['text'];(output/f'response-{attempt}.txt').write_text(text)
                try:
                    text=text.rsplit('</think>',1)[-1].strip()
                    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
                    value=json.loads(text)
                    if set(value)!={'program','algorithm_ref','rationale'} or '@' not in value['algorithm_ref']:raise ValueError('exact JSON/versioned algorithm_ref required')
                    compile(value['program'],'model-program.py','exec');static_review(value['program'])
                    (output/'model-program.py').write_text(value['program']);write(output/'selected.json',value)
                    state.update(status='awaiting_trusted_review',program_sha256=digest(output/'model-program.py'),algorithm_ref=value['algorithm_ref']);break
                except (ValueError,TypeError,SyntaxError,KeyError) as error:
                    write(output/f'static-error-{attempt}.json',{'error':str(error),'new_performance_feedback':False})
                    task.update(previous_program=text,notice='Static output correction only: '+str(error)+'. Return complete JSON, no new performance labels.')
            if state['status']!='awaiting_trusted_review':raise RuntimeError('two bounded real generation calls failed; no substitute program')
    except Exception as error:state.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:state['generation_phase_closed']=True;write(output/'state.json',state)

def reviewed_definition(item):
    from kernel_tuning_lab.catalog import AssetCatalog
    from kernel_tuning_lab.plugins import load_plugin
    file=Path(item['program']);review=json.loads(Path(item['review']).read_text())
    if review.get('approved') is not True or review.get('program_sha256')!=digest(file):raise ValueError('matching trusted source review required')
    c=AssetCatalog();load_plugin(str(file.resolve()),c)
    return c.resolve('algorithm',item['algorithm_ref'])

def run_trials(plan,protocol,output,split):
    from kernel_tuning_lab.asset_search import search_case
    from kernel_tuning_lab.runner import BenchmarkRunner
    from kernel_tuning_lab.sdk import KernelConfig,SearchBudget,ExperimentStore,SearchAlgorithmDefinition
    from kernel_tuning_lab.live_llm import LiveDecision,_Session,_SESSIONS
    from kernel_tuning_lab.inference_resources import COORDINATOR
    rows=[row for row in protocol['cases'] if row['machine']==plan['machine'] and row['id'] in plan['case_ids']]
    state={'status':'running','split':split,'trials':[],'test_labels_read':split=='test','source_commit':plan['source_commit']};write(output/'state.json',state)
    model=plan['models']['B'];session=None;began=time.monotonic()
    try:
        if plan['machine']=='H800':
            session=_Session(json.loads(Path(model['manifest']).read_text()),'single_card_resident',model['manifest_sha256'])
            _SESSIONS[(model['manifest_sha256'],'single_card_resident')]=session
            write(output/'resident-model.json',{'model':model['model_id'],'parameter_count':session.parameter_count,'load_wall_s':session.load_s,'same_residency_for_all_methods':True})
        class SharedDecision(LiveDecision):
            def close(self):pass # outer phase owns this shared resident session
        for row in rows:
            case=bind(row,split,protocol);actual=case.hardware.metadata['device_uuid'].lower().removeprefix('gpu-')
            if actual!=plan['gpu_uuid'].lower().removeprefix('gpu-'):raise ValueError('physical study UUID mismatch')
            directory=output/row['id'];directory.mkdir();seed=protocol['train_seed'] if split=='train' else protocol['test_seed']
            with ExperimentStore(directory/'measurements.sqlite') as store:
                accept=row.get('accept')
                if accept is None:
                    from kernel_tuning_lab.production.flute import candidates
                    selected=next(dict(c.parameters) for c in candidates() if str(c.native_id)==str(row['accept_native_template_id']))
                    accept={key:value for key,value in selected.items() if key in case.space.domains}
                runner=BenchmarkRunner(case.adapter,store=store,warmup=5,repetitions=15,seed=seed,experiment_id='production-study-'+split+'-acceptance')
                verified=runner.measure(case.hardware,case.operator,case.shape,KernelConfig(accept),case.space,split=split)
                if not verified.correctness_verified:raise RuntimeError('strict independent acceptance failed: '+str(verified.failure_reason))
                methods=['C0'] if split=='train' else [m for m in protocol['methods'] if m!='B_full_static_grammar' or plan['machine']=='H800']
                if split=='test':
                    # Frozen rotation spreads order effects without latency-based ordering.
                    random.Random(protocol['test_seed']+protocol['cases'].index(row)).shuffle(methods)
                for method in methods:
                    options={};c=catalog()
                    if method in ('C0','C1'):definition=reviewed_definition(plan['programs'][method])
                    elif method=='B_full_static_grammar':
                        definition=SearchAlgorithmDefinition('study/B_full_static_grammar','1.0.0',SharedDecision,source_files=(__file__,))
                        options={'model_manifest':model['manifest'],'manifest_sha256':model['manifest_sha256'],'max_new_tokens':192,'max_retries':1,'structured_output':True,'resource_mode':'single_card_resident'}
                    else:
                        definition=c.resolve('algorithm',method+'@1.0.0')
                        if method=='smac_rf':options={'initial_design_size':4}
                        if method=='kernel_tuner_de':options={'population_size':8,'execution_mode':'callback'}
                    count=protocol['train_observations'] if split=='train' else protocol['test_observations']
                    result=search_case(case,definition,options,SearchBudget(count,protocol['trial_wall_s']),store=store,seed=seed,
                        warmup=5,repetitions=15,split=split,experiment_id='production-study-'+split+'-'+row['id'],output_dir=directory/method,resume=True)
                    state['trials'].append({'case':row['id'],'method':method,**result});write(output/'state.json',state)
                if split=='test':
                    native=bind(row,split,protocol,native=True)
                    native_runner=BenchmarkRunner(native.adapter,store=store,warmup=5,repetitions=15,seed=seed,experiment_id='production-study-native-expert')
                    record=native_runner.measure(native.hardware,native.operator,native.shape,KernelConfig({}),native.space,split=split)
                    write(directory/'native-expert.json',{'record':record.to_dict(),'new_physical_measurement':not native_runner.last_cache_hit,'equal_observation_budget_claim':False})
                store.export_jsonl(directory/'records.jsonl')
        state.update(status='completed',no_tolerance_relaxed=True)
    except Exception as error:state.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        if session is not None:_SESSIONS.pop((model['manifest_sha256'],'single_card_resident'),None);session.close()
        state['registered_producers_after_close']=len(COORDINATOR.participants);state['phase_wall_s']=time.monotonic()-began;write(output/'state.json',state)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--phase',choices=('C0','C1','train','test'),required=True);p.add_argument('--output',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());protocol=json.loads(Path(plan['protocol']).read_text());out=Path(a.output)
    if digest(plan['protocol'])!=plan['protocol_sha256']:raise ValueError('frozen protocol changed')
    import subprocess
    if subprocess.check_output(['git','-C',plan['code'],'rev-parse','HEAD'],text=True).strip()!=plan['source_commit']:raise ValueError('source checkout changed')
    if (out/'state.json').exists():raise ValueError('existing attempt; recover original receipt, never repeat a finished phase')
    out.mkdir(parents=True,exist_ok=True)
    if a.phase in ('C0','C1'):generate(plan,protocol,out,0 if a.phase=='C0' else 1)
    else:run_trials(plan,protocol,out,a.phase)

if __name__=='__main__':main()
