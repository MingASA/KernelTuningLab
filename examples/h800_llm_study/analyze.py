"""Read-only analysis of existing v2 exports; no Runner, predictor or new store.

Input bundles contain unmodified StandardStore MeasurementRecord dictionaries,
the v2 research summary and original resource receipt. Raw bundles stay private.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict,Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from kernel_tuning_lab.models import MeasurementRecord,canonical_json
from kernel_tuning_lab.measurement_context import source_context_id
from examples.h800_llm_study.design import protocol


def audit_rule(expression):
    """Conservative exact check: f(config)+g(scalar context) cannot adapt ranks.

    Vector history/random features deliberately make this proof unknown. This
    is a program-quality observation, never a reason to exclude a legal config.
    """
    from kernel_tuning_lab.llm_search import validate_expression,PARAMETERS,FEATURES
    _,used=validate_expression(expression)
    def terms(node):
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub)):
            return [*terms(node.left),*terms(node.right)]
        return [node]
    vector_features={'NEAREST_MS','NEAREST_DISTANCE','INCUMBENT_DISTANCE','UNCERTAINTY','RANDOM'}
    mixed=[]
    for node in terms(ast.parse(expression,mode='eval').body):
        names={n.id for n in ast.walk(node) if isinstance(n,ast.Name)}
        if names&set(PARAMETERS) and names&set(FEATURES):mixed.append(ast.unparse(node))
    invariant=not mixed and not used&vector_features
    return {'used_parameters':sorted(used&set(PARAMETERS)),'used_features':sorted(used&set(FEATURES)),
        'context_config_mixed_terms':mixed,'proven_context_invariant_ranking_for_fixed_feasible_set':invariant,
        'proof_scope':'additively separable scalar context offset; differing feasible sets can still change available candidates',
        'uses_online_vector_feedback':bool(used&vector_features),
        'action':'diagnostic only, original model expression unchanged'}


def load_bundle(path):
    data=json.loads(Path(path).read_text());guard=data['batch']['resource_guard']
    if (guard['status'] not in ('passed','command_failed') or guard.get('external_processes_observed')!=0
            or not guard.get('command_pid') or guard.get('conflict_checks',0)<1):
        raise ValueError('resource admission/isolation was not completed; quarantine this batch')
    records={rid:MeasurementRecord.from_dict(raw) for rid,raw in data['records'].items()}
    uuid=guard['gpu']['uuid'].lower().removeprefix('gpu-')
    for rid,record in records.items():
        if rid!=record.id:raise ValueError('StandardStore record identity changed')
        if record.provenance!='measured':raise ValueError('GPU study cannot promote mock/imported records to live measurements')
        if record.hardware.metadata.get('device_uuid','').lower().removeprefix('gpu-')!=uuid:
            raise ValueError('physical measurement/guard device mismatch')
    for trial in data['summary']['trials']:
        trace=trial.get('observations_trace',[])
        if len(trace)!=trial['observations'] or ('record_ids' in trial and trial['record_ids']!=[r['record_id'] for r in trace]):
            raise ValueError('summary observation/record ledger changed')
        for row in trace:
            record=records[row['record_id']]
            if row['correctness_verified']!=record.correctness_verified:
                raise ValueError('summary/record correctness disagree')
            if record.shape.dims!=trial['shape'] or record.seed!=trial['seed']:
                raise ValueError('trace is not from the declared input/seed')
            if row.get('config_id')!=record.config.id:
                raise ValueError('trace configuration differs from the actual record')
            expected=record.median_ms if record.correctness_verified else None
            if row['latency_ms']!=expected:
                raise ValueError('summary latency differs from raw samples')
            if record.correctness_verified and record.metadata.get('correctness',{}).get('verified_on_gpu') is not True:
                raise ValueError('successful study observation lacks GPU correctness evidence')
    return data,records


def train_feedback(paths):
    """Only explicitly supplied train bundles may be read for rule iteration."""
    records={};acquisition=[];trials=[]
    for path in paths:
        # Reject the manifest before reading any performance from a non-train
        # bundle. Do not silently filter a mixed train/test file.
        raw=json.loads(Path(path).read_text())
        if any(j['experiment']['split']!='train' for j in raw['manifest']['jobs']):
            raise ValueError('development feedback accepts train-only bundles')
        data,rows=load_bundle(path);ids={r for t in data['summary']['trials'] for r in t.get('record_ids',[])}
        for rid in ids:
            r=rows[rid]
            if r.split!='train' or dict(r.shape.dims) not in protocol()['train_shapes']:
                raise ValueError('frozen split violation in training records')
            records[rid]=r
        trials.extend(data['summary']['trials'])
        acquisition.append({'stage':data['batch']['stage'],'bundle_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'source_commit':data['batch']['source_commit'],
            'observations':data['summary']['observations'],'new_research_attempts':data['summary']['research_evaluations'],
            'cache_observations':data['summary']['cache_hits'],
            'independent_acceptance':data['summary']['validation_evaluations'],
            'trial_wall_s':sum(t.get('wall_time_s') or 0 for t in data['summary']['trials'])})
    groups=defaultdict(dict)
    for r in records.values():
        if r.correctness_verified:groups[(r.operator.implementation,canonical_json(r.shape.dims))][r.config.id]=r
    selected=[];priors={}
    for (kernel,shape),values in sorted(groups.items()):
        ranked=sorted(values.values(),key=lambda r:(r.median_ms,r.config.id));chosen={r.id:r for r in [*ranked[:3],*ranked[-2:]]}
        for r in chosen.values():selected.append({'split':'train','kernel':kernel,'shape':dict(r.shape.dims),
            'config':dict(r.config.params),'latency_ms':r.median_ms,'record_id':r.id,'seed':r.seed,
            'source_context_id':source_context_id(r),'correctness_verified':True})
    # Strictly observed shared source points only; missing elements never get
    # optimistic values. This is a limited source prior, not an oracle matrix.
    for kernel in sorted({k[0] for k in groups}):
        cases=[v for (k,_),v in groups.items() if k==kernel]
        shared=set.intersection(*(set(v) for v in cases));ranked=[]
        for cid in shared:
            losses=[v[cid].median_ms/min(r.median_ms for r in v.values()) for v in cases]
            ranked.append((statistics.mean(losses),max(losses),cid))
        ranked.sort();priors[kernel]={'configs':[dict(cases[0][cid].config.params) for _,_,cid in ranked[:2]],
            'scope':'configs observed on EVERY train shape; ranked against finite per-shape best-known',
            'shared_config_count':len(shared),'train_shapes':len(cases)}
    return {'schema_version':1,'test_labels_read':False,'observations':selected,'source_acquisition':acquisition,
        'all_unique_research_records':len(records),'verified_research_records':sum(r.correctness_verified for r in records.values()),
        'failures':[{'split':'train','kernel':r.operator.implementation,'shape':dict(r.shape.dims),
            'config':dict(r.config.params),'status':r.status.value,'failure_reason':r.failure_reason}
            for r in records.values() if not r.successful],
        'source_prior':priors,'compression':'top3/bottom2 distinct verified configs per train task, plus all failures; no validation/test labels'}


def method(trial):
    options=trial.get('algorithm_options',{})
    if trial['algorithm'].startswith('llm/live_'):
        return 'B/'+(trial.get('algorithm_runtime') or {}).get('model_id','unresolved-model').split('/')[-1]
    if trial['algorithm'].startswith('llm/generated_'):
        return 'C/'+Path(options['artifact']).stem.replace('-final','')
    if trial['algorithm']=='smac_rf@1.0.0' and options.get('warm_start_configs'):
        return 'SMAC/train-prior' if trial.get('status')!='not_run' else 'SMAC/preflight-rejected-original'
    return trial['algorithm'].split('@')[0]


def evaluate(paths):
    """Target-only, matched-context references from actually observed unions."""
    bundles=[load_bundle(p) for p in paths];records={};trials=[];calls={}
    for data,rows in bundles:
        records.update(rows)
        jobs={j['id']:j for j in data['manifest']['jobs']}
        for t in data['summary']['trials']:
            if 'algorithm_options' not in t:
                t['algorithm_options']=jobs[t['job_id']]['experiment']['algorithms'][0].get('options',{})
            if any(rows[rid].split!='test' for rid in t.get('record_ids',[])):
                raise ValueError('target report cannot include development observations')
            # Validation is charged separately and never helps a search curve.
            trials.append(t)
        for file,events in data['calls'].items():
            prefix=str(Path(file).parent)
            raw=data['trial_files'].get(prefix+'/trial.json')
            if raw:calls[raw['trial_id']]=events
    reference={};input_hashes=defaultdict(set);context_shapes=defaultdict(set);physical_blocks=defaultdict(set)
    for t in trials:
        for event in t.get('observations_trace',[]):
            r=records[event['record_id']]
            key=(r.operator.implementation,r.hardware.id,r.shape.id,r.seed,source_context_id(r))
            context_shapes[(r.operator.implementation,r.hardware.id,r.shape.id,r.seed)].add(source_context_id(r))
            physical_blocks[(r.operator.implementation,r.shape.id,r.seed,r.operator.dtype,
                canonical_json(r.operator.parameters))].add(r.hardware.id)
            if r.correctness_verified:reference[key]=min(reference.get(key,math.inf),r.median_ms)
            integrity=r.metadata.get('correctness',{}).get('input_integrity',{})
            if r.correctness_verified and (integrity.get('status')!='passed' or not integrity.get('expected')):
                raise ValueError('successful CUDA comparison lacks full input identity')
            if integrity.get('expected'):input_hashes[key].add(canonical_json(integrity['expected']))
    if any(len(v)>1 for v in context_shapes.values()):
        raise ValueError('target methods mix measurement/source/software contexts')
    if any(len(v)>1 for v in physical_blocks.values()):
        raise ValueError('matched target methods used different physical GPUs; split the blocks before comparison')
    if any(len(v)>1 for v in input_hashes.values()):
        raise ValueError('same target case/seed had different input contents')
    rows=[];model_calls=[]
    for t in trials:
        events=t.get('observations_trace',[]);available=[records[e['record_id']] for e in events]
        key=None
        if available:
            r=available[0];key=(r.operator.implementation,r.hardware.id,r.shape.id,r.seed,source_context_id(r))
        for b in (1,2,4,8,16,32):
            if b>t['budget']['max_evaluations']:continue
            complete=len(events)>=b;prefix=events[:b];verified=[e['latency_ms'] for e in prefix if e['correctness_verified']]
            best=min(verified,default=None);ref=reference.get(key);regret=best/ref-1 if best is not None and ref else None
            rows.append({'method':method(t),'kernel':t['kernel'],'shape':t['shape'],'seed':t['seed'],'budget':b,
                'reached_budget':complete,'revealed_observations':len(prefix),'status':t['status'],
                'best_ms':best,'finite_union_best_ms':ref,'regret':regret,'stop_reason':t['stop_reason'],
                'physical_attempts':sum(not e['cache_hit'] for e in prefix),
                'cache_observations':sum(e['cache_hit'] for e in prefix),
                'failed_measurements':sum(e['status']!='success' for e in prefix),
                'trial_wall_s':t.get('wall_time_s'),'prefix_wall_s':prefix[-1]['elapsed_wall_time_s'] if prefix else None,
                'full_trial_algorithm_cost_ms':t.get('algorithm_costs_ms'),
                'physical_compile_ms':sum(e['physical_compile_time_ms'] for e in prefix),
                'physical_measurement_wall_s':sum(e['physical_measurement_wall_time_s'] for e in prefix),
                'p90_p10_relative_width':statistics.median((e['p90_ms']-e['p10_ms'])/e['latency_ms'] for e in prefix if e['correctness_verified']) if verified else None,
                'context_id':t.get('source_context_id')})
            rows[-1]['hardware_id']=t.get('hardware_id')
        for e in calls.get(t.get('trial_id'),[]):
            model_calls.append({'method':method(t),'kernel':t['kernel'],'shape':t['shape'],'seed':t['seed'],
                **{k:e.get(k) for k in ('observation_index','attempt','static_valid','rejection','error','input_tokens','output_tokens',
                    'generation_wall_s','upload_wall_s','offload_and_release_wall_s','CPU_model_load_wall_s','total_inference_proposal_wall_s',
                    'CUDA_allocated_before_bytes','CUDA_allocated_after_bytes','all_parameters_on_CPU_before_Kernel','finish_reason')}})
    aggregate=[]
    for key in sorted({(r['method'],r['budget']) for r in rows}):
        group=[r for r in rows if (r['method'],r['budget'])==key];ok=[r for r in group if r['reached_budget'] and r['regret'] is not None]
        matched=[r for r in ok if r['seed'] in (7,19)]
        seed7=[r for r in ok if r['seed']==7]
        aggregate.append({'method':key[0],'budget':key[1],'tasks':len(group),'complete_tasks':len(ok),
            'early_or_no_result':len(group)-len(ok),'mean_regret_complete':statistics.mean(r['regret'] for r in ok) if ok else None,
            'matched_7_19_mean_regret_complete':statistics.mean(r['regret'] for r in matched) if matched else None,
            'matched_7_mean_regret_complete':statistics.mean(r['regret'] for r in seed7) if seed7 else None,
            'within5pct_all_tasks':sum(r['regret']<=.05 for r in ok)/len(group),
            'within10pct_all_tasks':sum(r['regret']<=.10 for r in ok)/len(group),
            'median_prefix_wall_s_complete':statistics.median(r['prefix_wall_s'] for r in ok) if ok else None})
    return {'schema_version':1,'reference':'finite verified target union per exact kernel/GPU/shape/input-seed/protocol, not global optimum',
        'aggregation':'equal task weight; primary mean conditional on reaching B, coverage/failure denominator includes all planned trials; no unseen latency imputation',
        'rows':rows,'aggregate':aggregate,'model_calls':model_calls,
        'counts':{'trials':len(trials),'research_observations':sum(t['observations'] for t in trials),
            'new_research_attempts':sum(t['evaluations'] for t in trials),'cache_observations':sum(t['cache_hits'] for t in trials),
            'independent_acceptance':sum(d['summary']['validation_evaluations'] for d,_ in bundles)},
        'trial_statuses':dict(Counter(t['status'] for t in trials))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('feedback','evaluate'));p.add_argument('--evidence',nargs='+',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    value=train_feedback(args.evidence) if args.mode=='feedback' else evaluate(args.evidence)
    Path(args.output).write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
