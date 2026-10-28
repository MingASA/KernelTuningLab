"""Public-safe summaries from standard journals; never measures/imputes latency."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random

from analyze_phase1 import current_records
from kernel_tuning_lab.analysis.report import _bar_chart,_line_chart
from kernel_tuning_lab.research import read_json,snapshot_directory


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def native(directory,output):
    rows=current_records(directory);manifest=read_json(directory/'research_manifest.json');summary=read_json(directory/'summary.json')
    result=[];profiles=set()
    for r in rows:
        a=r.metadata.get('correctness',{}).get('execution',{}).get('annotations',{});p=a.get('production')
        if not p:continue
        profiles.add(p['host_library_sha256'])
        result.append({'id':r.id,'kernel':r.operator.implementation,'shape':dict(r.shape.dims),'config':dict(r.config.params),
            'seed':r.seed,'status':r.status.value,'reference_verified':r.correctness_verified,
            'verified_on_gpu':r.metadata['correctness'].get('verified_on_gpu'),
            'latency_ms':r.median_ms if r.correctness_verified else None,'raw_samples_ms':list(r.runtime_samples_ms),
            'actual_call_flags':p['actual_native_call_flags'],'kernel_symbols':p['kernel_symbols'],
            'allocation_ops':p['observed_allocation_ops'],'logical_input_identity':a['logical_input_identity'],
            'packing_identity':a['backend_packing_identity']})
    public={'compute_commit':manifest.get('git_commit'),'status':summary['status'],
        'GPU_models':sorted({r.hardware.name for r in rows}),'binary_profiles':sorted(profiles),
        'new_research_measurements':summary['research_evaluations'],'cache_hits':summary['cache_hits'],
        'independent_acceptance':summary['validation_evaluations'],'records':result,
        'scope':'Runtime native options, original rules and whole Graph GPU plan including output copy. Not CUBIN counts or production-serving speedups.'}
    write(output/'production.json',public);return public


def controlled(directory,output):
    records={r.id:r for r in current_records(directory)};manifest=read_json(directory/'research_manifest.json')
    best_known={};trials=[]
    from kernel_tuning_lab.catalog import get_catalog
    from kernel_tuning_lab.plugins import load_plugin
    from kernel_tuning_lab.asset_space import build_effective_space
    catalog=get_catalog();load_plugin('kernel_tuning_lab.production.marlin',catalog)
    random_samples={}
    for r in records.values():
        if r.correctness_verified:
            k=(r.operator.implementation,r.shape.id);best_known[k]=min(best_known.get(k,float('inf')),r.median_ms)
    for job in manifest['jobs']:
        snapshot=snapshot_directory(directory,job)
        if snapshot is None or not (snapshot/'trials.json').is_file():continue
        for raw in read_json(snapshot/'trials.json'):
            rows=[records[i] for i in raw['record_ids']];key=(raw['kernel_ref'],rows[0].shape.id) if rows else None
            outside=None
            if rows and raw.get('space_cardinality',0)>4096:
                if key not in random_samples:
                    r=rows[0];space=build_effective_space(catalog.resolve('kernel',raw['kernel_ref']),r.hardware,r.operator,r.shape)
                    random_samples[key]={c.id for c in space.sample_configs(random.Random(7),512)}
                outside={'sample_seed':7,'sample_size':512,'scope':'Raw uniform sample from current domains, not a rerun of the old optimizer',
                    'proposed_outside':sum(r.config.id not in random_samples[key] for r in rows),
                    'GPU_verified_outside':sum(r.config.id not in random_samples[key] and r.correctness_verified and r.metadata.get('correctness',{}).get('verified_on_gpu') is True for r in rows)}
            running=None;curve=[]
            for i,r in enumerate(rows,1):
                if r.correctness_verified:running=min(running or float('inf'),r.median_ms)
                curve.append({'observation':i,'best_ms':running,'best_known_loss':running/best_known[key]-1 if running is not None else None})
            trials.append({'case':job['case'],'kernel':raw['kernel_ref'],'algorithm':raw['algorithm_ref'],
                'shape':raw['shape'],'seed':job['seed'],'budget':job['budget'],'status':raw['status'],
                'stop_reason':raw.get('stop_reason'),'observations':len(rows),'physical':raw['evaluations'],
                'cache_hits':raw['cache_hits'],'statuses':dict(Counter(r.status.value for r in rows)),
                'record_ids':raw['record_ids'],'space_cardinality':raw.get('space_cardinality'),
                'wall_s':raw.get('wall_time_s'),'algorithm_costs_ms':raw.get('algorithm_costs_ms'),
                'diagnostics':raw.get('algorithm_diagnostics'),'curve':curve,
                'reachability_beyond_a_512_point_sample':outside})
    result={'compute_commit':manifest.get('git_commit'),'manifest_status':manifest['status'],'trials':trials,
        'unique_observed_records':len(records),'statuses':dict(Counter(r.status.value for r in records.values())),
        'reference':'best-known in the observed union for each implementation/Shape; not an exhaustive oracle',
        'sampling':'one algorithm seed; capability verification, not a statistically powered optimizer ranking',
        'cost':'Shared compatible caches; compilation is a subset of measurement wall; independent cold-start cost was not measured.'}
    write(output/'controlled.json',result)
    groups=defaultdict(list)
    for t in trials:groups[t['case'],t['kernel'],t['budget']['max_evaluations']].append(t)
    for index,((case,kernel,budget),members) in enumerate(groups.items()):
        series={t['algorithm']:[(p['observation'],p['best_known_loss']*100) for p in t['curve'] if p['best_known_loss'] is not None] for t in members}
        _line_chart(output/f'controlled-{index}.svg',f'{case} / {kernel} / B{budget}',
            'Actual revealed feedback; failures/cache consume observations; best-known union reference',series,
            'Observations','Loss to finite observed best-known (%)',render_png=False)
    return result


def supplement(directory,output):
    """Exact current/archived attempt costs, plus real selected execution traces."""
    stages={}
    names=('ai-train-validation','moe-pilot','production-dispatch','common-inputs','ai-test',
           'controlled-confirmation','final-core-regression','production-table-dispatch')
    for name in names:
        root=directory/name
        if not root.exists():continue
        summary=read_json(root/'summary.json');manifest=read_json(root/'research_manifest.json');records=current_records(root)
        attempts=[]
        for path in [root/'summary.json',*(root/'attempts').glob('*/summary.json')]:
            value=read_json(path)
            attempts.append({k:value.get(k) for k in ('run_id','status','observations','research_evaluations',
                'cache_hits','validation_evaluations','job_wall_time_s','validation_wall_time_s')})
        trials=[]
        for job in manifest['jobs']:
            snapshot=snapshot_directory(root,job)
            if snapshot is None or not (snapshot/'trials.json').exists():continue
            for trial in read_json(snapshot/'trials.json'):
                trials.append({k:trial.get(k) for k in ('kernel_ref','algorithm_ref','shape','status','stop_reason',
                    'record_ids','evaluations','cache_hits','best_verified_latency_ms','best_verified_config',
                    'algorithm_costs_ms','algorithm_runtime','algorithm_diagnostics')})
        compact=[];identity_groups=defaultdict(set)
        for r in records:
            annotations=r.metadata.get('correctness',{}).get('execution',{}).get('annotations',{})
            if 'logical_input_identity' in annotations:
                identity_groups[r.shape.id,r.seed].add(json.dumps(annotations['logical_input_identity'],sort_keys=True))
            compact.append({'id':r.id,'kernel':r.operator.implementation,'shape':dict(r.shape.dims),'config':dict(r.config.params),
                'seed':r.seed,'status':r.status.value,'reference_verified':r.correctness_verified,
                'verified_on_gpu':r.metadata.get('correctness',{}).get('verified_on_gpu'),
                'latency_ms':r.median_ms if r.correctness_verified else None,'raw_samples_ms':list(r.runtime_samples_ms),
                'logical_input_identity':annotations.get('logical_input_identity'),
                'native_call_flags':annotations.get('production',{}).get('actual_native_call_flags')})
        if any(len(v)!=1 for v in identity_groups.values()):raise ValueError('paired logical input identities disagree')
        stages[name]={'status':summary['status'],'compute_commit':manifest.get('git_commit'),
            'counts':{k:summary[k] for k in ('observations','research_evaluations','cache_hits','validation_evaluations')},
            'attempts':attempts,'trials':trials,'records':compact,
            'paired_logical_identity_groups':len(identity_groups),'all_logical_identities_match':True}
    write(output/'supplement.json',stages)
    return stages


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--controlled',type=Path);p.add_argument('--production',type=Path)
    p.add_argument('--l3',type=Path);p.add_argument('--supplement',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.controlled:controlled(a.controlled,a.output)
    if a.production:native(a.production,a.output)
    if a.supplement:supplement(a.supplement,a.output)
    if a.l3:
        x=read_json(a.l3);public={k:x[k] for k in ('status','native_rules','selections','chosen_dispatch','unseen_input_policy','counts','reference_scope','independent_input_repetitions')}
        public.update({k:x[k] for k in ('cohort','precision_scope') if k in x})
        public['matrix']={k:x['matrix'][k] for k in ('workload','configs','latencies_ms','provenance')}
        write(a.output/'l3.json',public)
        series={}
        for row in x['selections']:
            key=row['workload']+'/'+row['method'];series.setdefault(key,[]).append((row['k'],row['relative_loss']*100))
        _line_chart(a.output/'l3-loss.svg','Native Marlin finite Config Budget',
            f"Synthetic frequencies; {x['counts']['candidate_config_count']} runtime option combinations, not CUBIN counts",series,
            'Config budget K','Loss to finite explicit-native reference (%)',render_png=False)


if __name__=='__main__':main()
