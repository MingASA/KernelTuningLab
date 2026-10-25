"""Supplemental cost/drift, W4 and online-vs-Replay checks; no CUDA execution."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as st

import analyze_phase1 as core
from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.catalog import get_catalog


def cost_summary(records):
    phases=defaultdict(float);missing=0;build=setup=graph=event=0.
    shapes=defaultdict(list)
    for record in records:
        timing=record.metadata.get('phase_wall_time_ms')
        if timing is None: missing+=1
        else:
            for key,value in timing.items(): phases[key]+=value
        execution=record.metadata.get('correctness',{}).get('execution',{})
        build+=execution.get('build_time_ms',0.) or 0.
        setup+=execution.get('setup_time_ms',0.) or 0.
        graph+=execution.get('cuda_graph_setup_time_ms',0.) or 0.
        batch=(record.metadata['protocol'].get('cuda_graph') or {}).get('calls_per_replay',1)
        event+=sum(record.runtime_samples_ms)*batch
        shapes[record.shape.id].append(record)
    return {'records':len(records),'missing_phase_trace_records':missing,'known_phase_wall_ms':dict(phases),
        'compile_callback_total_ms':sum(r.compile_time_ms for r in records),
        'source_record_wall_s':sum(r.wall_time_s for r in records),'known_build_ms':build,'known_one_time_setup_ms':setup,
        'known_graph_capture_setup_ms':graph,'summed_timed_gpu_events_ms':event,
        'by_shape':[{'shape':dict(rows[0].shape.dims),'records':len(rows),'mean_measurement_wall_s':st.mean(r.wall_time_s for r in rows),
            'max_measurement_wall_s':max(r.wall_time_s for r in rows),'compile_ms':sum(r.compile_time_ms for r in rows)} for rows in shapes.values()],
        'notice':'Compile includes build/setup/capture. GPU event sum is part of measurement callback wall. Components overlap across these views; do not add them. Old absent phase fields remain unknown.'}


def telemetry(path):
    rows=[]
    if not path.is_file(): return {'status':'not_run'}
    for line in path.read_text().splitlines():
        row=json.loads(line)
        fields=row.get('csv','').split(', ')
        if len(fields)!=8: continue
        try: rows.append([float(fields[i].split()[0]) for i in (2,3,4,5,6,7)])
        except ValueError: continue
    return {'samples':len(rows),'ranges':{name:{'min':min(row[i] for row in rows),'max':max(row[i] for row in rows)}
        for i,name in enumerate(('temperature_C','power_W','sm_clock_MHz','memory_clock_MHz','utilization_pct','memory_used_MiB'))} if rows else {},
        'notice':'Includes idle/preflight/compile periods, sampled every 2s; clocks and desktop load were not locked.'}


def drift(before,after):
    old=core.current_records(before);new=core.current_records(after)
    core.require_baseline_compatible(old[0],new[0])
    key=lambda r:(r.shape.id,r.config.id,r.seed)
    a={key(r):r for r in old if r.correctness_verified};b={key(r):r for r in new if r.correctness_verified}
    if set(a)!=set(b): raise ValueError('anchor pairs missing; do not infer complete drift check')
    return {'pairs':[{'shape':dict(a[k].shape.dims),'config':dict(a[k].config.params),'seed':a[k].seed,
        'before_record_id':a[k].id,'after_record_id':b[k].id,'before_us':a[k].median_ms*1000,'after_us':b[k].median_ms*1000,
        'ratio_after_before':b[k].median_ms/a[k].median_ms} for k in sorted(a)],
        'median_relative_change':st.median(b[k].median_ms/a[k].median_ms-1 for k in a),
        'max_absolute_relative_change':max(abs(b[k].median_ms/a[k].median_ms-1) for k in a),
        'notice':'Same source, config, Shape, input seed and timing protocol; new experiment IDs ensure independent physical measurements.'}


def w4_study(directory):
    records=core.current_records(directory);by_kernel=defaultdict(list)
    for record in records: by_kernel[record.operator.implementation].append(record)
    manifest=core.read_json(Path(directory)/'research_manifest.json')
    outputs=[];replays=[]
    for kernel,rows in by_kernel.items():
        job=next(j for j in manifest['jobs'] if j['kernel']==kernel)
        domains=job['experiment']['kernels'][0]['space'];space=core.ConfigSpace(domains,name='W4 finite controlled space')
        configs=tuple(space.iter_configs());shapes=tuple(core.ShapeSpec(s) for s in job['shapes'])
        matrix=core.PerformanceMatrix.from_records(core.WorkloadDistribution(shapes,tuple([1]*len(shapes))),configs,rows)
        effects=[]
        for i,shape in enumerate(shapes):
            for key in ('BLOCK_M','GROUP_M','SPLIT_K'):
                values=domains[key];reference=values[0]
                for high in values[1:]:
                    pairs=defaultdict(dict)
                    for j,config in enumerate(configs):
                        pairs[tuple((k,config.params[k]) for k in domains if k!=key)][config.params[key]]=matrix.latencies_ms[i][j]
                    ratios=[p[high]/p[reference] for p in pairs.values() if p.get(high) and p.get(reference)]
                    effects.append({'shape':dict(shape.dims),'parameter':key,'low':reference,'high':high,
                        'median_high_over_low':st.median(ratios) if ratios else None,'pairs':len(ratios)})
        outputs.append({'kernel':kernel,'matrix':matrix.to_dict(),'research_records':len(rows),'effects':effects,
            'compiled_artifacts':core.artifact_counts(rows),'costs':cost_summary(rows)})
        for measurement_seed in (7,19):
            for shape in shapes:
                source=[r for r in rows if r.shape.id==shape.id and r.seed==measurement_seed]
                for name in ('random','heuristic','surrogate'):
                    for seed in range(10):
                        for budget in (4,8):
                            result=core.run_replay(source,core.make_searcher(name,seed=seed),space,core.SearchBudget(budget),seed=seed)
                            replays.append({'kernel':kernel,'shape':dict(shape.dims),'measurement_seed':measurement_seed,
                                'algorithm':name,'search_seed':seed,'budget':budget,'regret':result.regret,
                                'evaluations':result.evaluations,'failed':result.failed_evaluations,'missing':result.missing_evaluations})
    return {'implementations':outputs,'replay':replays,
        'scope':'canonical bitstream W4A16 full execution plan; fused/pipeline matrices remain separate. No GPTQ/AWQ or production inference speedup claim. GEMM training labels are incompatible with this operator and are not fed to Transfer.'}


def online_check(primary_directory,online_directory):
    primary=core.current_records(primary_directory);live=core.read_json(Path(online_directory)/'summary.json')
    catalog=get_catalog();kernel=catalog.resolve('kernel','triton/t1_gemm')
    rows=[]
    for trial in live['trials']:
        source=[r for r in primary if r.seed==trial['seed'] and r.shape.dims==trial['shape']]
        first=source[0]
        space=build_effective_space(kernel,first.hardware,first.operator,first.shape,core.design.SPACE)
        name=trial['algorithm'].split('@')[0]
        replay=core.run_replay(source,core.make_searcher(name,seed=trial['seed']),space,core.SearchBudget(trial['budget']['max_evaluations']),seed=trial['seed'])
        actual=[r['config_id'] for r in trial['observations_trace']];simulated=[step.config.id for step in replay.trace]
        rows.append({'algorithm':name,'shape':trial['shape'],'budget':trial['budget'],
            'live_config_ids':actual,'replay_config_ids':simulated,'identical_trajectory':actual==simulated,
            'identical_first_four':actual[:4]==simulated[:4],'live_best_us':trial['best_verified_latency_ms']*1000,
            'replay_best_us':replay.best_latency_ms*1000,'live_over_replay_latency':trial['best_verified_latency_ms']/replay.best_latency_ms,
            'new_measurements':trial['evaluations'],'cache_observations':trial['cache_hits']})
    return {'comparisons':rows,'summary':{k:live[k] for k in ('status','observations','research_evaluations','cache_hits','validation_evaluations')},
        'notice':'Random/Heuristic can reproduce proposed config sequences; adaptive Surrogate can branch differently on fresh noisy measurements. GPU timings are fresh observations, Replay timings are prior measured labels.'}


def main(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    result={'protocol':'supplemental analyses are stored separately from the primary full matrix','studies':{}}
    for label in ('correctness','pilot','matrix','library','anchor-before','anchor-after','bf16','simple','w4','online','expert-probe'):
        path=root/label/'summary.json'
        if path.is_file():
            summary=core.read_json(path)
            result['studies'][label]={k:summary[k] for k in ('run_id','git_commit','status','job_statuses','observations','research_evaluations','cache_hits','validation_evaluations','job_wall_time_s','physical_gpu_validated')}
            result['studies'][label]['telemetry']=telemetry(root/(label+'-telemetry.jsonl'))
        else:result['studies'][label]={'status':'not_run'}
    if (root/'matrix/summary.json').is_file(): result['primary_costs']=cost_summary(core.current_records(root/'matrix'))
    if all((root/label/'summary.json').is_file() for label in ('anchor-before','anchor-after')): result['drift']=drift(root/'anchor-before',root/'anchor-after')
    if (root/'w4/summary.json').is_file():
        w4=w4_study(root/'w4');core.write(output/'w4.json',w4)
        result['w4_summary']=[{'kernel':r['kernel'],'complete':all(v is not None for row in r['matrix']['latencies_ms'] for v in row),
            'records':r['research_records'],'compiled_artifacts':r['compiled_artifacts']} for r in w4['implementations']]
    if all((root/label/'summary.json').is_file() for label in ('matrix','online')):result['online_check']=online_check(root/'matrix',root/'online')
    core.write(output/'followups.json',result)
    print(json.dumps({label:study['status'] for label,study in result['studies'].items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();main(args.root,args.output)
