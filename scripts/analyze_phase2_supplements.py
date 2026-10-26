"""Analyze recorded phase2 controls and the finite physical-binary POC.

No GPU work, label prediction, new database or general deployment solver.
The small exact mixed selector assumes one specialized binary per Shape and
one generic binary per config; these assumptions must first pass real evidence.
"""
import argparse
from collections import defaultdict
import itertools
import math
from pathlib import Path
import statistics as st

import analyze_phase2 as phase
core=phase.core


def mixed_selection(generic,specialized,weights,k,mode='mixed'):
    """Exact finite POC: generic subsets plus independent per-Shape upgrades."""
    n=len(specialized)
    if (mode not in ('mixed','generic','specialized') or k<0 or n==0 or len(weights)!=n
            or any(len(g)!=n for g in generic) or any(w<0 for w in weights) or sum(weights)<=0):
        raise ValueError('inconsistent physical POC dimensions or budget')
    total=sum(weights);best=None
    subsets=[()] if mode=='specialized' else itertools.chain.from_iterable(
        itertools.combinations(range(len(generic)),size) for size in range(min(k,len(generic))+1))
    for subset in subsets:
        remaining=k-len(subset)
        baseline=[min((generic[j][i] for j in subset),default=math.inf) for i in range(n)]
        if not subset:
            if mode=='generic' or remaining<n:continue
            upgraded=list(range(n))
        else:
            upgraded=[] if mode=='generic' else sorted(range(n),key=lambda i:weights[i]*(baseline[i]-specialized[i]),reverse=True)[:remaining]
            upgraded=[i for i in upgraded if specialized[i]<baseline[i]]
        values=[specialized[i] if i in upgraded else baseline[i] for i in range(n)]
        if any(not math.isfinite(v) for v in values):continue
        score=sum(w*v for w,v in zip(weights,values))/total
        if best is None or score<best['weighted_latency_ms']:
            best={'weighted_latency_ms':score,'generic_indices':list(subset),'specialized_shape_indices':upgraded,
                  'used_compiler_identities':len(subset)+len(upgraded),'latencies_ms':values}
    return best


def resources(record):
    return record.metadata.get('correctness',{}).get('execution',{}).get('compiled_resources',{})


def pooled(records):return st.median(v for r in records for v in r.runtime_samples_ms)


def generic_study(rows,compiled_index=None):
    implementations=sorted({r.operator.implementation for r in rows})
    generic=next(i for i in implementations if 'generic' in i)
    specialized=next(i for i in implementations if 'generic' not in i)
    shapes=sorted({r.shape.id:r.shape for r in rows}.values(),key=lambda s:tuple(s.dims.values()))
    configs=tuple(core.KernelConfig(c) for c in core.read_json(
        core.ROOT/'examples/research_phase2/frozen-design.json')['generic_configs'])
    cells=defaultdict(list)
    for r in rows:
        if not r.correctness_verified or not r.metadata['correctness'].get('verified_on_gpu'):raise ValueError('POC requires verified GPU records')
        cells[r.operator.implementation,r.shape.id,r.config.id].append(r)
    for i,s,c in itertools.product(implementations,shapes,configs):
        if len(cells[i,s.id,c.id])!=2:raise ValueError('physical POC is incomplete or duplicated')
    latencies={i:[[pooled(cells[i,s.id,c.id]) for s in shapes] for c in configs] for i in implementations}
    by_impl={i:[r for r in rows if r.operator.implementation==i] for i in implementations}
    for s,c in itertools.product(shapes,configs):
        core.require_baseline_compatible(cells[generic,s.id,c.id][0],cells[specialized,s.id,c.id][0])
    actual_checks=[];objects=defaultdict(set);identity_coverage=defaultdict(set)
    identities_by_impl={i:set() for i in implementations}
    cubins_by_identity=defaultdict(set)
    for r in rows:
        rr=resources(r)
        if len(rr)!=1:raise ValueError('physical POC assumes one kernel per invocation')
        resource=next(iter(rr.values()));identity=resource.get('compiler_cache_identity')
        binary=resource.get('artifacts',{}).get('cubin')
        if not identity or not binary:raise ValueError('physical compiler/binary evidence unavailable')
        identities_by_impl[r.operator.implementation].add(identity)
        identity_coverage[identity].add(r.shape.id);cubins_by_identity[identity].add(binary['sha256'])
        if r.operator.implementation==generic:
            names=resource.get('runtime_scalar_names',[])
            check=(not resource.get('shape_arguments_constexpr',True) and len(names)==9
                and all(resource['compiler_signature'].get(name)=='i32' for name in names)
                and resource.get('actual_launch_compiler_identity')==identity
                and resource.get('actual_launch_object_instance')==resource.get('compiled_object_instance'))
            actual_checks.append(check)
            objects[resource['process_id'],r.config.id].add(resource.get('compiled_object_instance'))
    assumptions=(all(actual_checks) and all(len(v)==1 for v in objects.values())
        and len(identities_by_impl[generic])==len(configs)
        and len(identities_by_impl[specialized])==len(configs)*len(shapes)
        and all(len(identity_coverage[i])==len(shapes) for i in identities_by_impl[generic])
        and all(len(identity_coverage[i])==1 for i in identities_by_impl[specialized])
        and not identities_by_impl[generic]&identities_by_impl[specialized]
        and all(len(v)==1 for v in cubins_by_identity.values()))
    result={'status':'passed' if assumptions else 'physical_selection_not_computed',
        'shape_count':len(shapes),'config_count':len(configs),'runtime_and_launch_checks':all(actual_checks),
        'one_object_per_process_config':all(len(v)==1 for v in objects.values()),
        'implementations':{i:core.artifact_counts(rr) for i,rr in by_impl.items()},
        'identity_shape_coverage':{i:len(v) for i,v in identity_coverage.items()},
        'shapes':[dict(s.dims) for s in shapes],'configs':[dict(c.params) for c in configs],
        'latencies_ms':latencies,'comparisons':[], 'physical_budget':[],
        'notice':'Finite 6-Shape / 4-config POC. Compiler identity count, cubin identity and container bytes are separate; no byte-budget deployment solver or driver-resident size claim.'}
    for ci,c in enumerate(configs):
        for si,s in enumerate(shapes):
            spec_rows=cells[specialized,s.id,c.id];gen_rows=cells[generic,s.id,c.id]
            result['comparisons'].append({'shape':dict(s.dims),'config':dict(c.params),
                'generic_over_specialized':latencies[generic][ci][si]/latencies[specialized][ci][si],
                'specialized_compile_ms':[r.compile_time_ms for r in spec_rows],
                'generic_compile_ms':[r.compile_time_ms for r in gen_rows],
                'specialized_run_medians_ms':[r.median_ms for r in spec_rows],
                'generic_run_medians_ms':[r.median_ms for r in gen_rows]})
    if assumptions:
        specs=[min(row[si] for row in latencies[specialized]) for si in range(len(shapes))]
        for label,weights in [('uniform',[1]*len(shapes)),('synthetic_low_m',[4 if s.get('M')<=4 else 1 for s in shapes])]:
            reference=sum(w*min(specs[si],*(g[si] for g in latencies[generic])) for si,w in enumerate(weights))/sum(weights)
            for k,mode in itertools.product((1,2,4,6,8),('generic','specialized','mixed')):
                choice=mixed_selection(latencies[generic],specs,weights,k,mode)
                counts={}
                if choice:
                    selected=[]
                    for ci in choice['generic_indices']:
                        selected.extend(cells[generic,shapes[0].id,configs[ci].id])
                    for si in choice['specialized_shape_indices']:
                        ci=min(range(len(configs)),key=lambda j:latencies[specialized][j][si])
                        selected.extend(cells[specialized,shapes[si].id,configs[ci].id])
                    counts=core.artifact_counts(selected)
                    if counts['compiler_specializations']!=choice['used_compiler_identities']:
                        raise ValueError('selected physical identities do not match exact selector cost')
                    binaries={resource['artifacts']['cubin']['sha256'] for r in selected for resource in resources(r).values()}
                    index=(compiled_index or {}).get('artifacts',{})
                    counts['deduplicated_elf_text_bytes']=sum(index[b]['text_bytes'] for b in binaries) if all(b in index and index[b]['text_bytes'] is not None for b in binaries) else None
                result['physical_budget'].append({'weights':label,'k':k,'mode':mode,
                    'status':'passed' if choice else 'infeasible',**(choice or {}),'selected_artifacts':counts,
                    'relative_loss':choice['weighted_latency_ms']/reference-1 if choice else None})
    return result


def analyze(root):
    root=Path(root);out=root/'analysis';reports={};all_rows={}
    labels=('pilot','train','validation','test','generic-correctness','generic','controls','repeat','bf16','online','w4','group-beyond-l2','torch')
    for label in labels:
        path=root/label/'research_manifest.json'
        if not path.is_file():continue
        manifest=core.read_json(path);rows=core.current_records(root/label);all_rows[label]=rows
        attempts={m['run_id']:m for m in [manifest]+[core.read_json(p) for p in (root/label/'attempts').glob('*/research_manifest.json')]}
        reports[label]={'jobs':len(manifest['jobs']),'job_statuses':dict(core.Counter(j['status'] for j in manifest['jobs'])),
            'observations':len(rows),'success':sum(r.correctness_verified for r in rows),
            'job_wall_s':sum(j.get('wall_time_s',0) for j in manifest['jobs']),
            'new_evaluations':sum(j.get('result',{}).get('evaluations',0) for j in manifest['jobs']),
            'cache_hits':sum(j.get('result',{}).get('cache_hits',0) for j in manifest['jobs']),
            'validation_evaluations':sum(j.get('validation_evaluations',0) for j in manifest['jobs']),
            'source_measurement_wall_s':sum(r.wall_time_s for r in rows),
            'compile_ms':sum(r.compile_time_ms or 0 for r in rows),
            'all_attempts':[{'run_id':m['run_id'],'job_statuses':dict(core.Counter(j['status'] for j in m['jobs'])),
                'new_evaluations':sum(j.get('result',{}).get('evaluations',0) for j in m['jobs']),
                'cache_hits':sum(j.get('result',{}).get('cache_hits',0) for j in m['jobs']),
                'validation_evaluations':sum(j.get('validation_evaluations',0) for j in m['jobs']),
                'job_wall_s':sum(j.get('wall_time_s',0) for j in m['jobs'])} for m in attempts.values()]}
    core.write(out/'supplement-costs.json',reports)
    index=core.read_json(root/'compiled/index.json') if (root/'compiled/index.json').is_file() else None
    if 'generic' in all_rows:core.write(out/'generic.json',generic_study(all_rows['generic'],index))
    for label in ('controls','group-beyond-l2','w4','bf16','repeat'):
        if label not in all_rows:continue
        rows=all_rows[label];groups=defaultdict(list)
        for r in rows:groups[r.operator.implementation,r.shape.id].append(r)
        tables=[]
        for (impl,sid),rr in groups.items():
            by=defaultdict(list)
            for r in rr:
                if r.correctness_verified:by[r.config.id].append(r)
            values=[{'config':dict(v[0].config.params),'config_id':cid,'latency_us':pooled(v)*1000,
                'run_medians_us':[r.median_ms*1000 for r in v],
                'compiled_resources':core.artifact_counts(v)} for cid,v in by.items()]
            values.sort(key=lambda v:v['latency_us'])
            tables.append({'implementation':impl,'shape':dict(rr[0].shape.dims),'configs':values})
        core.write(out/(label+'-tables.json'),tables)
    if 'w4' in all_rows:
        replay=[];groups=defaultdict(list)
        for r in all_rows['w4']:groups[r.operator.implementation,r.shape.id,r.seed].append(r)
        source=core.read_json(core.ROOT/'examples/research_phase2/w4.json')
        for (impl,shape_id,input_seed),rr in groups.items():
            domains=next(k['space'] for k in source['cases'][0]['kernels'] if k['asset']==impl)
            space=core.ConfigSpace(domains,name='phase2 W4 finite per-implementation cohort')
            if len(rr)!=12 or not all(r.correctness_verified for r in rr):raise ValueError('W4 Replay requires the complete successful 12-config cell')
            for method,rng,budget in itertools.product(('random','grid','heuristic','surrogate'),range(5),(4,8)):
                trial=core.run_replay(rr,core.make_searcher(method,seed=rng),space,core.SearchBudget(budget),seed=rng).to_dict()
                replay.append({**trial,'implementation':impl,'input_seed':input_seed,'new_gpu_measurements':0})
        core.write(out/'w4-replay.json',{'results':replay,'notice':'New measured W4 labels; offline cold-start search; implementations never pooled. No historical training data used.'})
    print(core.canonical_json({key:{k:v for k,v in value.items() if k in ('observations','success','cache_hits','job_wall_s')} for key,value in reports.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    analyze(parser.parse_args().root)
