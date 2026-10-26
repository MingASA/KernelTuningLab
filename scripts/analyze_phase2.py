"""C48 study analysis using original v2 records, Replay and config-budget L3.

No CUDA execution or hidden-label model fitting. Freeze the train-only prior
before validation/test collection. Raw databases use the existing storage API.
"""
import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import statistics as st

import analyze_phase1 as core
from kernel_tuning_lab.datasets import require_training_records,training_record_manifest
from kernel_tuning_lab.measurement_context import source_context_id

module=importlib.util.spec_from_file_location('phase2_design',core.ROOT/'examples/research_phase2/design.py')
design=importlib.util.module_from_spec(module);module.loader.exec_module(design)
CONFIGS=tuple(core.KernelConfig(c) for c in design.configs())
SHAPES=tuple(core.ShapeSpec(s['shape']) for s in design.SHAPES)
OLD_CONSTANT=core.KernelConfig(dict(BLOCK_M=32,BLOCK_N=128,BLOCK_K=32,num_warps=4,num_stages=3,GROUP_M=1))


def matrix_for(rows,shapes=SHAPES,configs=CONFIGS):
    return core.PerformanceMatrix.from_records(core.WorkloadDistribution(tuple(shapes),tuple([1]*len(shapes)),name='phase2 finite observed workload'),configs,rows)


def physical_records(root,phases=('train','validation','test')):
    return tuple(r for phase in phases for r in core.current_records(Path(root)/phase))


def check_coverage(rows,entries,seeds=design.SEEDS,configs=CONFIGS):
    expected_shapes={core.ShapeSpec(entry['shape']).id:entry for entry in entries}
    expected_configs={c.id for c in configs}
    cells=defaultdict(list)
    for r in rows:
        if (r.shape.id not in expected_shapes or r.config.id not in expected_configs or r.seed not in seeds
                or r.operator.implementation!='triton/t1_gemm@1.0.0' or r.operator.dtype!='float16'
                or r.split!=expected_shapes[r.shape.id]['split']):
            raise ValueError('record outside the frozen primary Shape/config/dtype/seed/split cohort')
        cells[r.shape.id,r.config.id,r.seed].append(r)
    missing=[]
    for entry in entries:
        shape=core.ShapeSpec(entry['shape'])
        for config,seed in itertools.product(configs,seeds):
            found=cells[shape.id,config.id,seed]
            if len(found)!=1 or not found[0].correctness_verified:
                missing.append({'shape':entry['name'],'config_id':config.id,'seed':seed,'status':[r.status.value for r in found]})
    for r in rows:
        if r.successful:
            if not (r.provenance=='measured' and r.metadata['correctness'].get('verified_on_gpu') is True and len(r.runtime_samples_ms)==31
                and r.warmup==10 and r.metadata['protocol']['cuda_graph']['calls_per_replay']==512):
                raise ValueError('primary label lacks strict physical GPU / Graph512 evidence')
    return {'records':len(rows),'expected_records':len(entries)*len(configs)*len(seeds),'missing_successes':missing,
        'record_statuses':dict(Counter(r.status.value for r in rows)),
        'scope':'Only the frozen C48 cohort; no inference about the complete SDK space or historical C64 union'}


def freeze_prior(root):
    root=Path(root)
    for phase in ('validation','test'):
        path=root/phase/'research_manifest.json'
        if path.exists() and any(j['status']!='not_run' for j in core.read_json(path)['jobs']):
            raise ValueError('held-out acquisition already started; refuse retrospective prior freeze')
    rows=physical_records(root,('train',));entries=[s for s in design.SHAPES if s['split']=='train']
    coverage=check_coverage(rows,entries)
    if coverage['missing_successes'] or len(rows)!=960:raise ValueError('complete 10 x 48 x 2 training prefix required')
    table=matrix_for(rows,tuple(core.ShapeSpec(s['shape']) for s in entries))
    scores=[st.mean(row[j]/min(row)-1 for row in table.latencies_ms) for j in range(48)]
    from datetime import datetime,timezone
    import subprocess
    result={'kind':'train-only fixed configuration prior','created_utc':datetime.now(timezone.utc).isoformat(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=core.ROOT,text=True).strip(),
        'freeze_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'heldout_not_started':True,'training_shapes_role':'all 10 historically explored phase1 geometries; newly measured C48 labels',
        'training_data':training_record_manifest(rows),'score':'mean of train-Shape relative losses, pooled input runs; no test labels',
        'ranking':[{'config':dict(CONFIGS[j].params),'config_id':CONFIGS[j].id,'train_mean_relative_loss':scores[j]}
            for j in sorted(range(48),key=lambda j:(scores[j],CONFIGS[j].id))]}
    target=root/'pretest/prior.json'
    if target.exists() and core.read_json(target)['training_data']['record_ids_sha256']!=result['training_data']['record_ids_sha256']:
        raise ValueError('refuse to overwrite a prior frozen from different source records')
    if not target.exists():core.write(target,result)
    print(core.canonical_json({k:v for k,v in result.items() if k not in ('training_data','ranking')}))


def structures(table,rows):
    by_cell=defaultdict(list)
    for r in rows:by_cell[r.shape.id,r.config.id].append(r.median_ms)
    output={'best_by_shape':[],'sensitivity':[],'config_coverage':[],'repeat_cv':[],
        'spearman':[[core.correlation(a,b) for b in table.latencies_ms] for a in table.latencies_ms]}
    for i,entry in enumerate(design.SHAPES):
        row=table.latencies_ms[i];best=min(range(48),key=lambda j:row[j])
        output['best_by_shape'].append({**entry,'best_us':row[best]*1000,'best_config':dict(CONFIGS[best].params),
            'best_run_medians_us':[v*1000 for v in by_cell[SHAPES[i].id,CONFIGS[best].id]]})
        for key,values in design.SPACE.items():
            if len(values)<2:continue
            pairs=defaultdict(dict)
            for j,c in enumerate(CONFIGS):pairs[tuple((k,v) for k,v in c.params.items() if k!=key)][c.params[key]]=row[j]
            for high in values[1:]:
                ratios=[p[high]/p[values[0]] for p in pairs.values()]
                output['sensitivity'].append({'shape':entry['name'],'parameter':key,'low':values[0],'high':high,
                    'paired_geomean_ratio':math.exp(st.mean(math.log(v) for v in ratios)),
                    'p10_ratio':core.quantile(ratios,.1),'p90_ratio':core.quantile(ratios,.9)})
        output['repeat_cv'].extend(st.pstdev(v)/st.mean(v) for c in CONFIGS for v in [by_cell[SHAPES[i].id,c.id]])
    for j,c in enumerate(CONFIGS):
        for role,indices in [('historical',range(10)),('new_heldout',range(10,24)),('all_posthoc',range(24))]:
            losses=[table.latencies_ms[i][j]/min(table.latencies_ms[i])-1 for i in indices]
            output['config_coverage'].append({'config_id':c.id,'config':dict(c.params),'role':role,
                'mean_loss':st.mean(losses),'max_loss':max(losses),'within_5pct':sum(v<=.05 for v in losses),
                'within_10pct':sum(v<=.1 for v in losses),'shape_count':len(losses)})
    output['top_overlap']={}
    for fraction in (.05,.1,.2):
        n=math.ceil(48*fraction);sets=[set(sorted(range(48),key=lambda j:row[j])[:n]) for row in table.latencies_ms]
        output['top_overlap'][str(fraction)]={'top_count':n,'all_shape_intersection':sorted(set.intersection(*sets)),
            'jaccard':[[len(a&b)/len(a|b) for b in sets] for a in sets]}
    return output


def l2_study(rows,prior,history):
    results=[];training_manifests={};space=core.ConfigSpace(design.SPACE,name='frozen phase2 C48')
    target_shapes={SHAPES[i].id for i,s in enumerate(design.SHAPES) if s['role']=='new_heldout'}
    for measurement_seed in design.SEEDS:
        current=[r for r in rows if r.seed==measurement_seed]
        training=[r for r in current if r.split=='train']
        historical=[r for r in history if r.seed==measurement_seed and r.split=='train']
        for label,source in [('current_train',training),('phase1_train',historical)]:
            require_training_records(source,forbidden_shape_ids=target_shapes)
            training_manifests[f'{label}:{measurement_seed}']=training_record_manifest(source)
        for i,entry in enumerate(design.SHAPES):
            if entry['role']!='new_heldout':continue
            target=[r for r in current if r.shape.id==SHAPES[i].id]
            for method in ('random','grid','heuristic','surrogate','transfer_current_train','transfer_phase1'):
                source=training if method=='transfer_current_train' else historical if method=='transfer_phase1' else ()
                name='transfer' if method.startswith('transfer_') else method
                for seed,budget in itertools.product(range(5),(4,8,16,32)):
                    result=core.run_replay(target,core.make_searcher(name,seed=seed,training_records=source),space,core.SearchBudget(budget),seed=seed)
                    row=result.to_dict();ids={r.config.id:r.id for r in target}
                    for step in row['trace']:step['source_record_id']=ids.get(core.KernelConfig(step['config']).id)
                    row.update(algorithm=method,measurement_seed=measurement_seed,shape_name=entry['name'],
                        new_gpu_measurements=0,training_source=f'{"current_train" if method=="transfer_current_train" else "phase1_train"}:{measurement_seed}' if source else None,
                        target_10pct_evaluations=next((s.evaluation for s in result.trace if s.best_latency_ms<=result.oracle_best_latency_ms*1.1),None))
                    results.append(row)
    groups=defaultdict(list)
    for row in results:groups[row['split'],row['algorithm'],row['budget']].append(row)
    aggregates=[]
    for (split,name,budget),group in groups.items():
        values=[r['regret'] for r in group if r['regret'] is not None]
        aggregates.append({'split':split,'algorithm':name,'budget':budget,'budget_fraction':budget/48,'trials':len(group),
            'mean_regret':st.mean(values),'median_regret':st.median(values),'p10_regret':core.quantile(values,.1),'p90_regret':core.quantile(values,.9),
            'within_5pct':st.mean(r['regret'] is not None and r['regret']<=.05 for r in group),
            'within_10pct':st.mean(r['regret'] is not None and r['regret']<=.1 for r in group),
            'failures':sum(r['failed_evaluations']+r['missing_evaluations'] for r in group),
            'mean_source_cost_s':st.mean(r['known_measurement_cost_s'] for r in group),
            'mean_cpu_replay_wall_s':st.mean(r['cpu_wall_time_s'] for r in group)})
    prediction=[]
    for i,entry in enumerate(design.SHAPES):
        for seed in design.SEEDS:
            source=[r for r in rows if r.shape.id==SHAPES[i].id and r.seed==seed];best=min(r.median_ms for r in source)
            for label,cid in [('phase1_constant',OLD_CONSTANT.id),('frozen_train_constant',prior['ranking'][0]['config_id'])]:
                r=next(r for r in source if r.config.id==cid)
                prediction.append({'method':label,'shape':entry['name'],'role':entry['role'],'split':entry['split'],'measurement_seed':seed,
                    'config_id':cid,'regret':r.median_ms/best-1,'record_id':r.id,'target_measurements_required':0})
    return {'results':results,'aggregates':aggregates,'prediction':prediction,'training_manifests':training_manifests,
        'notice':'2 physical input runs x 5 search RNG seeds. Historical source acquisition is separate, not free cold-start data. Offline new GPU executions=0.'}


def config_l3(table,rows):
    outputs=[];keep=core.pareto_indices(table.latencies_ms)
    for label,weights in [('uniform',[1]*24),('synthetic_low_m',[4 if s['shape']['M']<=16 else 1 for s in design.SHAPES])]:
        full=replace(table,workload=core.WorkloadDistribution(SHAPES,tuple(weights),name=label))
        reduced=core.PerformanceMatrix(full.workload,tuple(CONFIGS[j] for j in keep),tuple(tuple(r[j] for j in keep) for r in full.latencies_ms),'measured',full.metadata)
        for k,method in itertools.product((1,2,4,8),('greedy','local_swap','exact')):
            solver={'greedy':core.greedy,'local_swap':core.greedy_local_swap,'exact':core.oracle}[method]
            try:result=solver(reduced if method=='exact' else full,k)
            except core.OracleLimitError as error:
                outputs.append({'weights':label,'k':k,'method':method,'status':'not_computed','reason':str(error)});continue
            source=reduced if method=='exact' else full;selected={c.id for c in result.configs}
            assigned={s.id:source.configs[j].id for s,j in zip(SHAPES,result.assignment_indices)}
            losses=[min(row[j] for j,c in enumerate(CONFIGS) if c.id in selected)/min(row)-1 for row in table.latencies_ms]
            outputs.append({**result.to_dict(),'method':method,'weights':label,'status':'passed','mean_shape_loss':st.mean(losses),
                'max_shape_loss':max(losses),'shape_losses':losses,'assignment_config_ids':assigned,
                'retained_all_pairs':core.artifact_counts(rows,selected),'dispatch_only':core.artifact_counts(rows,assignments=assigned)})
    subset=list(range(0,48,3))
    small=core.PerformanceMatrix(table.workload,tuple(CONFIGS[j] for j in subset),
        tuple(tuple(r[j] for j in subset) for r in table.latencies_ms),'measured',table.metadata)
    return {'budget_kind':'config_count','shape_count':24,'candidate_count':48,'pareto_candidates':len(keep),'results':outputs,
        'small_exact_scope':'predetermined every-third config, 16 candidates; its reference is separate from C48',
        'small_exact':[core.oracle(small,k).to_dict() for k in (1,2,4,8)]}


def analyze(root,skip_replay=False):
    root=Path(root);out=root/'analysis';figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    rows=physical_records(root);coverage=check_coverage(rows,design.SHAPES);core.write(out/'coverage.json',coverage)
    dataset_id=hashlib.sha256(core.canonical_json(sorted(r.id for r in rows)).encode()).hexdigest()
    folder=out/'datasets'/dataset_id
    with core.ExperimentStore(folder/'measurements.sqlite') as store:
        store.add_many(rows);store.export_jsonl(folder/'measurements.jsonl')
    core.write(out/'dataset-manifest.json',{'dataset_id':dataset_id,'records':len(rows),'sqlite':str((folder/'measurements.sqlite').relative_to(out)),
        'jsonl':str((folder/'measurements.jsonl').relative_to(out)),
        'collection_commits':{phase:core.read_json(root/phase/'research_manifest.json')['git_commit'] for phase in ('train','validation','test')}})
    if coverage['missing_successes']:raise ValueError('incomplete C48; exported all observations and missing statuses, no complete analysis claim')
    table=matrix_for(rows);core.write(out/'matrix.json',table.to_dict())
    structure=structures(table,rows);core.write(out/'structure.json',structure)
    core._heatmap(figures/'relative-loss.svg','C48 measured relative latency','24 Shapes; historical training versus new held-out geometry explicitly separated',
        [s['name'] for s in design.SHAPES],[str(i) for i in range(48)],[[v/min(row) for v in row] for row in table.latencies_ms])
    prior=core.read_json(root/'pretest/prior.json')
    train=[r for r in rows if r.split=='train']
    if training_record_manifest(train)['record_ids_sha256']!=prior['training_data']['record_ids_sha256']:raise ValueError('frozen prior source mismatch')
    l3=config_l3(table,rows);core.write(out/'l3.json',l3)
    for weights in ('uniform','synthetic_low_m'):
        core._line_chart(figures/('l3-'+weights+'.svg'),'C48 Config Budget',weights+'; finite 24-Shape observed reference',
            {method:[(r['k'],100*r['relative_loss']) for r in l3['results'] if r['weights']==weights and r['method']==method and r['status']=='passed']
             for method in ('greedy','local_swap','exact')},'Maximum config dictionaries K','Weighted latency loss (%)',render_png=False)
    if not skip_replay:
        history=core.current_records(core.ROOT/'artifacts/research-phase1/matrix')
        replay=l2_study(rows,prior,history);core.write(out/'replay.json',replay)
        for split in ('validation','test'):
            names=sorted({r['algorithm'] for r in replay['aggregates']})
            core._line_chart(figures/('l2-'+split+'.svg'),'C48 offline Replay',split+'; source data costs separate; no new GPU execution',
                {name:[(r['budget'],100*r['mean_regret']) for r in replay['aggregates'] if r['split']==split and r['algorithm']==name] for name in names},
                'Revealed target observations / 48','Mean finite-set regret (%)',render_png=False)
    summary={'status':'passed','records':len(rows),'cells':24*48,'physical_repeats':2,'dataset_id':dataset_id,
        'repeat_cv_median':st.median(structure['repeat_cv']),'repeat_cv_p90':core.quantile(structure['repeat_cv'],.9),
        'scope':'Frozen C48, historical train10 / new validation3 / new test11; not full SDK or C64 union'}
    core.write(out/'summary.json',summary);print(core.canonical_json(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--freeze-prior',action='store_true');parser.add_argument('--skip-replay',action='store_true')
    args=parser.parse_args()
    if args.freeze_prior:freeze_prior(args.root)
    else:analyze(args.root,args.skip_replay)
