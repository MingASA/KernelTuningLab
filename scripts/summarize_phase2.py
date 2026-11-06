"""Publish compact phase2 measurements and SVG figures, after source audit."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import shutil
import statistics as st

import analyze_phase2 as phase
import analyze_phase2_supplements as supplements
core=phase.core


def correlation_figure(path,labels,values):
    left,top,cell=185,165,32;parts=[]
    for i,name in enumerate(labels):
        parts.append(core._text(left-8,top+i*cell+21,name,size=10,anchor='end'))
        x=left+i*cell+15
        parts.append(core._text(x,top-12,name,size=9,extra=f'transform="rotate(-55 {x} {top-12})"'))
        for j,v in enumerate(values[i]):
            value=v or 0;color=f'rgb({int(240-160*max(value,0))},{int(240-100*abs(value))},{int(240-160*max(-value,0))})'
            parts.append(f'<rect x="{left+j*cell}" y="{top+i*cell}" width="30" height="30" fill="{color}"/>')
            parts.append(core._text(left+j*cell+15,top+i*cell+20,f'{value:.1f}',size=9,anchor='middle'))
    core._svg(path,'C48 cross-Shape rank correlation','Spearman rho [-1,1]; complete 24 x 48 finite table; no learned clustering claim',
        left+len(labels)*cell+35,top+len(labels)*cell+30,parts,render_png=False)


def summarize(root,docs):
    root,docs=Path(root),Path(docs);analysis=root/'analysis';data=docs/'data';figures=docs/'figures'
    data.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    load=lambda name:core.read_json(analysis/(name+'.json'))
    audit=core.read_json(root/'audit.json')
    if audit['status']!='passed':raise ValueError('publish only audited source evidence')
    structure,replay,l3,generic=map(load,('structure','replay','l3','generic'))
    rows=phase.physical_records(root);matrix=load('matrix');configs=[core.KernelConfig(c) for c in matrix['configs']]
    grouped=defaultdict(list)
    for r in replay['prediction']:grouped[r['split'],r['method']].append(r)
    prediction=[{'split':split,'method':method,'physical_label_count':len(rr),'target_measurements_required':0,
        'mean_regret':st.mean(r['regret'] for r in rr),'max_regret':max(r['regret'] for r in rr),
        'within_5pct':st.mean(r['regret']<=.05 for r in rr)} for (split,method),rr in grouped.items()]
    quality=[];by_cell=defaultdict(list)
    for r in (*rows,*core.current_records(root/'repeat')):by_cell[r.shape.id,r.config.id].append(r)
    for (sid,cid),rr in by_cell.items():
        if len(rr)!=3:continue
        medians=[r.median_ms for r in rr]
        quality.append({'shape_id':sid,'config_id':cid,'seeds':[r.seed for r in rr],
            'run_median_cv':st.pstdev(medians)/st.mean(medians),'max_over_min':max(medians)/min(medians)})
    sensitivity=[]
    for label in ('controls','group-beyond-l2'):
        for table in load(label+'-tables'):
            groups=defaultdict(dict)
            for item in table['configs']:
                c=item['config'];groups[c['BLOCK_M'],c['BLOCK_N']][c['GROUP_M'],c['num_stages']]=item['latency_us']
            for (bm,bn),values in groups.items():
                baseline=values[1,3]
                sensitivity.append({'study':label,'shape':table['shape'],'BLOCK_M':bm,'BLOCK_N':bn,
                    'group8_over_group1':values[8,3]/baseline,
                    'stage2_over_stage3':values[1,2]/baseline if (1,2) in values else None})
    w4=load('w4-tables');w4_replay=load('w4-replay');wg=defaultdict(list)
    for r in w4_replay['results']:wg[r['implementation'],r['algorithm'],r['budget']].append(r)
    w4_aggregates=[{'implementation':impl,'algorithm':method,'budget':b,'trials':len(rr),
        'mean_regret':st.mean(r['regret'] for r in rr),'within_5pct':st.mean(r['regret']<=.05 for r in rr),
        'failed_or_missing':sum(r['failed_evaluations']+r['missing_evaluations'] for r in rr)} for (impl,method,b),rr in wg.items()]
    baselines=[];torch_rows=core.current_records(root/'torch');spec_rows=core.current_records(root/'generic')
    for shape in {r.shape.id:r.shape for r in torch_rows}.values():
        library=[r for r in torch_rows if r.shape.id==shape.id]
        spec=[r for r in spec_rows if r.shape.id==shape.id and 'generic' not in r.operator.implementation]
        per_config=defaultdict(list)
        for r in spec:per_config[r.config.id].append(r)
        best=min(per_config.values(),key=supplements.pooled)
        guard=core.reference_stability(library)
        try:
            core.require_baseline_compatible(best[0],library[0]);comparability={'status':'passed'}
        except ValueError as error:
            # Never fill absent backend versions or silently loosen the shared
            # comparator. Preserve side-by-side points and the exact limitation.
            comparability={'status':'not_compared','reason':str(error)}
        baselines.append({'shape':dict(shape.dims),'specialized_best_of_four_us':supplements.pooled(best)*1000,
            'torch_mm_us':supplements.pooled(library)*1000,'library_stability':guard,'comparability':comparability,
            'torch_over_t1':supplements.pooled(library)/supplements.pooled(best) if guard['status']=='point_comparison_only' and comparability['status']=='passed' else None,
            'notice':'Preallocated Torch mm; actual internal library dispatch unknown, no production/expert-kernel superiority claim.'})
    manifest=load('dataset-manifest');costs=load('supplement-costs');phase_walls=defaultdict(float)
    for r in rows:
        for key,value in r.metadata.get('phase_wall_time_ms',{}).items():phase_walls[key]+=value
    compact={'schema_version':1,'study':'RTX 5060 Ti readiness / C48 phase2; finite observed references only',
        'dataset':manifest,'audit':audit,'primary':load('summary'),'coverage':load('coverage'),
        'structure':structure,'zero_target_prediction':prediction,
        'physical_repeats':{'two_run_cv_median':st.median(structure['repeat_cv']),
            'two_run_cv_p90':core.quantile(structure['repeat_cv'],.9),'three_run_cells':len(quality),
            'three_run_cv_median':st.median(r['run_median_cv'] for r in quality),
            'three_run_cv_p90':core.quantile([r['run_median_cv'] for r in quality],.9),
            'notice':'Input seed, order and time change together. Descriptive variation, not independent confidence bounds or a causal drift decomposition.'},
        'historical_overlap':load('historical-overlap'),'control_effects':sensitivity,'costs':costs,
        'primary_phase_costs':{'nonoverlapping_runner_wall_ms':dict(phase_walls),
            'graph_setup_ms':sum(r.metadata.get('correctness',{}).get('execution',{}).get('cuda_graph_setup_time_ms',0) for r in rows),
            'one_time_setup_ms':sum(r.metadata.get('correctness',{}).get('execution',{}).get('setup_time_ms',0) for r in rows),
            'scope':'Graph/setup are nested components, not additional costs to sum with Runner phase walls or job wall.'},
        'l2':{'replay_trials':len(replay['results']),'new_gpu_measurements':0,'aggregates':replay['aggregates'],
            'generation_source':core.read_json(analysis/'replay-source-version.json') if (analysis/'replay-source-version.json').is_file() else None,
            'training_manifests':{k:{a:b for a,b in v.items() if a!='record_ids'} for k,v in replay['training_manifests'].items()},
            'notice':replay['notice']},'online':load('online-check'),'l3':l3,'generic':generic,
        'w4':{'tables':w4,'replay_trials':len(w4_replay['results']),'aggregates':w4_aggregates},
        'matched_torch_baselines':baselines,
        'sources_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (core.ROOT/'scripts').glob('*phase2*.py')},
        'raw_root':'artifacts/research-phase2; sealed archive manifest published separately'}
    core.write(data/'research-phase2-summary.json',compact);core.write(data/'research-phase2-matrix.json',matrix)
    for name in ('relative-loss','l2-test','l2-validation','l3-uniform','l3-synthetic_low_m'):
        shutil.copyfile(analysis/'figures'/(name+'.svg'),figures/('research-phase2-'+name+'.svg'))
    names=[s['name'] for s in phase.design.SHAPES]
    correlation_figure(figures/'research-phase2-rank-correlation.svg',names,structure['spearman'])
    prior=core.read_json(root/'pretest/prior.json')['ranking'][0]['config_id'];j=next(i for i,c in enumerate(configs) if c.id==prior)
    core._bar_chart(figures/'research-phase2-fixed-prior.svg','Frozen train-only prior in C48','Historical train10 / new validation3 / new test11; pooled physical labels; not C64 global regret',
        names,[(r[j]/min(r)-1)*100 for r in matrix['latencies_ms']],xlabel='Relative finite-C48 latency loss (%)',render_png=False)
    matching=[(i,s) for i,s in enumerate(phase.design.SHAPES) if s['shape']['N']==s['shape']['K']==2048 and s['shape']['M']<=129]
    matching.sort(key=lambda pair:pair[1]['shape']['M'])
    import math
    series={}
    for bm in (16,32,64):
        c=core.KernelConfig(dict(BLOCK_M=bm,BLOCK_N=64,BLOCK_K=64,num_warps=4,num_stages=3,GROUP_M=1));ci=configs.index(c)
        series['BM='+str(bm)]=[(math.log2(s['shape']['M']),matrix['latencies_ms'][i][ci]*1000) for i,s in matching]
    series['finite C48 best']=[(math.log2(s['shape']['M']),min(matrix['latencies_ms'][i])*1000) for i,s in matching]
    core._line_chart(figures/'research-phase2-low-m.svg','M tile boundaries at fixed N=K=2048','Other params fixed: BN64/BK64/w4/stage3/G1; descriptive measured curves',series,'log2(M)','Hot Graph512 latency (us)',render_png=False)
    group=[r for r in sensitivity if r['study']=='group-beyond-l2']
    core._bar_chart(figures/'research-phase2-group.svg','GROUP control beyond 32 MiB L2','M256/N8192/K4096, FP16 B=64 MiB; two runs, same tile/warps/stages',
        [f"BM{r['BLOCK_M']}/BN{r['BLOCK_N']}" for r in group],[r['group8_over_group1'] for r in group],xlabel='GROUP8 latency / GROUP1 latency',render_png=False)
    for weights in ('uniform','synthetic_low_m'):
        core._line_chart(figures/('research-phase2-physical-'+weights+'.svg'),'Finite physical compiler-identity budget',
            '6 Shapes / 4 configs; guarded observed identity mappings; not a general deployment solver',
            {mode:[(r['k'],r['relative_loss']*100) for r in generic['physical_budget'] if r['weights']==weights and r['mode']==mode and r['status']=='passed'] for mode in ('generic','specialized','mixed')},
            'Maximum compiled identities','Weighted latency loss (%)',render_png=False)
    core.write(analysis/'repeat-quality.json',quality)
    print(core.canonical_json({'status':'published','dataset_id':manifest['dataset_id'],'replay_trials':len(replay['results'])}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--docs',default='docs')
    args=parser.parse_args();summarize(args.root,args.docs)
