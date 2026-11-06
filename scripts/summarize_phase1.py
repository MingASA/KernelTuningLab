"""Publish compact, reproducible study evidence; raw journals stay in artifacts.

Consumes existing analysis JSONs and the independent audit. No GPU work, model
fitting or alternative storage layer. The exported matrix is the standard L3
PerformanceMatrix serialization, with its measured provenance and full context.
"""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import shutil
import statistics as st

import analyze_phase1 as core


def summarize(root,docs):
    root,docs=Path(root),Path(docs);analysis=root/'analysis';data=docs/'data';figures=docs/'figures'
    data.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    load=lambda name:core.read_json(analysis/(name+'.json'))
    audit=core.read_json(root/'audit.json')
    if audit['status']!='passed': raise ValueError('publish only independently audited evidence')
    structure,replay,l3,joint=map(load,('structure','replay','l3','joint'))
    followups,w4=load('followups'),load('w4')
    cv=[v for values in structure['repeat_cv'].values() for v in values if v is not None]
    main_effects=[]
    for parameter in core.design.SPACE:
        pairs=[r for r in structure['sensitivity'] if r['parameter']==parameter]
        main_effects.append({'parameter':parameter,'low':pairs[0]['low'],'high':pairs[0]['high'],
            'median_shape_paired_geomean_ratio':st.median(r['paired_geomean_high_over_low'] for r in pairs),
            'mean_log_variance_fraction':st.mean(next(t['log_variance_fraction'] or 0 for t in effects['terms'] if t['parameters']==[parameter])
                for effects in structure['factorial_effects'].values())})
    hits=[]
    for method in ('random','grid','heuristic','surrogate','transfer'):
        rows=[r for r in replay['results'] if r['split']=='test' and r['algorithm']==method and r['budget']==32]
        for key in ('target_evaluations','target_10pct_evaluations'):
            known=[r[key] for r in rows if r[key] is not None]
            hits.append({'algorithm':method,'threshold':'5%' if key=='target_evaluations' else '10%',
                'trials':len(rows),'reached_by_32':len(known),'mean_observations_among_reached':st.mean(known) if known else None,
                'p10_observations_among_reached':core.quantile(known,.1),'p90_observations_among_reached':core.quantile(known,.9)})
    joint_groups=defaultdict(list)
    for row in joint['results']: joint_groups[row['method'],row['budget']].append(row)
    joint_summary=[{'method':name,'budget':budget,'trials':len(rows),
        'mean_shape_regret':st.mean(r['mean_shape_regret'] for r in rows),
        'p10_shape_regret':core.quantile([r['mean_shape_regret'] for r in rows],.1),
        'p90_shape_regret':core.quantile([r['mean_shape_regret'] for r in rows],.9),
        'mean_revealed_cells':st.mean(r['revealed_cells'] for r in rows),
        'mean_known_source_wall_s':st.mean(r['known_source_wall_s'] for r in rows)} for (name,budget),rows in joint_groups.items()]
    w4_summary=[];w4_replay=[]
    for implementation in w4['implementations']:
        matrix=implementation['matrix'];configs=[core.KernelConfig(c) for c in matrix['configs']]
        w4_summary.append({'kernel':implementation['kernel'],'records':implementation['research_records'],
            'shapes':matrix['workload']['shapes'],'candidate_count':len(configs),'complete':matrix['complete'],
            'best_us':[min(row)*1000 for row in matrix['latencies_ms']],
            'best_configs':[dict(configs[min(range(len(row)),key=lambda j:row[j])].params) for row in matrix['latencies_ms']],
            'effects':implementation['effects'],'compiled_artifacts':implementation['compiled_artifacts']})
    groups=defaultdict(list)
    for row in w4['replay']: groups[row['kernel'],row['algorithm'],row['budget']].append(row)
    for (kernel,name,budget),rows in groups.items():
        w4_replay.append({'kernel':kernel,'algorithm':name,'budget':budget,'trials':len(rows),
            'mean_regret':st.mean(r['regret'] for r in rows),'p90_regret':core.quantile([r['regret'] for r in rows],.9),
            'within_5pct_fraction':st.mean(r['regret']<=.05 for r in rows),'failed_or_missing':sum(r['failed']+r['missing'] for r in rows)})
    archive=core.read_json(root/'compiled-artifacts/index.json')
    main_cubins=archive['study_cubin_sha256']['matrix']
    primary_code={'distinct_cubins':len(main_cubins),'cubin_container_bytes':sum(archive['artifacts'][key]['bytes'] for key in main_cubins),
        'elf_text_section_bytes':sum(archive['artifacts'][key]['text_bytes'] or 0 for key in main_cubins),
        'matching_ptx_artifacts':sum('ptx_sha256' in archive['artifacts'][key] for key in main_cubins),
        'ptx_targets':sorted({archive['artifacts'][key]['ptx_target'] for key in main_cubins if archive['artifacts'][key].get('ptx_target')}),
        'tensor_core_instruction_forms':sorted({instruction for key in main_cubins for instruction in archive['artifacts'][key].get('tensor_core_instructions',[])}),
        'scope':archive['scope']}
    result={'schema_version':1,'study':'T1 / RTX 5060 Ti phase 1; finite observed references only',
        'dataset':load('dataset-manifest'),'audit':audit,'primary':load('summary'),
        'hardware_environment':{k:v for k,v in load('coverage').items() if k in ('hardware','kernel','dtype','kernel_fingerprint','protocol','environment')},
        'studies':followups['studies'],'shape_design':core.read_json(core.ROOT/'examples/research_phase1/shape-design.json'),
        'repeat_cv':{'median':st.median(cv),'p90':core.quantile(cv,.9),'physical_replicates':3},
        'primary_costs':followups['primary_costs'],'drift':{k:v for k,v in followups['drift'].items() if k!='pairs'},
        'reference_baselines':load('baselines'),'repeated_library_reference':followups['repeated_library_reference'],
        'matched_simple':load('simple-matched'),'expert_probe':load('expert-probe'),
        'structure':{'best_by_shape':structure['best_by_shape'],'parameter_effects':main_effects,
            'top_overlap':structure['top_overlap'],'spearman':structure['spearman'],
            'descriptive_average_linkage':structure['descriptive_average_linkage'],
            'config_coverage':sorted(structure['config_coverage'],key=lambda r:r['mean_relative_loss'])},
        'l2':{'replay_trials':len(replay['results']),'new_gpu_measurements':0,'aggregates':replay['aggregates'],
            'hitting_time':hits,'prior_cost_notice':replay['prior_cost_notice']},
        'prediction':load('prediction'),'fresh_online':followups['online_check'],
        'l3':l3,'primary_code_artifacts':primary_code,
        'joint':{'trials':len(joint['results']),'aggregates':joint_summary,'notice':joint['notice'],
            'training_prior_notice':joint['training_prior_notice']},
        'w4':{'implementations':w4_summary,'replay_trials':len(w4['replay']),'replay_aggregates':w4_replay,'scope':w4['scope']},
        'cross_hardware':core.read_json(root/'cross-hardware-preparation.json'),
        'raw_archive':'artifacts/research-phase1-evidence-20261025.tar.gz; see adjacent .sha256 and .audit.json',
        'analysis_sources_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (core.ROOT/'scripts').glob('*phase1*.py')}}
    core.write(data/'research-phase1-summary.json',result)
    core.write(data/'research-phase1-matrix.json',load('matrix'))
    for filename in ('relative-loss','rank-correlation','top10-overlap','sensitivity','factorial-main-effects',
        'repeat-variation','l2-test','l3-uniform','l3-synthetic_decode_skew','joint'):
        shutil.copyfile(analysis/'figures'/(filename+'.svg'),figures/('research-phase1-'+filename+'.svg'))
    costs=followups['primary_costs']['known_phase_wall_ms']
    core._bar_chart(figures/'research-phase1-cost.svg','Primary observation phase costs',
        '1920 real observations; non-overlapping phase walls. Acceptance/startup/storage costs are separate.',
        list(costs),[value/1000 for value in costs.values()],xlabel='Summed wall seconds',render_png=False)
    core._line_chart(figures/'research-phase1-w4.svg','W4A16 full execution-plan latency',
        'Separate fused/pipeline implementations; 3 Shapes x 12 configs x 2 input seeds. Canonical bitstream.',
        {r['kernel']:[(i,value) for i,value in enumerate(r['best_us'])] for r in w4_summary},
        'Shape index: M=1 / 128 / 512 (see report for N,K)','Finite measured best (us)',render_png=False)
    exact=[r for r in l3['results'] if r['method']=='exact' and r['weights']=='uniform']
    core._line_chart(figures/'research-phase1-compiled-count.svg','Config budgets and compiler specializations',
        'T1 M/N/K are constexpr. Count is not fresh JIT invocations or deployment byte size.',
        {'selected config dictionaries':[(r['k'],r['config_count']) for r in exact],
         'retain all Shapes x selected configs':[(r['k'],r['retained_all_shape_config_pairs']['compiler_specializations']) for r in exact],
         'retain dispatched pairs only':[(r['k'],r['dispatch_only_artifacts']['compiler_specializations']) for r in exact]},
        'Maximum config dictionary count K','Count',render_png=False)
    print(core.canonical_json({'published_summary':str(data/'research-phase1-summary.json'),
        'published_matrix':str(data/'research-phase1-matrix.json'),'dataset_id':result['dataset']['dataset_id']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--docs',default='docs')
    args=parser.parse_args();summarize(args.root,args.docs)
