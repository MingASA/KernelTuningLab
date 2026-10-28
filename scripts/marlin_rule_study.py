"""Finite native-Marlin rule/L3 study using existing journals and solvers.

prepare imports exact compatible smoke observations, never acceptance records.
analyze ranks only explicit-native records, then evaluates original rule traces
on those same columns. dispatch-spec creates a NEW GPU verification experiment;
it never claims that solving or exporting a table executed a GPU dispatch.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from analyze_phase1 import current_records
from kernel_tuning_lab.l3 import PerformanceMatrix, ShapeDispatcher, greedy, greedy_local_swap, oracle
from kernel_tuning_lab.l3.dispatcher import UnknownShapeError
from kernel_tuning_lab.models import KernelConfig, ShapeSpec, WorkloadDistribution, canonical_json
from kernel_tuning_lab.research import read_json, run_research
from kernel_tuning_lab.storage import ExperimentStore

EXPLICIT = 'production/marlin_explicit@1.0.0'
NATIVE = ('production/marlin_default@1.0.0', 'production/marlin_atomic_opt_in@1.0.0')


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def prepare(smoke, spec, output):
    output=Path(output)
    if (output/'research_manifest.json').exists():
        manifest=read_json(output/'research_manifest.json')
        if manifest['status']!='generated':
            raise ValueError('refuse to prepare an executing/completed study; resume it instead')
    else:
        run_research(spec,output,write_only=True)
        manifest=read_json(output/'research_manifest.json')
    records=current_records(smoke); imported=0; ids=set()
    for job in manifest['jobs']:
        matching=[r for r in records if r.operator.implementation==job['kernel'] and r.seed==job['seed']
            and dict(r.shape.dims) in job['shapes'] and r.operator.dtype==job['dtype']
            and r.split==job['experiment']['split']]
        if not matching:continue
        # Identity/protocol/hardware are NOT rewritten. Runner revalidates every
        # cache key against the actual loaded source/profile/software context.
        with ExperimentStore(output/job['output']/'measurements.sqlite') as store:
            imported+=store.add_many(matching)
        ids.update(r.id for r in matching)
    result={'status':'prepared','imported_journal_rows':imported,'source_record_ids':sorted(ids),
        'source_observations':len(ids),'new_GPU_measurements':0,
        'notice':'Original records and acquisition costs retained; cache observations are not independent repetitions.'}
    write(output/'smoke-cache-import.json',result);return result


def flags(record):
    trace=record.metadata['correctness']['execution']['annotations']['production']['actual_native_call_flags']
    return KernelConfig({'USE_ATOMIC_ADD':trace['use_atomic_add'],'USE_FP32_REDUCE':trace['use_fp32_reduce']})


def analyze(directory, output):
    directory=Path(directory);spec=read_json(directory/'source_experiment.json')
    shapes=tuple(ShapeSpec(shape) for case in spec['cases'] for shape in case['shapes'])
    if len({s.id for s in shapes})!=len(shapes):raise ValueError('workload shapes must be distinct')
    records=current_records(directory);explicit=[r for r in records if r.operator.implementation==EXPLICIT]
    if not records or any(not r.correctness_verified or not r.metadata['correctness'].get('verified_on_gpu') for r in records):
        raise ValueError('all study records must be reference-verified on actual GPU')
    configs=tuple(KernelConfig({'USE_ATOMIC_ADD':a,'USE_FP32_REDUCE':f}) for a in (False,True) for f in (True,False))
    identities=defaultdict(set)
    for r in records:
        a=r.metadata['correctness']['execution']['annotations']
        identities[r.shape.id,r.seed].add(canonical_json([a['logical_input_identity'],a['backend_packing_identity']]))
    if any(len(v)!=1 for v in identities.values()):raise ValueError('logical input/packing mismatch across implementations')
    policies=[];selections=[];dispatch=None;main_matrix=None
    for name,weights in [('uniform_synthetic',(1,)*len(shapes)),
                         ('decode_skew_synthetic',(8,4,2,1,1,1))]:
        if len(weights)!=len(shapes):raise ValueError('this bounded workload requires six frozen shapes')
        workload=WorkloadDistribution(shapes,weights,name)
        matrix=PerformanceMatrix.from_records(workload,configs,explicit)
        if not matrix.complete:raise ValueError('missing explicit-native cells; never impute or rank missing labels')
        if name=='uniform_synthetic':main_matrix=matrix
        for implementation in NATIVE:
            selected=[];actual=[]
            for i,shape in enumerate(shapes):
                rows=[r for r in records if r.operator.implementation==implementation and r.shape.id==shape.id]
                if not rows:raise ValueError('missing native rule trace')
                decisions={flags(r).id for r in rows}
                if len(decisions)!=1:raise ValueError('native decision changed between input repetitions')
                config=flags(rows[0]);j=next(j for j,c in enumerate(configs) if c.id==config.id)
                selected.append(config);actual.append(statistics.median([r.median_ms for r in rows]))
            same=sum(w*matrix.latencies_ms[i][configs.index(c)] for i,(w,c) in enumerate(zip(workload.normalized_weights,selected)))
            policies.append({'workload':name,'policy':implementation,'actual_distinct_config_count':len({c.id for c in selected}),
                'same_explicit_candidate_weighted_ms':same,'same_candidate_relative_loss':same/matrix.reference_latency_ms-1,
                'separate_native_call_weighted_ms':sum(w*t for w,t in zip(workload.normalized_weights,actual)),
                'assignments':[{'shape':dict(s.dims),'config':dict(c.params)} for s,c in zip(shapes,selected)],
                'notice':'Trace executes pinned original selector; native record timings are separate from explicit-matrix timings. Not forcibly equal K.'})
        for k in (1,2,4):
            for solve in (oracle,greedy,greedy_local_swap):
                result=solve(matrix,k);table=ShapeDispatcher.from_result(matrix,result)
                selections.append({'workload':name,**result.to_dict(),'dispatch':table.to_dict()})
                if name=='uniform_synthetic' and k==2 and solve is oracle:dispatch=table
    try:dispatch.select_config(ShapeSpec({'M':3,'N':512,'K':2048}))
    except UnknownShapeError:unseen='rejected_without_execution_or_budget_expansion'
    else:raise AssertionError('unseen input must not obtain an unmeasured dispatch')
    prod=[r.metadata['correctness']['execution']['annotations']['production'] for r in records]
    symbols=sorted({symbol for p in prod for symbol in p['kernel_symbols'] if 'marlin' in symbol.lower()})
    result={'status':'analyzed_not_GPU_dispatch_proof','source_record_ids':[r.id for r in records],
        'matrix':main_matrix.to_dict(),'native_rules':policies,'selections':selections,'chosen_dispatch':dispatch.to_dict(),
        'unseen_input_policy':unseen,'logical_packing_identity_groups_verified':len(identities),
        'counts':{'candidate_config_count':4,'observed_marlin_symbols':len(symbols),
            'symbol_names':symbols,'per_kernel_compiled_identities':None,'per_kernel_code_bytes':None,
            'loaded_library_files':len({p['host_library_sha256'] for p in prod}),
            'whole_library_container_bytes':sorted({p['host_library_container_bytes'] for p in prod}),
            'deployed_graph_count':None,
            'notice':'Runtime reduction options and symbols are not CUBIN counts. Whole library bytes are not per-config code size; this study does not deploy a Graph cache.'},
        'synthetic_workload':True,'reference_scope':'only four actually executable native option combinations; not global optimum',
        'independent_input_repetitions':sorted({r.seed for r in explicit})}
    write(output,result);return result


def dispatch_spec(analysis, template, destination):
    data=read_json(analysis);base=read_json(template);table=data['chosen_dispatch']
    base.update(name='production-marlin085-verified-finite-dispatch-v1',seeds=[31],
        budgets=[{'max_evaluations':1,'max_wall_time_s':120}])
    base['cases']=[{'name':'dispatch-'+str(i),'operator':'w4a16_gemm@2.0.0','split':'validation',
        'shapes':[entry['shape']],'kernels':[{'asset':EXPLICIT,'space':{k:[v] for k,v in entry['config'].items()}}]}
        for i,entry in enumerate(table['assignments'])]
    write(destination,base)
    write(str(destination)+'.selection.json',{'analysis_sha256':hashlib.sha256(Path(analysis).read_bytes()).hexdigest(),
        'table':table,'unseen_input_policy':'reject; no native fallback outside K budget',
        'evidence':'spec exported; actual verified execution requires the ordinary research command'})
    return {'status':'dispatch_spec_exported','cases':len(base['cases'])}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--smoke',required=True);a.add_argument('--spec',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('analyze');a.add_argument('--study',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('dispatch-spec');a.add_argument('--analysis',required=True);a.add_argument('--template',required=True);a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='prepare':v=prepare(args.smoke,args.spec,args.output)
    elif args.command=='analyze':v=analyze(args.study,args.output);v={k:v[k] for k in ('status','counts','unseen_input_policy')}
    else:v=dispatch_spec(args.analysis,args.template,args.output)
    print(json.dumps(v,indent=2))
