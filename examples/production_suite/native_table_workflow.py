"""Finite native-config L3 acceptance; existing records/solver/dispatcher only.

Inputs are explicit files, never a recursive experiment scan. Outputs contain
private measurement bindings: keep under artifacts. No GPU is invoked here.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path

from kernel_tuning_lab.asset_experiments import AssetExperiment,AlgorithmSelection,KernelSelection
from kernel_tuning_lab.asset_runner import _plugin_reference
from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.assets import SpaceContext
from kernel_tuning_lab.catalog import AssetCatalog,register_builtin_search_algorithms
from kernel_tuning_lab.operator_library import register_builtin_assets
from kernel_tuning_lab.plugins import load_plugin,discover_entry_points
from kernel_tuning_lab.models import MeasurementRecord,WorkloadDistribution,SearchBudget,KernelConfig,canonical_json,stable_id
from kernel_tuning_lab.measurement_context import source_context
from kernel_tuning_lab.storage import ExperimentStore
from kernel_tuning_lab.l3 import PerformanceMatrix,greedy_local_swap,MissingMeasurementError
from kernel_tuning_lab.production.dispatch import export_table,ProductionTable


def load_records(paths):
    result={}
    for name in paths:
        path=Path(name)
        if path.suffix in ('.sqlite','.db'):
            with ExperimentStore(path,read_only=True) as store:rows=list(store.iter_records())
        elif path.suffix=='.jsonl':rows=[MeasurementRecord.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
        else:
            value=json.loads(path.read_text())
            if isinstance(value,dict) and 'guard' in value and value['guard'].get('eligible') is not True:
                raise ValueError('input export guard is not eligible; do not rank quarantined observations')
            raw=value.get('records') if isinstance(value,dict) else value
            if not isinstance(raw,list):raise ValueError('JSON input requires a records list')
            rows=[MeasurementRecord.from_dict(row) for row in raw]
        for row in rows:result[row.id]=row
    return tuple(result.values())


def context_key(record):
    value=source_context(record);value['protocol']={k:v for k,v in value['protocol'].items() if k!='config_space'}
    return stable_id(value)


def select_cohort(records,kernel_ref,*,context_id=None,hardware_id=None,operator_id=None,allow_fixture=False):
    rows=[row for row in records if row.operator.implementation==kernel_ref and
        (hardware_id is None or row.hardware.id==hardware_id) and (operator_id is None or row.operator.id==operator_id)]
    if context_id:rows=[row for row in rows if context_key(row)==context_id]
    if not rows:raise ValueError('no observations match the requested implementation/context')
    contexts={(row.hardware.id,row.operator.id,context_key(row)) for row in rows}
    if len(contexts)!=1:raise ValueError('multiple hardware/semantics/timing contexts; select hardware_id/operator_id/context_id explicitly: '+canonical_json(sorted(contexts)))
    if not allow_fixture and any(row.provenance!='measured' or row.metadata.get('correctness',{}).get('verified_on_gpu') is not True for row in rows if row.successful):
        raise ValueError('production table requires actual GPU reference-verified measurements; CPU/mock fixtures need explicit allow_fixture')
    if len({row.shape.id for row in rows})<2:raise ValueError('L3 acceptance requires at least two Shapes; different semantic parameters are not Shapes in one workload')
    return tuple(rows)


def annotations(record):return record.metadata.get('correctness',{}).get('execution',{}).get('annotations',record.metadata.get('experiment_annotations',{}))


def native_comparison(rows,explicit,configs,k):
    first=explicit[0];keys=set(configs[0].params);candidate_ids={cfg.id for cfg in configs};shapes={row.shape.id for row in explicit}
    matched=[]
    for row in rows:
        if not row.correctness_verified or row.hardware.id!=first.hardware.id or row.shape.id not in shapes:continue
        a,b=row.operator,first.operator
        if (a.kind,a.dtype,dict(a.parameters),a.correctness_baseline)!=(b.kind,b.dtype,dict(b.parameters),b.correctness_baseline):continue
        if (row.measurement_mode,row.warmup,row.repetitions)!=(first.measurement_mode,first.warmup,first.repetitions):continue
        left=row.metadata.get('protocol',{});right=first.metadata.get('protocol',{})
        if any(left.get(key)!=right.get(key) for key in ('timing','cuda_graph','measurement_scope','tolerances','input_contract')):continue
        if row.environment.get('native_execution_identity')!=first.environment.get('native_execution_identity'):continue
        trace=annotations(row);values=dict(trace.get('native_selected_configuration',trace.get('effective_configuration',{})))
        launches=list(trace.get('actual_launch_parameters',{}).values())
        for key in keys:
            actual={canonical_json(point[key]):point[key] for point in launches if point.get(key) is not None}
            if len(actual)==1:values[key]=next(iter(actual.values()))
            elif len(actual)>1:values.pop(key,None)
        complete=keys<=set(values) and all(values[key] is not None for key in keys)
        config=KernelConfig({key:values[key] for key in keys}) if complete else None
        matched.append({'record_id':row.id,'shape_id':row.shape.id,'actual_configuration':None if config is None else dict(config.params),
            'inside_explicit_candidate_set':None if config is None else config.id in candidate_ids,'latency_ms':row.median_ms})
    known={stable_id(row['actual_configuration']) for row in matched if row['actual_configuration'] is not None}
    full=bool(matched) and all(row['actual_configuration'] is not None for row in matched)
    return {'status':'observed_native_rule' if matched else 'not_provided_or_not_comparable',
        'rows':matched,'covered_shapes':len({row['shape_id'] for row in matched}),'workload_shapes':len(shapes),
        'actual_config_count':len(known) if full else None,'known_config_count_lower_bound':len(known),
        'same_candidate_coverage':all(row['inside_explicit_candidate_set'] is True for row in matched) if full else None,
        'same_config_budget':len(known)<=k if full else None,
        'notice':'native selections remain actual native rule outputs; missing fields are unknown. No restricted native rule or speedup ratio is fabricated.'}


def build_workflow(records,spec_path,output,kernel_ref,*,context_id=None,hardware_id=None,operator_id=None,
                   ks=(1,2),native_records=(),native_kernel_ref=None,allow_fixture=False,resource_guard=None):
    rows=select_cohort(records,kernel_ref,context_id=context_id,hardware_id=hardware_id,operator_id=operator_id,allow_fixture=allow_fixture)
    first=rows[0];spec_path=Path(spec_path).resolve();template=AssetExperiment.load(spec_path)
    if native_records and not native_kernel_ref:raise ValueError('native comparison requires an explicit native_kernel_ref; do not confuse explicit records with native selections')
    native_records=tuple(row for row in native_records if row.operator.implementation==native_kernel_ref)
    if not allow_fixture:
        if not isinstance(resource_guard,dict) or resource_guard.get('eligible') is not True or resource_guard.get('status')!='passed':
            raise ValueError('real GPU matrix requires its passed/eligible resource guard; raw database success is insufficient')
        physical=resource_guard.get('gpu',{}).get('uuid')
        normalize=lambda value:(value or '').lower().removeprefix('gpu-')
        if not physical or any(normalize(row.hardware.metadata.get('device_uuid'))!=normalize(physical) for row in rows):
            raise ValueError('resource guard physical GPU does not match every selected measurement')
    if template.dtype!=first.operator.dtype:raise ValueError('template dtype differs from observations')
    catalog=AssetCatalog();register_builtin_assets(catalog);register_builtin_search_algorithms(catalog);discover_entry_points(catalog)
    plugins=tuple(_plugin_reference(plugin,spec_path.parent) for plugin in template.plugins)
    for plugin in plugins:load_plugin(plugin,catalog)
    kernel=catalog.resolve('kernel',kernel_ref)
    if kernel.fingerprint!=first.environment.get('kernel_fingerprint',first.metadata.get('protocol',{}).get('kernel_fingerprint')):
        raise ValueError('current implementation fingerprint differs from the measured source; use the frozen source/environment')
    selection=next((item for item in template.kernels if item.asset==kernel_ref),None)
    if selection is None:raise ValueError('template must contain the selected explicit implementation')
    shapes=tuple(sorted({row.shape.id:row.shape for row in rows}.values(),key=lambda shape:tuple(sorted(shape.dims.items()))))
    configs=tuple(sorted({row.config.id:row.config for row in rows}.values(),key=lambda config:config.id))
    workload=WorkloadDistribution(shapes,tuple(1. for _ in shapes),'synthetic uniform finite acceptance workload')
    matrix=PerformanceMatrix.from_records(workload,configs,rows,hardware_id=first.hardware.id,operator_id=first.operator.id)
    if not matrix.complete:raise MissingMeasurementError('selected finite matrix has failed/unverified/unmeasured cells; collect them or explicitly supply a different cohort, never predict-fill')
    spaces=[]
    for shape in shapes:
        space=build_effective_space(kernel,first.hardware,first.operator,shape,selection.space,
            context=SpaceContext(first.hardware,first.operator,shape,first.environment,template.workspace_memory_limit_bytes))
        for config in configs:
            valid=space.validate(config,first.hardware,first.operator,shape)
            if not valid.valid:raise ValueError('current authoritative space rejects observed candidate: '+';'.join(valid.reasons))
        spaces.append(space)
    output=Path(output).resolve()
    if output.exists():raise ValueError('preserve earlier tables; use a new workflow output directory')
    output.mkdir(parents=True,mode=0o700);(output/'matrix.json').write_text(canonical_json(matrix.to_dict())+'\n')
    report={'status':'exported_not_executed','GPU_evidence':not allow_fixture,'matrix_complete':True,'shape_count':len(shapes),
        'candidate_count':len(configs),'record_ids':sorted(row.id for row in rows),'context_id':context_key(first),
        'weight_source':'synthetic uniform, not production traffic','selections':[],'next_commands':[]}
    for k in ks:
        result=greedy_local_swap(matrix,k);table=output/f'config-table-k{k}.json'
        digest=export_table(table,matrix,result,hardware=first.hardware,operator=first.operator,space=spaces[0],kernel_fingerprint=kernel.fingerprint)
        policy=ProductionTable(table_artifact=table,table_sha256=digest)
        for shape,space in zip(shapes,spaces):policy.select(first.hardware,first.operator,shape,space.read_only())
        run=replace(template,name=template.name+f'-config-table-k{k}',plugins=tuple(dict.fromkeys((*plugins,'kernel_tuning_lab.production.dispatch'))),
            kernels=(selection,),algorithms=(AlgorithmSelection('production/config_table@1.0.0',{'table_artifact':str(table),'table_sha256':digest}),),
            shapes=shapes,parameters=dict(first.operator.parameters),budget=SearchBudget(1,template.budget.max_wall_time_s),
            warmup=first.warmup,repetitions=first.repetitions)
        path=output/f'dispatch-k{k}.json';run.save(path)
        report['selections'].append({'selection':result.to_dict(),'table_sha256':digest,'native_rule':native_comparison(native_records,rows,configs,k),
            'physical_compiled_variants':None,'packing_artifacts':None,'code_bytes':None,'graphs':None})
        report['next_commands'].append(['ktlab','run','--spec',str(path),'--output',str(output/f'actual-dispatch-k{k}')])
    (output/'workflow-summary.json').write_text(canonical_json(report)+'\n');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--records',action='append',required=True)
    parser.add_argument('--native-records',action='append',default=[]);parser.add_argument('--spec',required=True)
    parser.add_argument('--native-kernel-ref')
    parser.add_argument('--kernel-ref',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--context-id');parser.add_argument('--hardware-id');parser.add_argument('--operator-id')
    parser.add_argument('--guard',help='original passed resource-guard JSON, required for actual GPU matrix construction')
    parser.add_argument('--allow-fixture',action='store_true',help='CPU/mock workflow contract only, never GPU evidence')
    args=parser.parse_args();result=build_workflow(load_records(args.records),args.spec,args.output,args.kernel_ref,
        context_id=args.context_id,hardware_id=args.hardware_id,operator_id=args.operator_id,
        native_records=load_records(args.native_records),native_kernel_ref=args.native_kernel_ref,allow_fixture=args.allow_fixture,
        resource_guard=json.loads(Path(args.guard).read_text()) if args.guard else None)
    print(json.dumps({'status':result['status'],'shapes':result['shape_count'],'configs':result['candidate_count'],'next_commands':result['next_commands']}))


if __name__=='__main__':main()
