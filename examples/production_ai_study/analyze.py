"""Read original records after the target freeze; never runs a GPU or model."""
import argparse,csv,hashlib,json
from pathlib import Path
from kernel_tuning_lab.models import MeasurementRecord,stable_id
from kernel_tuning_lab.measurement_context import source_context

def analyze(evidence,protocol,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[];traces=[];physical_failures=0;experts=[];phases=[]
    freeze=evidence['freeze']
    if freeze['test_performance_opened'] or not freeze['GPU_execution_source']:
        raise ValueError('invalid pre-target freeze')
    for case in protocol['cases']:
        item=evidence['cases'].get(case['id'])
        if item is None:raise ValueError('missing frozen case '+case['id'])
        guard=item['guard'];state=item['state']
        if guard.get('status')!='passed' or guard.get('eligible') is not True:raise ValueError('quarantined/unverified resource batch '+case['id'])
        if state['source_commit']!=freeze['GPU_execution_source'] or state.get('registered_producers_after_close')!=0:raise ValueError('changed source or unclosed producer')
        expected={m for m in protocol['methods'] if m!='B_full_static_grammar' or case['machine']=='H800'}
        if {t['method'] for t in state['trials']}!=expected or len(state['trials'])!=len(expected):raise ValueError('missing/duplicate method')
        recovery=item.get('construction_recovery')
        if recovery and (recovery['guard'].get('eligible') is not True or recovery['guard']['gpu']['uuid']!=guard['gpu']['uuid'] or recovery['state']['source_commit']!=state['source_commit']):raise ValueError('invalid construction recovery binding')
        phases.append({'case':case['id'],'resident_model_load_s':(item.get('resident_model') or {}).get('load_wall_s',0),
            'initial_batch_wall_s':state.get('phase_wall_s',state.get('initial_batch_phase_wall_s')),
            'recovery_batch_wall_s':state.get('recovery_phase_wall_s',0),
            'recovery_resident_model_load_s':((recovery or {}).get('resident_model') or {}).get('load_wall_s',0),
            'initial_failed_construction_s':sum(t['wall_time_s'] for t in item.get('initial_zero_observation_trials',[]))})
        records={}
        for row in item['records']:
            record=MeasurementRecord.from_dict(row['record'])
            if record.id!=row['record_id']:raise ValueError('record identity changed')
            uuid=record.hardware.metadata.get('device_uuid','').lower().removeprefix('gpu-')
            if uuid!=guard['gpu']['uuid'].lower().removeprefix('gpu-'):raise ValueError('mixed physical GPUs')
            if record.shape.dims!=case['test'] or record.seed!=protocol['test_seed'] or record.split!='test':raise ValueError('wrong target task/seed/split')
            records[record.id]=record
        contexts=set()
        for trial in state['trials']:
            if trial['case']!=case['id']:raise ValueError('mixed case')
            revealed=[records[identity] for identity in trial['record_ids']]
            for record in revealed:
                contexts.add(stable_id(source_context(record)))
                if record.operator.implementation!=case['kernel'] or record.operator.parameters!=case['parameters'] or record.operator.dtype!=protocol['dtype']:
                    raise ValueError('mixed implementation/operator/dtype')
            if len(revealed)!=trial['observations']:raise ValueError('observation count mismatch')
            if len(trial['observation_trace'])!=len(revealed) or trial['evaluations']+trial['cache_hits']!=len(revealed):raise ValueError('trace/cache count mismatch')
            for index,(event,record) in enumerate(zip(trial['observation_trace'],revealed),1):
                if event['record_id']!=record.id or event['correctness_verified']!=record.correctness_verified:raise ValueError('trace/reference mismatch')
                if record.correctness_verified and (not record.metadata['correctness'].get('verified_on_gpu') or event['latency_ms']!=record.median_ms):raise ValueError('fake/changed GPU latency')
                physical_failures+=int(not event['cache_hit'] and not record.correctness_verified)
                traces.append({'case':case['id'],'method':trial['method'],'observation':index,
                    'record_id':record.id,'cache_hit':event['cache_hit'],'status':record.status.value,
                    'latency_ms':record.median_ms if record.correctness_verified else None,'best_ms':event['best_verified_latency_ms']})
            verified=[r for r in revealed if r.correctness_verified];best=min((r.median_ms for r in verified),default=None)
            if best!=trial['best_verified_latency_ms']:raise ValueError('best observation was altered')
            costs=trial['algorithm_costs_ms'];diag=trial.get('algorithm_diagnostics',{}).get('optimizer',{})
            rows.append({'case':case['id'],'machine':case['machine'],'method':trial['method'],'status':trial['status'],
                'observations':len(revealed),'new_physical_attempts':trial['evaluations'],'cache_hits':trial['cache_hits'],
                'verified_observations':len(verified),'failed_measurements':trial['failed_measurements'],'best_ms':best,
                'compile_error_observations':trial['statuses'].get('compile_error',0),'numerical_error_observations':trial['statuses'].get('correctness_error',0),
                'factory_ms':costs['factory_ms'],'proposal_ms':costs['proposal_ms'],'feedback_ms':costs['feedback_ms'],
                'trial_wall_s':trial['wall_time_s'],'physical_compile_ms':trial['physical_compile_time_ms'],
                'static_build_s':diag.get('grammar',{}).get('build_wall_s',trial.get('algorithm_diagnostics',{}).get('space',{}).get('wall_s',0)),
                'full_static_configs':diag.get('grammar',{}).get('full_static_configs',diag.get('feasible_count')),
                'model_generation_s':diag.get('generation_wall_s',0),'model_cleanup_s':diag.get('offload_wall_s',0),
                'model_calls':diag.get('actual_model_calls',0),'stop_reason':trial['stop_reason'],
                'best_config':min(verified,key=lambda r:r.median_ms).config.params if verified else None})
        if len(contexts)!=1:raise ValueError('mixed software/timing/residency contexts in '+case['id'])
        if item.get('native'):
            record=MeasurementRecord.from_dict(item['native']['record'])
            if (record.shape.dims!=case['test'] or record.seed!=protocol['test_seed'] or record.split!='test'
                or record.operator.parameters!=case['parameters'] or record.operator.implementation!=case['native']
                or record.operator.dtype!=protocol['dtype'] or record.hardware.metadata['device_uuid'].lower().removeprefix('gpu-')!=guard['gpu']['uuid'].lower().removeprefix('gpu-')):
                raise ValueError('native expert semantic/input mismatch')
            annotations=record.metadata.get('correctness',{}).get('execution',{}).get('annotations',{})
            public_keys=('backend','version','family','mode','scope','configuration_mapping','native_internal_tuning',
                'native_template_id','effective_configuration','native_selection','actual_generated_configurations',
                'actual_raw_launch','actual_per_sequence_splits','native_selected_configuration','actual_launch_parameters')
            experts.append({'case':case['id'],'status':record.status.value,'verified':record.correctness_verified,
                'latency_ms':record.median_ms if record.correctness_verified else None,
                'compile_and_native_selection_ms':record.compile_time_ms,'preparation_ms':record.metadata.get('preparation_time_ms'),
                'phase_wall_time_ms':record.metadata.get('phase_wall_time_ms',{}),
                'annotations':{key:annotations[key] for key in public_keys if key in annotations}})
    for row in rows:
        baseline=next(r for r in rows if r['case']==row['case'] and r['method']=='random')['best_ms']
        row['speedup_vs_random']=baseline/row['best_ms'] if baseline and row['best_ms'] else None
    with (output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows([{**r,'best_config':json.dumps(r['best_config'],sort_keys=True)} for r in rows])
    with (output/'trajectories.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(traces[0]),lineterminator='\n');writer.writeheader();writer.writerows(traces)
    report={'scope':protocol['scope'],'rows':rows,'observations':sum(r['observations'] for r in rows),
        'new_physical_attempts':sum(r['new_physical_attempts'] for r in rows),'cache_hits':sum(r['cache_hits'] for r in rows),
        'revealed_failed_measurements':sum(r['failed_measurements'] for r in rows),'new_physical_failed_attempts':physical_failures,
        'model_calls':sum(r['model_calls'] for r in rows),'native_experts':experts,'phase_costs':phases,
        'notice':'One seed, finite cases; no global optimum or general algorithm winner claim; native expert and model load/generation phase costs are separate.'}
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 生产配置空间补充实测','',report['notice'],'',
        '| Case | Method | Observations | Verified | Failures | Best ms | vs Random | Wall s |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        best='n/a' if r['best_ms'] is None else f"{r['best_ms']:.6f}";ratio='n/a' if r['speedup_vs_random'] is None else f"{r['speedup_vs_random']:.3f}x"
        lines.append(f"| {r['case']} | {r['method']} | {r['observations']} | {r['verified_observations']} | {r['failed_measurements']} | {best} | {ratio} | {r['trial_wall_s']:.2f} |")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--protocol',default=str(Path(__file__).with_name('protocol.json')));p.add_argument('--output',required=True);a=p.parse_args()
    evidence=json.loads(Path(a.evidence).read_text());protocol_bytes=Path(a.protocol).read_bytes()
    if evidence['freeze']['protocol_sha256']!=hashlib.sha256(protocol_bytes).hexdigest():raise ValueError('frozen protocol changed')
    result=analyze(evidence,json.loads(protocol_bytes),a.output)
    print(json.dumps({key:result[key] for key in ('observations','new_physical_attempts','cache_hits','revealed_failed_measurements','new_physical_failed_attempts','model_calls')}))

if __name__=='__main__':main()
