"""Synthetic unit fixtures only: no GPU measurements or published GPU claims."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_cross_gpu as analysis
from summarize_cross_gpu import publication_matrix
from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,MeasurementRecord,RecordStatus,SearchBudget
from kernel_tuning_lab.search import SearchHistory


def rows(name,latencies):
    hardware=HardwareSpec('SYNTHETIC '+name,architecture='mock',metadata={'device_uuid':'GPU-PRIVATE'})
    operator=OperatorSpec('unit_fixture',kind='unit_fixture',implementation='synthetic_no_gpu')
    result=[]
    for config,latency in zip(analysis.CONFIGS,latencies):
        result.append(MeasurementRecord(hardware,operator,analysis.SHAPES[0],config,RecordStatus.SUCCESS,
            runtime_samples_ms=(float(latency),),provenance='mock',split='test',seed=7,wall_time_s=1,
            environment={'torch_execution_context':{'compiler_overrides':{'TRITON_PTXAS_PATH':'/private/server/bin/ptxas'}}},
            metadata={'correctness':{'correctness_origin':'synthetic_unit_fixture_only'}}))
    return result


def test_transfer_scores_use_target_optimum_and_topk_is_only_oracle_coverage():
    configs=analysis.CONFIGS[:4]
    result=analysis.paired_metrics([[1,4,3,2]],[[20,1,9,5]],configs)[0]
    assert result['source_top1_target_regret']==19
    assert result['source_topk_target_coverage'][1]=={'k':2,'regret':4}
    assert result['top_overlap']['0.05']['intersection']==0


def test_single_source_prior_never_uses_target_oracle_to_choose_config():
    configs=analysis.CONFIGS[:3];source=[[1,2,4],[4,2,1]]
    first=analysis.paired_metrics(source,[[100,5,1],[1,5,100]],configs)
    second=analysis.paired_metrics(source,[[1,5,100],[100,5,1]],configs)
    assert all(r['source_global_config']==dict(configs[1].params) for r in first+second)
    assert all(r['source_global_target_regret']==4 for r in first+second)


def test_replay_threshold_cost_is_censored_and_rng_variance_is_within_physical_case():
    group=[{'trace':[{'evaluation':1,'best_latency_ms':v}],
            'oracle_best_latency_ms':1,'shape_id':'same','input_seed':7,'regret':v-1}
           for v in (1.04,1.08,2)]
    result=analysis.threshold_statistics(group)
    assert result['not_reached_5pct']==2 and result['not_reached_10pct']==1
    assert result['mean_observations_to_5pct_when_reached']==1
    assert result['mean_rng_regret_std_within_shape_input']>0


def test_hardware_fold_retains_physical_ids_and_static_priority_only_sees_revealed_history():
    source=rows('source',[1,2,3,4]);target=rows('target',[4,3,2,1]);configs=analysis.CONFIGS[:4]
    fold=analysis.hardware_fold(source,target,'unit hardware fold')
    assert all(r.split=='test' for r in source+target)
    assert [r.metadata['hardware_holdout_fold']['source_record_id'] for r in fold.train]==[r.id for r in source]
    observed=[]
    class Spy(analysis.StaticSourcePriority):
        def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
            observed.append(tuple(r.config.id for r in history))
            return super().suggest(hardware,operator,shape,space,history,budget,batch_size)
    result=analysis.replay_case(fold.test,Spy(fold.train,configs),configs,2,0)
    assert observed==[(),(configs[0].id,)]
    assert result['regret']==2
    assert [s['original_source_record_id'] for s in result['trace']]==[target[0].id,target[1].id]
    with pytest.raises(ValueError,match='leaks'):
        analysis.hardware_fold(source,source,'invalid overlap')


def test_public_device_output_removes_uuid_and_private_compiler_paths():
    dataset={'label':'unit','status':'unit_fixture','records':rows('source',[1]),'coverage':[], 'manifest':{'jobs':[]}}
    public=analysis.public_device(dataset);text=json.dumps(public)
    assert '/private/' not in text and 'GPU-PRIVATE' not in text
    assert public['execution_controls']['compiler_override_names']==['TRITON_PTXAS_PATH']
    assert len(public['execution_controls']['compiler_overrides_sha256'])==64


def test_missing_remote_data_stays_unexecuted_and_protocols_are_distinct(tmp_path):
    assert analysis.load_dataset('H800',tmp_path)['status']=='not_executed'
    a=rows('source',[1])[0]
    from dataclasses import replace
    a=replace(a,metadata={'protocol':{'cuda_graph':{'calls_per_replay':512}}})
    b=replace(a,metadata={'protocol':{'cuda_graph':{'calls_per_replay':128}}})
    assert analysis.pairing_protocol(a)!=analysis.pairing_protocol(b)


def test_pairing_requires_same_commit_sources_scope_and_distinct_physical_identity():
    left={'manifest':{'git_commit':'frozen'},'records':rows('source',[1]),'text_sources':'source bytes','pairing_protocol':'scope512'}
    right={**left,'records':rows('target',[1])}
    assert analysis.pairing_issue(left,right) is None
    assert 'commit' in analysis.pairing_issue(left,{**right,'manifest':{'git_commit':'different'}})
    assert 'identity' in analysis.pairing_issue(left,left)
    assert 'scope' in analysis.pairing_issue(left,{**right,'pairing_protocol':'scope128'})
    assert 'source' in analysis.pairing_issue(left,{**right,'text_sources':'changed reference'})


def test_publication_ignores_stale_files_for_missing_or_unexported_datasets(tmp_path):
    path=tmp_path/'H800-matrix-public.json';path.write_text('{"stale":true}')
    export={'H800':{'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
    assert publication_matrix(tmp_path,{'label':'H800','status':'not_executed'},export) is None
    assert publication_matrix(tmp_path,{'label':'H800','records':1},{}) is None
    assert publication_matrix(tmp_path,{'label':'H800','records':1},export)==path


def test_publication_rejects_changed_or_nonlocal_matrix(tmp_path):
    path=tmp_path/'matrix.json';path.write_text('{}')
    device={'label':'unit','records':1}
    with pytest.raises(ValueError,match='changed'):
        publication_matrix(tmp_path,device,{'unit':{'file':path.name,'sha256':'wrong'}})
    with pytest.raises(ValueError,match='local'):
        publication_matrix(tmp_path,device,{'unit':{'file':'../matrix.json','sha256':'wrong'}})


def test_resource_admission_requires_completed_conflict_free_command_and_physical_identity(tmp_path):
    path=tmp_path/'anchor';path.mkdir();records=rows('source',[1])
    with pytest.raises(ValueError,match='lacks resource'):
        analysis.resource_admission(path,records,required=True)
    assert analysis.resource_admission(path,records)['status']=='not_required_local_desktop'
    receipt={'status':'passed','eligible':True,'exit_code':0,'external_processes_observed':0,
             'conflict_checks':3,'guard_source_sha256':'a'*64,'gpu':{'uuid':'GPU-PRIVATE'},
             'command':['/private/bin/python','collect_phase1.py','--output','/private/anchor'],
             'initial_samples':{'private_other_user':'do not publish'}}
    proof=tmp_path/'anchor-resource-guard.json'
    def check(value):
        proof.write_text(json.dumps(value));return analysis.resource_admission(path,records,required=True)
    public=check(receipt)
    assert public['physical_identity_matched']
    assert '/private' not in json.dumps(public) and 'GPU-PRIVATE' not in json.dumps(public)
    for change in ({'status':'aborted'},{'eligible':False},{'exit_code':75},
                   {'external_processes_observed':1},{'gpu':{'uuid':'GPU-another'}},
                   {'command':['python','--output','/private/another-study']}):
        with pytest.raises(ValueError):check({**receipt,**change})


def test_legacy_receipt_binding_keeps_original_bytes_and_rejects_swapped_runs(tmp_path):
    path=tmp_path/'anchor';path.mkdir();logs=tmp_path/'ops-logs';logs.mkdir()
    manifest={'run_id':'original_dataset'};(path/'research_manifest.json').write_text(json.dumps(manifest))
    proof={'status':'passed','eligible':True,'exit_code':0,'external_processes_observed':0,
           'conflict_checks':3,'guard_source_sha256':analysis.LEGACY_GUARD_SHA256,'gpu':{'uuid':'GPU-PRIVATE'}}
    blob=json.dumps(proof).encode();(tmp_path/'anchor-resource-guard.json').write_bytes(blob)
    (logs/'original.guard.json').write_bytes(blob)
    console=json.dumps({**manifest,'status':'passed'}).encode();(logs/'original.log').write_bytes(console)
    control={'derived_after_collection':True,'stage':'anchor','manifest_run_id':manifest['run_id'],
             'raw_guard_basename':'original.guard.json','raw_guard_sha256':hashlib.sha256(blob).hexdigest(),
             'raw_console_basename':'original.log','raw_console_sha256':hashlib.sha256(console).hexdigest()}
    control_path=tmp_path/'anchor-resource-control.json';control_path.write_text(json.dumps(control))
    assert analysis.resource_admission(path,rows('source',[1]),True)['command_binding'].startswith('legacy_')
    assert (tmp_path/'anchor-resource-guard.json').read_bytes()==blob
    (path/'research_manifest.json').write_text(json.dumps({'run_id':'swapped'}))
    with pytest.raises(ValueError,match='binding differs'):analysis.resource_admission(path,rows('source',[1]),True)


def test_online_analysis_separates_run_drift_from_config_regret_and_checks_observed_best(tmp_path,monkeypatch):
    # Synthetic-only scoring fixture, with admission and source loading replaced.
    from dataclasses import replace
    prior=rows('target',[1,2,3,4]);fresh=[replace(r,runtime_samples_ms=(r.median_ms/2,)) for r in prior]
    path=tmp_path/'online';path.mkdir();snapshot=tmp_path/'snapshot';snapshot.mkdir()
    trial={'status':'passed','record_ids':[r.id for r in fresh],'observations':4,'physical_evaluations':4,'cache_hits':0,
           'best_verified_record_id':fresh[0].id,'algorithm_options':{'method':'random'},'wall_time_s':4,
           'physical_compile_time_ms':1,'physical_measurement_wall_time_s':3,
           'observation_trace':[{'record_id':r.id,'best_verified_latency_ms':fresh[0].median_ms,'cache_hit':False} for r in fresh]}
    (snapshot/'trials.json').write_text(json.dumps([trial]))
    (path/'research_manifest.json').write_text(json.dumps({'git_commit':analysis.RUNTIME['frozen_experiment_commit'],'jobs':[{}]}))
    (path/'summary.json').write_text(json.dumps(dict(observations=4,research_evaluations=4,cache_hits=0,
        validation_evaluations=0,job_wall_time_s=4,physical_measurement_wall_time_s=3,known_physical_compile_time_ms=1,failed_measurement_observations=0)))
    monkeypatch.setattr(analysis,'load_dataset',lambda *a:{'records':prior})
    monkeypatch.setattr(analysis,'resource_admission',lambda *a,**k:{'status':'synthetic_test_only'})
    monkeypatch.setattr(analysis.core,'current_records',lambda *a:fresh)
    monkeypatch.setattr(analysis.core,'snapshot_directory',lambda *a:snapshot)
    monkeypatch.setattr(analysis.core,'require_baseline_compatible',lambda *a:None)
    analysis.analyze_online([('unit','unused')],[('unit',path)],tmp_path)
    result=json.loads((tmp_path/'online-checks.json').read_text())['trials'][0]
    assert result['point_delta_to_prior_anchor_best']==-.5 and result['selected_config_anchor_regret']==0
    trial['best_verified_record_id']=fresh[1].id;(snapshot/'trials.json').write_text(json.dumps([trial]))
    with pytest.raises(ValueError,match='observed verified best'):
        analysis.analyze_online([('unit','unused')],[('unit',path)],tmp_path)
