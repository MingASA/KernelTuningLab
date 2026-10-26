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
