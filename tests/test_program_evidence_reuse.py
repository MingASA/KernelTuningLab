"""Identical C iterations reuse frozen evidence without executing a Runner."""
import json
from pathlib import Path
import pytest
from kernel_tuning_lab import program_artifacts as programs


@pytest.fixture
def versions(tmp_path):
    program=Path(__file__).resolve().parents[1]/'examples/space_production/generated_policy_v1.py'
    response=tmp_path/'response.txt';response.write_text('explicit CPU fixture, not model output')
    provenance={'route':'C','provider':'fixture','model':'none','evidence_level':'fixture'}
    spec={'schema_version':2,'operator':'gemm@1.0.0','dtype':'float32',
        'kernels':[{'asset':'python/gemm@1.0.0'}],'algorithms':[{'asset':'generated/constructive_v1@1.0.0'}],
        'shapes':[{'M':2,'N':3,'K':5}],'split':'train'}
    file=tmp_path/'spec.json';file.write_text(json.dumps(spec));parent=tmp_path/'parent'
    programs.submit(parent,response=response,program=program,algorithm_ref='generated/constructive_v1@1.0.0',provenance=provenance)
    programs.review(parent,'fixture-reviewer')
    manifest=programs.verify(parent)
    for stage,key in [('check','checks'),('development','development')]:
        attempt=parent/'evaluations'/stage;run=attempt/'run';run.mkdir(parents=True)
        bound={**spec,'plugins':[str(parent/'program.py')]};programs.write(attempt/'spec.json',bound)
        programs.write(run/'run_summary.json',{'status':'passed','fixture':True})
        programs.write(run/'trials.json',[])
        manifest[key].append({'status':'passed','stage':stage,'output':str(run),'task_ids':programs.task_ids(spec),
            'summary_sha256':programs.sha(run/'run_summary.json'),'spec_sha256':programs.sha(attempt/'spec.json')})
    programs.write(parent/'artifact.json',manifest);programs.freeze(parent)
    child=tmp_path/'child'
    programs.submit(child,response=response,program=program,algorithm_ref='generated/constructive_v1@1.0.0',provenance=provenance,parent=parent)
    programs.review(child,'fixture-reviewer')
    return parent,child,file


def test_same_program_and_spec_reuse_without_execution(versions,monkeypatch):
    parent,child,file=versions
    def forbidden(*a,**k):raise AssertionError('reuse must never execute')
    monkeypatch.setattr(programs,'execute_asset_experiment',forbidden)
    for stage in ('check','development'):
        result=programs.reuse_evaluation(child,parent,file,stage=stage)
        assert result['new_physical_measurements']==0 and result['evidence_kind']=='historical_evaluation_reuse'
    assert programs.freeze(child)
    assert programs.feedback(child)['development_feedback']['observations']==[]
    summary=Path(programs.verify(parent)['development'][0]['output'])/'run_summary.json'
    summary.write_text('{"status":"changed"}')
    with pytest.raises(ValueError,match='evidence changed'):programs.verify(child)


def test_reuse_refuses_changed_spec_source_or_program(versions):
    parent,child,file=versions
    spec=json.loads(file.read_text());spec['seed']=99;file.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='identical full specification'):
        programs.reuse_evaluation(child,parent,file,stage='development')
    with pytest.raises(ValueError,match='separate child artifact'):
        programs.reuse_evaluation(child,child.parent/'child',file,stage='development')


def test_reuse_cannot_hide_test_exposure_or_changed_program(versions):
    parent,child,file=versions
    spec=json.loads(file.read_text());spec['split']='test';file.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='exposed tasks'):
        programs.reuse_evaluation(child,parent,file,stage='development')
    (child/'program.py').write_text('# changed')
    with pytest.raises(ValueError,match='changed'):programs.reuse_evaluation(child,parent,file,stage='check')


def test_reuse_rejects_parent_spec_changed_to_match_a_new_request(versions):
    parent,child,file=versions
    entry=programs.verify(parent)['development'][0];saved=Path(entry['output']).parent/'spec.json'
    original=json.loads(saved.read_text());original['seed']=99;saved.write_text(json.dumps(original))
    requested=json.loads(file.read_text());requested['seed']=99;file.write_text(json.dumps(requested))
    with pytest.raises(ValueError,match='specification identity'):
        programs.reuse_evaluation(child,parent,file,stage='development')
