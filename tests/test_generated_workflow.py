"""Frozen policy versions and real SDK contracts, CPU-only."""
import importlib.util
import sys
from pathlib import Path

import pytest

from kernel_tuning_lab.acceptance import check_algorithm
from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec,SearchBudget
from kernel_tuning_lab.policies import PolicyRepository
from kernel_tuning_lab.space import ConfigSpace

PATH=Path(__file__).resolve().parents[1]/'examples/space_production/generated_policy_v1.py'


@pytest.mark.parametrize('filename',['generated_policy_v1.py','generated_policy_v2.py'])
def test_generated_program_uses_existing_contract_and_immutable_policy_repository(tmp_path,filename):
    program=PATH.with_name(filename)
    spec=importlib.util.spec_from_file_location('reviewed_program',program);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    h=HardwareSpec('synthetic');o=OperatorSpec('fixture');s=ShapeSpec({'M':4,'N':64,'K':128})
    space=ConfigSpace({'BLOCK_M':(16,32),'BLOCK_N':(32,64),'BLOCK_K':(32,64)})
    assert check_algorithm(module.ASSETS[0],{},7,h,o,s,space)['status']=='passed'
    policy=module.create_policy(space.read_only(),seed=7);repo=PolicyRepository(tmp_path/'versions')
    frozen=repo.register(policy,'v1',metadata={'development_splits':[],'frozen_before_test':True})
    assert repo.register(policy,'v1',metadata={'development_splits':[],'frozen_before_test':True})==frozen
    assert policy.select_config(h,o,s)
    with pytest.raises(ValueError,match='immutable'):
        repo.register(policy,'v1',metadata={'development_splits':['test']})
    copied=tmp_path/'versions'/policy.name/'v1/source.py'
    assert copied.read_text()==program.read_text()


def test_feedback_rejects_test_roster_before_opening_any_journal(tmp_path,monkeypatch):
    import json
    root=PATH.parents[2];monkeypatch.syspath_prepend(str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('external_workflow',root/'scripts/external_policy_workflow.py')
    workflow=importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)
    protocol=root/'examples/space_production/ai-protocol.json'
    study=tmp_path/'study';study.mkdir()
    (study/'research_manifest.json').write_text(json.dumps({'status':'finished','jobs':[
        {'experiment':{'split':'test'},'shapes':[{'M':32,'N':512,'K':1024}]}]}))
    def forbidden(*args):raise AssertionError('test journals must never be opened')
    monkeypatch.setattr(workflow,'current_records',forbidden)
    with pytest.raises(ValueError,match='test journal was not opened'):
        workflow.export_feedback(study,protocol,tmp_path/'feedback.json')
    assert not (tmp_path/'feedback.json').exists()


def test_frozen_ai_example_has_independent_acceptance_budget_for_each_requested_shape():
    import json
    from kernel_tuning_lab.research import expand_research
    path=PATH.with_name('ai-train-validation.json')
    plan=expand_research(json.loads(path.read_text()),directory=path.parent)
    assert plan['planned']['max_observations']==72
    assert plan['planned']['max_validation_evaluations']==9
    assert all(j['experiment']['validation']['max_configs']>=len(j['shapes']) for j in plan['jobs'])
