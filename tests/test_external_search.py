"""Three-route interface primitives; fixtures are never LLM/GPU performance evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from kernel_tuning_lab.external_search import ExternalSearch,describe_task
from kernel_tuning_lab.space import ConfigSpace
from kernel_tuning_lab.search import SearchHistory
from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec,SearchBudget,KernelConfig
from kernel_tuning_lab.space_tools import Relation

ROOT=Path(__file__).resolve().parents[1]
H=HardwareSpec('unit',architecture='mock');O=OperatorSpec('unit');S=ShapeSpec({'M':4,'N':512,'K':1024})


def test_artifact_policy_is_loaded_hashed_and_proposes_from_structured_space():
    path=ROOT/'examples/space_production/models/interface_policy.json';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    model=ExternalSearch(model_artifact=path,model_sha256=digest)
    space=ConfigSpace({'BLOCK_M':(16,32,64,128),'BLOCK_N':(32,64,128),'x':tuple(range(1000)),'y':tuple(range(1000))},rules=(Relation('x','==','y'),))
    chosen=model.suggest(H,O,S,space.read_only(),SearchHistory(),SearchBudget(4))[0]
    assert space.validate(chosen,H,O,S).valid and chosen.get('x')==chosen.get('y')
    assert model.diagnostics['evidence_level']=='interface_fixture_not_trained'
    assert model.diagnostics['calls']==1 and model.runtime_identity['model_sha256']==digest
    with pytest.raises(ValueError,match='SHA256'):ExternalSearch(model_artifact=path,model_sha256='bad')


def test_task_description_is_bounded_and_has_no_unrevealed_label_or_file_paths():
    space=ConfigSpace({'x':tuple(range(10000))})
    task=describe_task(H,O,S,space.read_only(),SearchHistory(),SearchBudget(4))
    assert task['space']['parameters']['x']['values'] is None
    assert len(json.dumps(task))<4096 and task['history']==[]
    assert not any(key in task for key in ('oracle_best_latency_ms','test_database','file_path'))


def test_local_process_and_http_contracts_use_same_legal_proposal_protocol(tmp_path):
    fixture=ROOT/'examples/space_production/decision_fixture.py';space=ConfigSpace({'x':(1,2),'y':(1,2)})
    process=ExternalSearch(command=[sys.executable,str(fixture),'--once'],model_id='protocol-fixture-not-LLM')
    assert process.suggest(H,O,S,space.read_only(),SearchHistory(),SearchBudget(1))[0].get('x')==1
    assert process.diagnostics['calls']==1
    server=subprocess.Popen([sys.executable,str(fixture),'--requests','4','--wall-seconds','10'],stdout=subprocess.PIPE,text=True)
    try:
        info=json.loads(server.stdout.readline());assert info['evidence']=='protocol_fixture_not_LLM'
        model=ExternalSearch(endpoint='http://127.0.0.1:'+str(info['port']),model_id='protocol-fixture-not-LLM')
        assert model.suggest(H,O,S,space.read_only(),SearchHistory(),SearchBudget(1))[0].get('x')==1
    finally:
        server.terminate();server.wait(timeout=5)


def test_invalid_output_and_timeout_are_bounded_not_fake_measurements():
    invalid=ExternalSearch(command=[sys.executable,'-c','print("not JSON")'],max_retries=1)
    result=invalid.suggest(H,O,S,ConfigSpace({'x':(1,2)}).read_only(),SearchHistory(),SearchBudget(1))
    assert result==[] and invalid.diagnostics['calls']==2 and invalid.last_proposal_status=='external_proposal_budget_exhausted'
    timeout=ExternalSearch(command=[sys.executable,'-c','import time; time.sleep(5)'],timeout_s=.05,max_retries=0)
    assert timeout.suggest(H,O,S,ConfigSpace({'x':(1,2)}).read_only(),SearchHistory(),SearchBudget(1))==[]
    assert timeout.diagnostics['calls']==1
