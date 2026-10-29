"""Shared contracts; fixture text is never claimed to be a real model call."""
import hashlib
import json
import random
from pathlib import Path
import pytest
from kernel_tuning_lab.ai_actions import parse_action,resolve_action,space_query
from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec,SearchBudget
from kernel_tuning_lab.search import SearchHistory
from kernel_tuning_lab.space import ConfigSpace
from kernel_tuning_lab import program_artifacts as programs
from kernel_tuning_lab.provider_search import ProviderSearch,cpu_provider_environment


H=HardwareSpec('fixture');O=OperatorSpec('fixture');S=ShapeSpec({'M':1})


def test_provider_sdk_preflight_is_repeat_free_without_invoking_any_model():
    from kernel_tuning_lab.provider_search import ASSETS
    from kernel_tuning_lab.acceptance import check_algorithm
    result=check_algorithm(ASSETS[0],{'provider':{'kind':'cpu_process','command':['never-execute-this-command']}},7,H,O,S,ConfigSpace({'x':(1,4,16)}))
    assert result['status']=='passed' and len(result['cases'])==3


def test_partial_query_region_and_non_tile_categories_share_authoritative_space():
    space=ConfigSpace({'family':('a','b'),'threads':(32,64),'tactic':('x','y')}).read_only()
    result=space_query({'kind':'values','parameter':'family'},H,O,S,space)
    assert result['values']==['a','b']
    config,trace=resolve_action(parse_action('{"partial":{"family":"b"}}'),H,O,S,space,SearchHistory(),rng=random.Random(7))
    assert config.get('family')=='b' and trace['platform_completed_fields']==['tactic','threads']
    with pytest.raises(ValueError):resolve_action({'region':{'family':['invalid']}},H,O,S,space,SearchHistory(),rng=random.Random(1))
    with pytest.raises(ValueError):parse_action('{"config":{},"latency_ms":1}')


def test_provider_queries_then_proposes_without_candidate_pool_or_inherited_keys(tmp_path,monkeypatch):
    provider=ProviderSearch(provider={'kind':'cpu_process','command':['python','not_executed.py'],'model_id':'contract-fixture'})
    provider.bind_trial(output_dir=tmp_path,identity={});requests=[]
    def fixture(task):
        requests.append(json.loads(json.dumps(task)))
        return '{"query":{"kind":"values","parameter":"x","offset":64,"limit":2}}' if len(requests)==1 else '{"config":{"x":65}}'
    monkeypatch.setattr(provider,'_request',fixture)
    config=provider.suggest(H,O,S,ConfigSpace({'x':range(10000)}).read_only(),SearchHistory(),SearchBudget(1))[0]
    assert config.get('x')==65 and requests[-1]['space_query_result']['values']==[64,65]
    assert provider.diagnostics['queries']==1 and provider.diagnostics['calls']==2
    monkeypatch.setenv('PRIVATE_API_TOKEN','must_not_inherit')
    assert 'PRIVATE_API_TOKEN' not in cpu_provider_environment() and cpu_provider_environment()['CUDA_VISIBLE_DEVICES']==''


def test_api_timeout_is_not_retried_as_a_model_output_repair(tmp_path,monkeypatch):
    policy=ProviderSearch(provider={'kind':'cpu_process','command':['python','never.py']})
    policy.bind_trial(output_dir=tmp_path,identity={})
    def timeout(task):raise TimeoutError('ambiguous transport outcome')
    monkeypatch.setattr(policy,'_request',timeout)
    with pytest.raises(TimeoutError):policy.suggest(H,O,S,ConfigSpace({'x':(1,2)}).read_only(),SearchHistory(),SearchBudget(2))
    assert policy.diagnostics['calls']==1 and policy.diagnostics['repairs']==0


def test_local_provider_and_api_program_generation_share_the_same_C_contract(tmp_path,monkeypatch):
    from kernel_tuning_lab.local_inference import LocalModelClient
    from kernel_tuning_lab.api_inference import APIProvider,PublicModelClient,APIAuthorization
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'model_id':'fixture-not-LLM','full_weight_hashes':True}))
    sha=hashlib.sha256(manifest.read_bytes()).hexdigest();client=LocalModelClient(model_manifest=manifest,manifest_sha256=sha)
    class ControlFixture:
        def infer(self,messages,*args):return {'text':'{"program":"# fixture","algorithm_ref":"example@1.0.0","rationale":"test"}','messages':messages}
        def close(self):self.closed=True
    control=ControlFixture();client.session=control
    task={'public_search_interface':'SDK Searcher.suggest(..., SpaceView, revealed history, budget)',
        'development_feedback':{'test_labels_read':False,'observations':[]}}
    local=programs.generate(client,task);assert local['route']=='C'
    requests=[]
    def transport(body):
        request=json.loads(body);requests.append(request)
        return {'status':'completed','id':'fixture','output':[{'type':'message','content':[{'type':'output_text','text':local['text']}]}]},{}
    api=PublicModelClient(APIProvider('fixture','https://api.example.test/v1/responses','contract-model',protocol='responses',structured_output='json_schema'),transport=transport)
    assert programs.generate(api,task)['text']==local['text']
    assert requests[0]['text']['format']['schema']['required']==['program','algorithm_ref','rationale']
    client.close();assert control.closed
    with pytest.raises(ValueError,match='booleans'):APIAuthorization(allow_paid_requests='false')


def test_reviewed_full_sdk_program_contract_development_freeze_test(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1];program=root/'examples/space_production/generated_policy_v1.py'
    original=program.read_text();program=tmp_path/'reviewed.py'
    program.write_text("import os\nassert 'KTL_TEST_SECRET' not in os.environ\n"+original)
    monkeypatch.setenv('KTL_TEST_SECRET','never_forward_to_generated_code')
    response=tmp_path/'response.txt';response.write_text('CPU workflow fixture; not a new model response')
    artifact=tmp_path/'version1'
    programs.submit(artifact,response=response,program=program,algorithm_ref='generated/constructive_v1@1.0.0',
        provenance={'route':'C','provider':'test','model':'none','evidence_level':'contract_fixture_not_LLM'})
    spec={'schema_version':2,'name':'C-workflow-fixture','operator':'gemm@1.0.0','dtype':'float32',
        'kernels':[{'asset':'python/gemm@1.0.0'}],'algorithms':[{'asset':'generated/constructive_v1@1.0.0'}],
        'shapes':[{'M':2,'N':3,'K':5}],'warmup':0,'repetitions':1,'budget':{'max_evaluations':2,'max_wall_time_s':10},
        'validation':{'max_configs':1,'max_cases':0,'max_wall_time_s':10,'trust_claims':False},'split':'validation'}
    file=tmp_path/'spec.json';file.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='review'):programs.evaluate(artifact,file,stage='check')
    programs.review(artifact,'test-reviewer')
    assert programs.evaluate(artifact,file,stage='check')['status']=='passed'
    assert programs.evaluate(artifact,file,stage='development')['status']=='passed'
    before=programs.feedback(artifact);assert before['development_feedback']['test_labels_read'] is False
    assert programs.freeze(artifact)['source_sha256']==hashlib.sha256(program.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='frozen'):programs.evaluate(artifact,file,stage='development')
    spec['split']='test';file.write_text(json.dumps(spec))
    assert programs.evaluate(artifact,file,stage='test')['status']=='passed'
    assert programs.verify(artifact)['test_exposed'] is True
    assert programs.feedback(artifact)==before
    (artifact/'program.py').write_text('# altered')
    with pytest.raises(ValueError,match='changed'):programs.verify(artifact)
