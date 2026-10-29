"""Official wire-contract fixtures. No paid endpoint or real API request."""
import json
import pytest
from kernel_tuning_lab.api_inference import (APIProvider,APIAuthorization,APIAuthorizationRequired,
    PublicModelClient,request_body,parse_response,ProviderResponseError,outbound_task)


def provider(**kwargs):return APIProvider('example','https://api.example.test/v1/chat/completions','explicit-test-model',**kwargs)


def test_unauthorized_client_never_inspects_key_or_opens_network(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','SENTINEL_DO_NOT_USE')
    monkeypatch.setattr('urllib.request.build_opener',lambda *x:pytest.fail('unauthorized network'))
    with pytest.raises(APIAuthorizationRequired):PublicModelClient(provider()).invoke({'shape':{'M':1}},system='JSON only')


def test_chat_contract_logs_usage_ids_and_is_explicitly_a_fixture(tmp_path):
    sent=[]
    def transport(payload):
        sent.append(json.loads(payload))
        return {'id':'response-fixture','model':'snapshot-fixture','choices':[{'finish_reason':'stop','message':{'content':'{"partial":{"tile":32}}'}}],
            'usage':{'prompt_tokens':12,'completion_tokens':8}}, {'x-request-id':'request-fixture'}
    log=tmp_path/'calls.jsonl';client=PublicModelClient(provider(),transport=transport,log_path=log)
    result=client.invoke({'shape':{'M':1}},system='Return JSON only')
    assert sent[0]['response_format']=={'type':'json_object'}
    assert result['usage']=={'input_tokens':12,'output_tokens':8} and result['request_id']=='request-fixture'
    assert result['estimated_usd'] is None and result['evidence_level']=='protocol_fixture_not_real_API'
    assert json.loads(log.read_text())['request']['messages'][1]['role']=='user'


def test_responses_protocol_uses_text_format_and_no_persistent_store():
    spec=provider(protocol='responses',structured_output='json_schema')
    body=request_body(spec,[{'role':'user','content':'JSON'}],schema={'type':'object','properties':{},'additionalProperties':False})
    assert body['text']['format']['type']=='json_schema' and body['store'] is False
    result=parse_response(spec,{'id':'r','status':'completed','model':'m','output':[{'type':'reasoning'},{'type':'message','content':[{'type':'output_text','text':'{}'}]}],
        'usage':{'input_tokens':2,'output_tokens':3}})
    assert result['text']=='{}' and result['usage']['input_tokens']==2


@pytest.mark.parametrize('finish,message',[('length',{'content':'{'}),('stop',{'refusal':'no'}),('tool_calls',{'tool_calls':[]})])
def test_incomplete_refused_and_unsupported_tool_outputs_are_not_candidates(finish,message):
    with pytest.raises(ProviderResponseError):parse_response(provider(),{'choices':[{'finish_reason':finish,'message':message}]})


def test_provider_capabilities_and_outbound_boundary_are_not_guessed():
    with pytest.raises(ValueError):request_body(provider(),[],sampling={'temperature':.7})
    with pytest.raises(ValueError):outbound_task({'shape':{'device_uuid':'private'}})
    with pytest.raises(ValueError):outbound_task({'private_source':'do not send'})
    with pytest.raises(ValueError):outbound_task({'development_feedback':{'test_labels_read':True,'observations':[]}})
    with pytest.raises(ValueError):APIAuthorization(max_requests=float('nan'))
