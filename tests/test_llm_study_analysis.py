"""No target performance may enter generated-policy development."""
import json
import pytest
from examples.h800_llm_study.analyze import train_feedback,load_bundle
from examples.h800_llm_study.design import rule_messages,extract_rule


def test_development_refuses_mixed_test_bundle_before_reading_performance(tmp_path):
    p=tmp_path/'mixed.json';p.write_text(json.dumps({'manifest':{'jobs':[{'experiment':{'split':'train'}},{'experiment':{'split':'test'}}]}}))
    with pytest.raises(ValueError,match='train-only'):train_feedback([p])


def test_resource_aborted_measurements_cannot_become_policy_feedback(tmp_path):
    p=tmp_path/'conflict.json';p.write_text(json.dumps({'batch':{'resource_guard':{'status':'aborted','external_processes_observed':1}}}))
    with pytest.raises(ValueError,match='quarantine'):load_bundle(p)


def test_rule_prompt_refuses_test_labels_and_does_not_reveal_frozen_test_shapes():
    with pytest.raises(ValueError):rule_messages([],train_feedback={'test_labels_read':False,'observations':[{'split':'test'}]})
    messages=rule_messages([],train_feedback={'test_labels_read':False,'observations':[{'split':'train'}]})
    assert 'test_shapes' not in messages[-1]['content'] and 'target_path' not in messages[-1]['content']


def test_generated_policy_is_validated_without_rewriting_model_program():
    assert extract_rule('```json\n{"score_expression":"BLOCK_M+num_warps","rationale":"fixture"}\n```')['score_expression']=='BLOCK_M+num_warps'
    with pytest.raises(SyntaxError):extract_rule('{"score_expression":"BLOCK_M+","rationale":"fixture"}')
