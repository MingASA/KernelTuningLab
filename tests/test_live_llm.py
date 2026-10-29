"""Control-plane fixtures only; actual model/GPU evidence is opt-in experiments."""
import hashlib
import json
from pathlib import Path

import pytest

from kernel_tuning_lab import live_llm
from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec,SearchBudget
from kernel_tuning_lab.search import SearchHistory
from kernel_tuning_lab.space import ConfigSpace


def setup(tmp_path,monkeypatch,texts):
    manifest={'model_id':'unit-not-real-model','full_weight_hashes':True};file=tmp_path/'manifest.json';file.write_text(json.dumps(manifest));sha=hashlib.sha256(file.read_bytes()).hexdigest()
    class Fixture:
        def infer(self,messages,seed,max_tokens,decoding='greedy',constraint=None):
            return {'text':texts.pop(0),'messages':list(messages),'input_token_ids':[],'output_token_ids':[],
                'decoding':decoding,
                'output_tokens':2,'upload_wall_s':0.,'generation_wall_s':0.,
                'offload_and_release_wall_s':0.,'all_parameters_on_CPU_before_Kernel':True,
                'CUDA_allocated_before_bytes':0,'CUDA_allocated_after_bytes':0}
    monkeypatch.setitem(live_llm._SESSIONS,sha,Fixture())
    policy=live_llm.LiveDecision(model_manifest=file,manifest_sha256=sha,max_retries=1)
    policy.bind_trial(output_dir=tmp_path,identity={})
    return policy


def test_live_decision_repairs_only_by_another_model_response_and_archives_both(tmp_path,monkeypatch):
    policy=setup(tmp_path,monkeypatch,['{"config":{"x":99}}','{"config":{"x":1}}'])
    h=HardwareSpec('unit');o=OperatorSpec('fixture');s=ShapeSpec({'M':1,'N':1,'K':1});space=ConfigSpace({'x':(1,2)})
    result=policy.suggest(h,o,s,space.read_only(),SearchHistory(),SearchBudget(1))
    assert result[0].get('x')==1 and policy.diagnostics['actual_model_calls']==2
    rows=[json.loads(x) for x in (tmp_path/'llm-live-calls.jsonl').read_text().splitlines()]
    assert [r['static_valid'] for r in rows]==[False,True]
    assert len(rows[-1]['messages'])==4


def test_live_invalid_outputs_never_fall_back_to_heuristic_or_fake_measurement(tmp_path,monkeypatch):
    policy=setup(tmp_path,monkeypatch,['not JSON','still not JSON'])
    with pytest.raises(ValueError,match='no heuristic substitution'):
        policy.suggest(HardwareSpec('unit'),OperatorSpec('fixture'),ShapeSpec({'N':1}),ConfigSpace({'x':(1,2)}).read_only(),SearchHistory(),SearchBudget(1))
    assert policy.diagnostics['actual_model_calls']==2


def test_decision_parser_rejects_code_and_prediction_as_latency():
    assert live_llm.parse_decision('```json\n{"config":{"x":1}}\n```')['config']=={'x':1}
    with pytest.raises(ValueError):live_llm.parse_decision('{"config":{"x":1},"latency_ms":0.1}')


def test_small_model_presentation_has_scalar_choices_without_nested_defaults():
    space=ConfigSpace({'x':(1,2),'y':(4,8)})
    task=live_llm.decision_task(HardwareSpec('unit'),OperatorSpec('fixture'),ShapeSpec({'M':1}),
        space.read_only(),SearchHistory(),SearchBudget(2))
    assert task['space']['choices']=={'x':[1,2],'y':[4,8]}
    assert 'parameters' not in task['space'] and 'allowed_responses' not in task
    assert set(task['response_contract']['required_keys'])=={'x','y'}


def test_bare_config_pilot_response_gets_only_a_lossless_envelope(tmp_path,monkeypatch):
    policy=setup(tmp_path,monkeypatch,['{"x":2}'])
    result=policy.suggest(HardwareSpec('unit'),OperatorSpec('fixture'),ShapeSpec({'M':1}),
        ConfigSpace({'x':(1,2)}).read_only(),SearchHistory(),SearchBudget(1))
    assert result[0].params=={'x':2}
    row=json.loads((tmp_path/'llm-live-calls.jsonl').read_text())
    assert row['text']=='{"x":2}' and row['response_form']=='bare_complete_config_lossless_envelope'
    for bad in ['{"x":[1,2]}','{"x":1,"latency_ms":0.1}','{"other":1}']:
        with pytest.raises(ValueError):live_llm.parse_decision(bad,{'x'})


def test_sampled_profile_reaches_the_actual_model_call_without_hidden_fallback(tmp_path,monkeypatch):
    policy=setup(tmp_path,monkeypatch,['{"config":{"x":2}}']);policy.decoding='sampled'
    policy.suggest(HardwareSpec('unit'),OperatorSpec('fixture'),ShapeSpec({'M':1}),
        ConfigSpace({'x':(1,2)}).read_only(),SearchHistory(),SearchBudget(1))
    assert json.loads((tmp_path/'llm-live-calls.jsonl').read_text())['decoding']=='sampled'
