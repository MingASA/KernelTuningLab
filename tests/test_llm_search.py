"""Protocol fixtures test adapters; they are never real-LLM/GPU evidence."""
import hashlib
import json

import pytest

from kernel_tuning_lab.llm_search import FrozenPrediction,GeneratedRule,validate_expression,task_key
from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec,SearchBudget,KernelConfig,MeasurementRecord,RecordStatus
from kernel_tuning_lab.search import SearchHistory
from kernel_tuning_lab.space import ConfigSpace

H=HardwareSpec('unit',architecture='mock');O=OperatorSpec('gemm',implementation='unit/gemm@1');S=ShapeSpec({'M':8,'N':64,'K':128})


def artifact(tmp_path,value):
    p=tmp_path/'artifact.json';p.write_text(json.dumps(value));return str(p),hashlib.sha256(p.read_bytes()).hexdigest()


def test_static_predictions_keep_original_order_after_a_slow_target_observation(tmp_path):
    space=ConfigSpace({'BLOCK_M':(16,32),'BLOCK_N':(64,), 'BLOCK_K':(32,), 'num_warps':(4,), 'num_stages':(1,), 'GROUP_M':(1,)})
    configs=list(space.iter_configs());key=task_key(H,O,S,space)
    p,sha=artifact(tmp_path,{'kind':'llm_static_predictions','generation_evidence':{'evidence_level':'real_local_model_inference','fixture_not_actual_model':True},'tasks':{key:{'configs':[c.params for c in configs]}}})
    policy=FrozenPrediction(artifact=p,artifact_sha256=sha)
    first=policy.suggest(H,O,S,space.read_only(),SearchHistory(),SearchBudget(2))[0]
    row=MeasurementRecord(H,O,S,first,RecordStatus.SUCCESS,runtime_samples_ms=(999.,),provenance='mock')
    assert policy.suggest(H,O,S,space.read_only(),SearchHistory((row,)),SearchBudget(1))[0]==configs[1]
    assert policy.diagnostics['mode'].startswith('static')


@pytest.mark.parametrize('text',["__import__('os')",'M.__class__','[M][0]','[x for x in range(2)]','BLOCK_M**M','min(M,N,K)'])
def test_generated_expression_has_no_arbitrary_python_or_unbounded_powers(text):
    with pytest.raises(ValueError):validate_expression(text)


def test_generated_rule_scores_full_space_and_can_use_only_revealed_history(tmp_path):
    pytest.importorskip('numpy')
    space=ConfigSpace({'BLOCK_M':(16,32),'BLOCK_N':(32,64),'BLOCK_K':(32,), 'num_warps':(4,), 'num_stages':(1,2),'GROUP_M':(1,)})
    value={'kind':'llm_generated_rule','generation_evidence':{'evidence_level':'real_local_model_inference','fixture_not_actual_model':True},
        'scope':{'architectures':['mock'],'kernel_refs':[O.implementation]},
        'score_expression':'-BLOCK_M - BLOCK_N - 5 * UNCERTAINTY + 0.01 * NEAREST_MS'}
    p,sha=artifact(tmp_path,value);policy=GeneratedRule(artifact=p,artifact_sha256=sha)
    first=policy.suggest(H,O,S,space.read_only(),SearchHistory(),SearchBudget(4))[0]
    assert first.get('BLOCK_M')==32 and first.get('BLOCK_N')==64
    row=MeasurementRecord(H,O,S,first,RecordStatus.SUCCESS,runtime_samples_ms=(1.,),provenance='mock')
    second=policy.suggest(H,O,S,space.read_only(),SearchHistory((row,)),SearchBudget(3))[0]
    assert second.id!=first.id and space.validate(second,H,O,S).valid
    assert policy.diagnostics['target_feedback_records']==1


def test_protocol_fixture_evidence_cannot_be_loaded_as_a_real_model_artifact(tmp_path):
    p,sha=artifact(tmp_path,{'kind':'llm_static_predictions','generation_evidence':{'evidence_level':'protocol_fixture_not_a_real_model'}})
    with pytest.raises(ValueError,match='real-local-model'):FrozenPrediction(artifact=p,artifact_sha256=sha)
