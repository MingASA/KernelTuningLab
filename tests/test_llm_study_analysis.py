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


def test_research_preparation_is_cpu_safe_and_previews_same_full_tasks(tmp_path):
    from examples.h800_llm_study.prepare import prepare
    from kernel_tuning_lab.research import expand_research
    model=tmp_path/'model.json';model.write_text(json.dumps({'full_weight_hashes':True,'fixture':'not an actual model'}))
    rule=tmp_path/'rule.json';rule.write_text(json.dumps({'generation_evidence':{'evidence_level':'real_local_model_inference','fixture_not_actual_model':True}}))
    spec=prepare(model,rule);plan=expand_research(spec)
    assert plan['planned']['jobs']==18 and plan['planned']['max_observations']==144
    assert spec['algorithms'][4]['options']['structured_output']


def test_preflight_rejection_stays_separate_from_corrected_warm_result():
    from examples.h800_llm_study.analyze import method
    trial={'algorithm':'smac_rf@1.0.0','algorithm_options':{'warm_start_configs':[{'x':1}]},'status':'not_run'}
    assert method(trial)=='SMAC/preflight-rejected-original'
    trial['status']='passed';assert method(trial)=='SMAC/train-prior'


def test_rule_mentions_shape_without_necessarily_changing_config_rank():
    from examples.h800_llm_study.analyze import audit_rule
    assert audit_rule('where(M<32,1,0)+BLOCK_M')['proven_context_invariant_ranking_for_fixed_feasible_set']
    assert not audit_rule('M*BLOCK_M')['proven_context_invariant_ranking_for_fixed_feasible_set']
    assert not audit_rule('UNCERTAINTY')['proven_context_invariant_ranking_for_fixed_feasible_set']


def test_method_comparison_rejects_different_devices_even_with_hardware_bound_space_context(monkeypatch):
    from examples.h800_llm_study import analyze
    from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec,KernelConfig,MeasurementRecord,RecordStatus
    # Explicit CPU fixtures bypass only bundle I/O, never claimed GPU proof.
    bundles=[]
    for index in (1,2):
        record=MeasurementRecord(HardwareSpec('fixture-unit'+str(index)),
            OperatorSpec('fixture',implementation='fixture/kernel'),ShapeSpec({'M':1}),KernelConfig({'x':1}),
            RecordStatus.SUCCESS,runtime_samples_ms=(1.,),seed=7,split='test',provenance='mock',
            metadata={'correctness_verified':True,'fixture_not_actual_gpu':True,'protocol':{'hardware_bound_space':index},
                'correctness':{'input_integrity':{'status':'passed','expected':{'x':'fixture'}}}})
        trial={'job_id':str(index),'record_ids':[record.id],'algorithm_options':{},
            'observations_trace':[{'record_id':record.id,'correctness_verified':True}]}
        bundles.append(({'summary':{'trials':[trial]},'manifest':{'jobs':[]},'calls':{},'trial_files':{}},{record.id:record}))
    monkeypatch.setattr(analyze,'load_bundle',lambda _:bundles.pop(0))
    with pytest.raises(ValueError,match='different physical GPUs'):analyze.evaluate(['fixture1','fixture2'])
