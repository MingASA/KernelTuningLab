"""Actual CPU journals prove cache-import identity; no synthetic GPU evidence."""
import importlib.util
import json
from pathlib import Path

from kernel_tuning_lab.research import run_research


def test_study_extension_reuses_exact_experiment_ids_without_rewriting_records(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('rule_study_fixture',root/'scripts/marlin_rule_study.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    value={'schema_version':1,'kind':'research','name':'compatible-extension','algorithms':['grid'],
        'dtypes':['float32'],'seeds':[7],'budgets':[{'max_evaluations':1,'max_wall_time_s':10}],
        'defaults':{'warmup':0,'repetitions':2,'validation':{'max_configs':1,'max_cases':0,'max_wall_time_s':10,'trust_claims':False}},
        'cases':[{'name':'original','operator':'gemm@1.0.0','kernels':['python/gemm@1.0.0'],
                  'shapes':[{'M':2,'N':3,'K':5}]}]}
    path=tmp_path/'spec.json';path.write_text(json.dumps(value));smoke=tmp_path/'smoke'
    first=run_research(path,smoke);assert first['status']=='passed' and first['research_evaluations']==1
    original_ids=json.loads((smoke/'trials.json').read_text())[0]['record_ids']
    value['cases'].append({**value['cases'][0],'name':'additional','shapes':[{'M':3,'N':3,'K':5}]})
    path.write_text(json.dumps(value));target=tmp_path/'matrix'
    prepared=helper.prepare(smoke,path,target)
    assert prepared['new_GPU_measurements']==0 and prepared['source_record_ids']==original_ids
    second=run_research(path,target)
    assert second['status']=='passed' and second['cache_hits']==1 and second['research_evaluations']==1
    assert original_ids==json.loads((smoke/'trials.json').read_text())[0]['record_ids']
    value['name']='different-experiment';path.write_text(json.dumps(value))
    incompatible=helper.prepare(smoke,path,tmp_path/'different')
    assert incompatible['source_observations']==0
