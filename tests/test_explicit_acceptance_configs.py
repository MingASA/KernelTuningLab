"""Pinned independent validation never narrows optimizer authority."""
import json
import pytest
from kernel_tuning_lab.asset_runner import execute_asset_experiment
from kernel_tuning_lab.asset_experiments import ValidationBudget


def specification(config):
    return {'schema_version':2,'name':'explicit-acceptance-full-search','operator':'gemm@1.0.0',
        'kernels':['python/gemm@1.0.0'],'algorithms':['grid@1.0.0'],'dtype':'float32','shapes':[{'M':2,'N':3,'K':5}],
        'warmup':0,'repetitions':1,'budget':{'max_evaluations':3,'max_wall_time_s':10},
        'validation':{'max_configs':1,'max_cases':0,'max_wall_time_s':10,'trust_claims':False,
            'configs':{'python/gemm@1.0.0':[config]}}}


def test_explicit_safe_acceptance_point_keeps_full_search_and_strict_validation(tmp_path):
    file=tmp_path/'spec.json';file.write_text(json.dumps(specification({'tile_size':1})))
    result=execute_asset_experiment(file,tmp_path/'run')
    assert result['status']=='passed' and result['observations']==3 and result['validation_evaluations']==1
    acceptance=json.loads((tmp_path/'run/acceptance.json').read_text())
    assert acceptance['contexts'][0]['space_probe']['selection']=='explicit_validation_configs'
    assert acceptance['samples'][0]['config']=={'tile_size':1}
    trial=json.loads((tmp_path/'run/trials.json').read_text())[0]
    assert len(trial['record_ids'])==3 # all authoritative Python tile configs


def test_invalid_explicit_acceptance_point_fails_without_silent_replacement(tmp_path):
    file=tmp_path/'spec.json';file.write_text(json.dumps(specification({'tile_size':999})))
    result=execute_asset_experiment(file,tmp_path/'run')
    assert result['status']=='failed' and result['evaluations']==0
    assert 'outside authoritative space' in (tmp_path/'run/acceptance.json').read_text()
    with pytest.raises(ValueError):ValidationBudget(configs={'unversioned':[{}]})
