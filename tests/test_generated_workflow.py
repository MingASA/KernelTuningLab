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


def test_generated_program_uses_existing_contract_and_immutable_policy_repository(tmp_path):
    spec=importlib.util.spec_from_file_location('reviewed_program',PATH);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
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
    assert copied.read_text()==PATH.read_text()
