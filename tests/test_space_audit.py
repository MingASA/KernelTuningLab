"""Audit count matches an independent nested-conditional Cartesian oracle."""
import importlib.util
import itertools
from pathlib import Path

from kernel_tuning_lab.sdk import ParameterSpec, SpaceDefinition, KernelDefinition, HardwareSpec, OperatorSpec, ShapeSpec
from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.models import KernelConfig


def test_nested_activation_count_and_unknown_budget():
    script=Path(__file__).resolve().parents[1]/'scripts/audit_config_spaces.py'
    loader=importlib.util.spec_from_file_location('space_audit',script)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    specs={'mode':ParameterSpec(('a','b')),
        'x':ParameterSpec((0,1,2),active_when={'mode':'b'},default=0),
        'y':ParameterSpec((4,8),active_when={'x':(1,2)},default=4),
        'z':ParameterSpec((10,20))}
    k=KernelDefinition('fixture','1','fixture@1',lambda ctx,**kw:lambda:None,SpaceDefinition(specs))
    space=build_effective_space(k,HardwareSpec('fixture'),OperatorSpec('fixture'),ShapeSpec({'N':1}))
    expected={space.normalize(KernelConfig(dict(zip(specs,values)))).id
              for values in itertools.product(*(s.values for s in specs.values()))}
    count=module.normalized_count(space,specs)
    assert count['status']=='exact' and count['count']==len(expected)==12
    assert module.normalized_count(space,specs,max_parent_states=1)['status']=='unknown'
