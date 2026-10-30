"""Task checks are scoped to a build; environment changes invalidate full sets."""
import random
from kernel_tuning_lab.assets import KernelDefinition,ParameterSpec,SpaceDefinition
from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec,KernelConfig

H=HardwareSpec('CPU contract');O=OperatorSpec('fixture');S=ShapeSpec({'N':1})

def test_task_support_once_per_construction_without_skipping_config_constraints():
    calls=[];checked=[]
    def support(*args):calls.append(1)
    def constraint(config,ctx):
        checked.append(config.get('x'))
        return 'odd' if config.get('x')%2 else None
    kernel=KernelDefinition('fixture/task_checks','1.0.0','fixture@1.0.0',lambda *a:None,
        SpaceDefinition({'x':ParameterSpec(tuple(range(32)))},constraints=(constraint,)),supports=support)
    space=build_effective_space(kernel,H,O,S);original_constraints=space.constraints
    result=space.build_feasible(H,O,S)
    assert result.complete and len(result.configs)==16
    assert calls==[1] and checked==list(range(32))
    assert space.constraints is original_constraints
    assert not space.validate(KernelConfig({'x':1}),H,O,S).valid
    assert len(calls)==2 # direct validation still rechecks its task context

def test_complete_set_cache_invalidates_when_runtime_identity_changes_or_disappears():
    state={'revision':0,'limit':2,'available':True}
    def identity():
        if not state['available']:raise RuntimeError('component removed')
        return {'revision':state['revision']}
    def constraint(config,ctx):return None if config.get('x')<state['limit'] else 'outside current context'
    kernel=KernelDefinition('fixture/runtime_identity','1.0.0','fixture@1.0.0',lambda *a:None,
        SpaceDefinition({'x':ParameterSpec((0,1,2,3))},constraints=(constraint,)),execution_identity=identity)
    space=build_effective_space(kernel,H,O,S)
    assert len(space.build_feasible(H,O,S).configs)==2
    state.update(revision=1,limit=4)
    result=space.sample_feasible(random.Random(7),4,H,O,S)
    assert {c.get('x') for c in result.configs}=={0,1,2,3}
    state['available']=False
    result=space.sample_feasible(random.Random(7),4,H,O,S)
    assert result.status=='unsupported' and not result.configs


def test_cached_sampling_rechecks_support_even_without_an_identity_hook():
    state={'available':True}
    kernel=KernelDefinition('fixture/support_changes','1.0.0','fixture@1.0.0',lambda *a:None,
        SpaceDefinition({'x':ParameterSpec((0,1))}),supports=lambda *a:None if state['available'] else 'unavailable')
    space=build_effective_space(kernel,H,O,S);assert space.build_feasible(H,O,S).complete
    state['available']=False
    result=space.sample_feasible(random.Random(7),1,H,O,S)
    assert result.status=='unsupported' and not result.configs
