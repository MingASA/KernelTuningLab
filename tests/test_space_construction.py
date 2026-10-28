"""Independent finite-space fixtures; these are not GPU feasibility evidence."""
import itertools
import random

import pytest

from kernel_tuning_lab.sdk import (KernelConfig, ParameterSpec, SpaceDefinition, KernelDefinition,
    Relation, Forbidden, SpaceBuildBudget, HardwareSpec, OperatorSpec, ShapeSpec)
from kernel_tuning_lab.space import ConfigSpace
from kernel_tuning_lab.asset_space import build_effective_space

HW = HardwareSpec('unit', architecture='mock')
OP = OperatorSpec('fixture', kind='fixture')
SHAPE = ShapeSpec({'N': 1})


def conditional_space():
    declaration = SpaceDefinition({'mode': ParameterSpec(('a','b')),
        'x': ParameterSpec((0,1,2)),
        'y': ParameterSpec((2,4),active_when={'mode':'b'},default=2)},
        rules=(Forbidden({'mode':'b','x':2,'y':4}),))
    kernel = KernelDefinition('fixture','1','fixture@1',lambda ctx,**kw:lambda:None,declaration)
    return build_effective_space(kernel,HW,OP,SHAPE)


def test_conditional_normalization_and_full_set_match_independent_cartesian_oracle():
    space = conditional_space(); expected = set()
    for mode,x,y in itertools.product(('a','b'),(0,1,2),(2,4)):
        canonical_y = 2 if mode == 'a' else y
        if mode == 'b' and x == 2 and canonical_y == 4:
            continue
        expected.add(KernelConfig({'mode':mode,'x':x,'y':canonical_y}).id)
    result = space.build_feasible(HW,OP,SHAPE)
    assert result.complete and {c.id for c in result.configs} == expected
    assert space.cardinality == 12 and len(result.configs) == 8
    assert space.normalize({'mode':'a','x':1,'y':4}).id == space.normalize({'mode':'a','x':1}).id
    assert not space.validate(KernelConfig({'mode':'a','x':1,'y':4}),HW,OP,SHAPE).valid
    assert len(list(space.iter_configs())) == 9


def test_dependency_construction_reaches_disconnected_sparse_regions_without_cartesian_rejection():
    # Three feasible configurations out of 100 million raw points.
    space = ConfigSpace({'x':tuple(range(10_000)),'y':tuple(range(10_000))},
        rules=(Relation('x','==','y'),),
        constraints=(lambda c,h,o,s:None if c.get('x') in (0,5000,9999) else 'opaque disjoint regions',))
    result = space.build_feasible(HW,OP,SHAPE,budget=SpaceBuildBudget(max_nodes=40_000,max_wall_time_s=8))
    assert result.complete and {c.get('x') for c in result.configs} == {0,5000,9999}
    assert result.diagnostics['nodes'] < 40_000 and result.diagnostics['final_checks'] == 10_000
    sample = space.sample_feasible(random.Random(7),2,HW,OP,SHAPE)
    assert len(sample.configs) == 2 and 'uniform' in sample.diagnostics['distribution']


def test_partial_unknown_never_executes_opaque_callback_and_joint_moves_can_cross_components():
    calls=[]
    def opaque(c,h,o,s):
        calls.append(c);return None
    space=ConfigSpace({'x':(0,1,2),'y':(0,1,2)},rules=(Relation('x','==','y'),),constraints=(opaque,))
    assert space.validate_partial({'x':1},HW,OP,SHAPE).status == 'unknown' and not calls
    assert space.validate_partial({'x':1,'y':2},HW,OP,SHAPE).status == 'invalid' and not calls
    original=KernelConfig({'x':0,'y':0})
    assert not any(space.validate(c,HW,OP,SHAPE).valid for c in space.neighbors(original))
    joint=[c for c in space.neighbors(original,max_changed=2) if space.validate(c,HW,OP,SHAPE).valid]
    assert {c.get('x') for c in joint} == {1,2}
    completed=space.complete({'x':2},HW,OP,SHAPE)
    assert completed.configs == (KernelConfig({'x':2,'y':2}),)


def test_timeout_and_callback_error_are_not_false_empty_proofs():
    space=ConfigSpace({'x':(1,2)},constraints=(lambda *a:'unreachable',))
    timed=space.build_feasible(HW,OP,SHAPE,budget=SpaceBuildBudget(max_wall_time_s=0))
    assert timed.status == 'budget_exhausted' and not timed.complete
    assert space.build_feasible(HW,OP,SHAPE).status == 'proven_empty'
    def broken(*a):raise RuntimeError('constraint failure')
    assert ConfigSpace({'x':(1,)},constraints=(broken,)).build_feasible(HW,OP,SHAPE).status == 'unsupported'


def test_native_candidates_are_not_their_cartesian_envelope_and_keep_stable_parameter_identities():
    space=ConfigSpace.from_configs([{'tactic':'native-A','split':1},{'tactic':'native-B','split':4}],identity={'backend':'unit@1'})
    assert space.cardinality == 2 and len(list(space.iter_configs())) == 2
    assert not space.validate(KernelConfig({'tactic':'native-A','split':4}),HW,OP,SHAPE).valid
    assert space.build_feasible(HW,OP,SHAPE).complete
    view=space.read_only();view.native_configs[0].params['split']=999
    assert space.config_at(0).get('split') == 1
    assert view.capabilities['native_candidates']


def test_excluded_candidates_are_exhausted_not_a_proof_of_an_empty_space():
    space=ConfigSpace({'x':(1,2)})
    excluded={c.id for c in space.iter_configs()}
    result=space.complete({},HW,OP,SHAPE,count=1,exclude_ids=excluded)
    assert result.status=='exhausted'
    assert space.build_feasible(HW,OP,SHAPE).complete


def test_unavailable_asset_runtime_is_unsupported_not_a_false_empty_space_proof():
    kernel=KernelDefinition('fixture','1','fixture@1',lambda ctx,**kw:lambda:None,
        SpaceDefinition({'x':ParameterSpec((1,2))}),supports=lambda h,o,s:'required runtime unavailable')
    space=build_effective_space(kernel,HW,OP,SHAPE)
    built=space.build_feasible(HW,OP,SHAPE)
    assert built.status=='unsupported' and not built.complete and built.diagnostics['nodes']==0
    assert space._feasible_cache=={}
    # A zero construction budget still performs no support work/proof.
    assert space.build_feasible(HW,OP,SHAPE,budget=SpaceBuildBudget(max_wall_time_s=0)).status=='budget_exhausted'


def test_joint_only_neighbor_budget_is_not_consumed_by_one_parameter_moves():
    space=ConfigSpace({'x':tuple(range(1000)),'y':tuple(range(1000))},rules=(Relation('x','==','y'),))
    candidates=space.neighbors(KernelConfig({'x':0,'y':0}),min_changed=2,max_changed=2,max_neighbors=4)
    assert KernelConfig({'x':1,'y':1}) in candidates


def test_preflight_reaches_constructible_sparse_space_without_a_cartesian_random_pool():
    from kernel_tuning_lab.acceptance import probe_legal_configs
    space=ConfigSpace({'anchor':(0,5000,9999),'x':tuple(range(10000)),'y':tuple(range(10000))},
        rules=(Relation('anchor','==','x'),Relation('x','==','y')))
    result=probe_legal_configs(space,HW,OP,SHAPE,seed=7,count=3,max_inspected=64)
    assert result['selected']==3 and result['inspected']==3
    assert {c.get('x') for c in result['configs']}=={0,5000,9999}
    assert 'construction_status' in result


def test_external_optimizer_can_measure_the_optimum_outside_legacy_random_pool(tmp_path):
    from kernel_tuning_lab.acceptance import request_candidates
    from kernel_tuning_lab.backends.mock import MockAdapter
    from kernel_tuning_lab.models import SearchBudget
    from kernel_tuning_lab.runner import BenchmarkRunner
    from kernel_tuning_lab.search import SearchHistory
    # This tests admission/feedback, not GPU performance. A separate physical
    # full-space W4 experiment verifies execution of open-space proposals.
    space=ConfigSpace({'x':tuple(range(10000))})
    pool={c.id for c in space.sample_configs(random.Random(7),512)}
    outside=next(c for c in space.iter_configs() if c.id not in pool)
    class External:
        def suggest(self,h,o,s,view,history,budget,batch_size=1):return [outside]
    candidate=request_candidates(External(),HW,OP,SHAPE,space,SearchHistory(),SearchBudget(1))[0]
    # Exercise the actual Runner and journal, with explicitly synthetic costs.
    from kernel_tuning_lab.space import ValidationResult
    from kernel_tuning_lab.storage import ExperimentStore
    class Objective(MockAdapter):
        def validate(self,*args):return ValidationResult(True)
        def prepare(self,*args):return None
        def compile(self,prepared,config):return config
        def check_correctness(self,compiled):return True,{'synthetic':True,'verified_on_gpu':False}
        def measure(self,compiled,repetitions):return (.01 if compiled.id==outside.id else 1.,)*repetitions
    with ExperimentStore(tmp_path/'records.sqlite') as store:
        runner=BenchmarkRunner(Objective(),store=store,warmup=0,repetitions=2)
        best=runner.measure(HW,OP,SHAPE,candidate,space)
        others=[runner.measure(HW,OP,SHAPE,c,space) for c in space.sample_configs(random.Random(7),8)]
        assert store.get_record(best.id).successful and best.provenance=='mock'
        assert min([best,*others],key=lambda r:r.median_ms).config.id==outside.id
