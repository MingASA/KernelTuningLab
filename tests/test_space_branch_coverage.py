import random
from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec
from kernel_tuning_lab.space import ConfigSpace
from kernel_tuning_lab.space_tools import SpaceBuildBudget,Relation

H=HardwareSpec('CPU contract');O=OperatorSpec('fixture');S=ShapeSpec({'N':1})


def test_short_constructive_batch_visits_many_high_level_regions_instead_of_one_prefix():
    for domains in ({'x':tuple(range(64)),'y':tuple(range(10000))},
                    {'y':tuple(range(10000)),'x':tuple(range(64))}):
        space=ConfigSpace(domains)
        result=space.complete({},H,O,S,count=16,rng=random.Random(7),budget=SpaceBuildBudget(max_nodes=3000,max_wall_time_s=3))
        assert len(result.configs)==16
        assert len({c.get('x') for c in result.configs})>=10
        assert len({c.get('y') for c in result.configs})>=10
        assert result.diagnostics['constructive_restarts']>=16


def test_sparse_joint_relations_still_construct_without_cartesian_rejection():
    space=ConfigSpace({'x':tuple(range(10000)),'y':tuple(range(10000))},rules=(Relation('x','==','y'),))
    result=space.complete({},H,O,S,count=16,rng=random.Random(11),budget=SpaceBuildBudget(max_nodes=1000,max_wall_time_s=3))
    assert len(result.configs)==16 and all(c.get('x')==c.get('y') for c in result.configs)
    assert result.diagnostics['final_checks']==16


def test_opaque_scarcity_budget_exhaustion_is_not_a_proof_of_emptiness():
    space=ConfigSpace({'x':tuple(range(1000)),'y':tuple(range(1000))},constraints=(lambda c,*_: 'hidden constraint' if c.get('x')!=999 else None,))
    result=space.complete({},H,O,S,count=8,rng=random.Random(3),budget=SpaceBuildBudget(max_nodes=50,max_wall_time_s=3))
    assert result.status=='budget_exhausted'


def test_exhausted_small_space_and_seeded_restore_have_no_repeated_observations():
    space=ConfigSpace({'x':(0,1),'y':(0,1)})
    first=space.complete({},H,O,S,count=8,rng=random.Random(7))
    second=space.complete({},H,O,S,count=8,rng=random.Random(7))
    assert len(first.configs)==4 and first.status=='complete'
    assert [c.id for c in first.configs]==[c.id for c in second.configs]
