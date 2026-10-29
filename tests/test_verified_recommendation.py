"""Unverified accepted claims remain evidence, never a tuning recommendation."""
from kernel_tuning_lab.assets import KernelDefinition,OperatorDefinition,ParameterSpec,SpaceDefinition,SearchAlgorithmDefinition
from kernel_tuning_lab.asset_context import AssetCase
from kernel_tuning_lab.asset_search import search_case
from kernel_tuning_lab.backends.mock import MockAdapter
from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec,SearchBudget
from kernel_tuning_lab.search import GridSearch
from kernel_tuning_lab.space import ConfigSpace,ValidationResult
from kernel_tuning_lab.storage import ExperimentStore


class ClaimFixture(MockAdapter):
    """Explicit synthetic timings, not CPU or GPU performance evidence."""
    def validate(self,*args):return ValidationResult(True)
    def prepare(self,*args):return None
    def compile(self,prepared,config):return config
    def check_correctness(self,config):
        verified=config.get('x')==1
        return True,{'status':'verified' if verified else 'declared',
            'correctness_origin':'platform_reference' if verified else 'submitter_claim',
            'comparison_performed':verified,'verified_on_gpu':False,'fixture':'synthetic'}
    def measure(self,config,repetitions):return (.1 if config.get('x')==0 else 2.,)*repetitions


def run_fixture(tmp_path,budget):
    op=OperatorDefinition('fixture','1',('N',),('float16',),lambda ctx,seed:{},lambda ctx:{},
        tolerances={'float16':(0.,0.)})
    kernel=KernelDefinition('fixture/kernel','1',op.ref,lambda ctx,**p:lambda:None,
        SpaceDefinition({'x':ParameterSpec((0,1))}),runtime='python')
    h=HardwareSpec('synthetic');o=OperatorSpec('fixture',kind='fixture',implementation=kernel.ref)
    case=AssetCase(kernel,op,h,o,ShapeSpec({'N':1}),ConfigSpace({'x':(0,1)}),ClaimFixture())
    algorithm=SearchAlgorithmDefinition('fixture/grid','1',GridSearch)
    with ExperimentStore(tmp_path/'records.sqlite') as store:
        result=search_case(case,algorithm,{},SearchBudget(budget,10),store=store,seed=7,
            warmup=0,repetitions=1,split='train',experiment_id='unit',output_dir=tmp_path/'trial')
        assert len(list(store.iter_records()))==budget
    return result


def test_verified_candidate_wins_over_faster_unverified_claim(tmp_path):
    result=run_fixture(tmp_path,2)
    assert result['best_config']=={'x':1}
    assert result['best_latency_ms']==2. and result['recommendation_verified']
    assert result['fastest_observed_success']['config']=={'x':0}
    assert result['fastest_observed_success']['correctness_verified'] is False


def test_claim_only_run_has_no_verified_recommendation(tmp_path):
    result=run_fixture(tmp_path,1)
    assert result['best_config'] is None and result['best_latency_ms'] is None
    assert not result['recommendation_verified']
    assert result['unverified_successes']==1 and result['observations']==1
