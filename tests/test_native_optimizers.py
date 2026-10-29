"""Optional native-library CPU tests. Synthetic costs are never GPU evidence."""
from copy import deepcopy
import os
import random

import pytest

pytest.importorskip('smac')
pytest.importorskip('kernel_tuner')
pytestmark=pytest.mark.skipif(os.environ.get('PYTHONHASHSEED')!='0',reason='native replay test requires PYTHONHASHSEED=0')

from kernel_tuning_lab.optimizers import SMACSearch,KernelTunerDE
from kernel_tuning_lab.models import (HardwareSpec,OperatorSpec,ShapeSpec,KernelConfig,SearchBudget,
                                      MeasurementRecord,RecordStatus)
from kernel_tuning_lab.space import ConfigSpace
from kernel_tuning_lab.space_tools import Relation
from kernel_tuning_lab.search import SearchHistory
from kernel_tuning_lab.acceptance import request_candidates

HW=HardwareSpec('synthetic optimizer contract',architecture='mock')
OP=OperatorSpec('synthetic_optimizer_fixture',kind='fixture')
SHAPE=ShapeSpec({'N':1})


def objective(config):return (config.get('x')-3)**2+(config.get('y')-6)**2+.1


def drive(searcher,count,space=None,cache=None):
    space=space or ConfigSpace({'x':tuple(range(8)),'y':tuple(range(8))});records=[];cache={} if cache is None else cache;physical=0
    for step in range(count):
        history=SearchHistory(tuple(records),frozenset(r.config.id for r in records))
        candidates=request_candidates(searcher,HW,OP,SHAPE,space,history,SearchBudget(count-step))
        if not candidates:break
        config=candidates[0]
        cached=config.id in cache
        if not cached:
            cache[config.id]=MeasurementRecord(HW,OP,SHAPE,config,RecordStatus.SUCCESS,
                runtime_samples_ms=(objective(config),),provenance='mock',wall_time_s=.01,
                metadata={'correctness_verified':True,'fixture':'synthetic native optimizer CPU contract'})
            physical+=1
        record=cache[config.id];records.append(record);searcher.observe(deepcopy(record),cache_hit=cached)
    return records,physical,cache


@pytest.mark.parametrize('factory',[SMACSearch,KernelTunerDE])
def test_native_feedback_and_seeded_resume_reconstruct_same_trajectory_without_duplicate_evaluations(factory):
    uninterrupted,_,_=drive(factory(seed=7),12)
    prefix,physical,cache=drive(factory(seed=7),7)
    resumed,new_physical,_=drive(factory(seed=7),12,cache=cache)
    assert [r.config.id for r in prefix]==[r.config.id for r in uninterrupted[:7]]
    assert [r.config.id for r in resumed]==[r.config.id for r in uninterrupted]
    assert physical==7 and new_physical==5


def test_smac_adapter_agrees_with_direct_upstream_ask_tell():
    import ConfigSpace as CS
    from smac import Scenario,HyperparameterOptimizationFacade as Facade
    from smac.runhistory.dataclasses import TrialValue
    from smac.runhistory.enumerations import StatusType
    import tempfile
    adapted=SMACSearch(seed=11);ours,_,_=drive(adapted,12)
    cs=CS.ConfigurationSpace(seed=11)
    cs.add([CS.OrdinalHyperparameter(k,sequence=list(range(8)),default_value=0) for k in ('x','y')])
    with tempfile.TemporaryDirectory() as folder:
        scenario=Scenario(cs,deterministic=True,n_trials=10_000,seed=11,n_workers=1,output_directory=__import__('pathlib').Path(folder),crash_cost=1e12)
        engine=Facade(scenario,target_function=None,initial_design=Facade.get_initial_design(scenario,n_configs=4,max_ratio=1),
                      intensifier=Facade.get_intensifier(scenario,max_config_calls=1),logging_level=False,overwrite=True)
        engine._model._rf_opts['n_jobs']=1;direct=[]
        for _ in range(12):
            trial=engine.ask();config=KernelConfig({k:int(v) for k,v in dict(trial.config).items()});direct.append(config.id)
            engine.tell(trial,TrialValue(cost=objective(config),time=.01,status=StatusType.SUCCESS))
    assert direct==[r.config.id for r in ours]
    assert adapted.diagnostics['model_fitted'] and adapted.diagnostics['proposals_with_fitted_model']>0
    assert any('Acquisition Function' in k for k in adapted.diagnostics['native_proposal_origins'])


def test_kernel_tuner_adapter_reaches_real_evolution_and_flushes_final_feedback():
    engine=KernelTunerDE(seed=19,population_size=4)
    rows,_,_=drive(engine,16)
    assert len(rows)==16 and engine.diagnostics['completed_generations']>=1
    assert engine.diagnostics['feedback_records']==16 and engine.diagnostics['computational_replays']>=16
    assert engine.diagnostics['new_physical_measurements_by_optimizer']==0


def test_native_failure_encoding_never_changes_measured_latency_or_promotes_failure():
    engine=SMACSearch(seed=13);space=ConfigSpace({'x':tuple(range(8)),'y':tuple(range(8))})
    candidate=engine.suggest(HW,OP,SHAPE,space.read_only(),SearchHistory(),SearchBudget(8))[0]
    failed=MeasurementRecord(HW,OP,SHAPE,candidate,RecordStatus.COMPILE_ERROR,failure_reason='synthetic compiler failure',provenance='mock')
    engine.observe(failed)
    assert failed.runtime_samples_ms==() and failed.median_ms is None and engine.diagnostics['failed_feedback']==1


def test_native_smac_exhaustion_after_all_finite_points_is_not_a_contract_failure():
    space=ConfigSpace({'x':(0,1,2),'y':(6,)})
    engine=SMACSearch(seed=7,initial_design_size=2)
    rows,physical,_=drive(engine,8,space)
    assert len(rows)==physical==3
    assert engine.last_proposal_status=='native_space_exhausted'
    assert engine.diagnostics['native_space_exhausted'] and engine.diagnostics['tells']>=3


def test_warm_smac_uses_native_additional_initial_design_then_native_model():
    warm=[{'x':3,'y':6},{'x':4,'y':6}]
    engine=SMACSearch(seed=7,warm_start_configs=warm,warm_start_source={'split':'train','fixture':True})
    rows,_,_=drive(engine,12)
    assert {KernelConfig(c).id for c in warm}<={r.config.id for r in rows[:4]}
    assert engine.diagnostics['warm_initial_configs']==2
    assert engine.diagnostics['total_initial_design_requested']==4
    assert engine.diagnostics['model_fitted'] and engine.diagnostics['proposals_with_fitted_model']>0


def test_smac_structured_equality_mapping_preserves_feasibility():
    space=ConfigSpace({'x':tuple(range(8)),'y':tuple(range(8))},rules=(Relation('x','==','y'),))
    engine=SMACSearch(seed=2);rows,_,_=drive(engine,4,space)
    assert all(r.config.get('x')==r.config.get('y') for r in rows)
    assert engine.diagnostics['mapped_rules']==[Relation('x','==','y').to_dict()]


def test_kernel_tuner_adapter_matches_direct_upstream_evolution():
    import numpy as np
    import scipy.stats.qmc as qmc
    from functools import partial
    from unittest.mock import patch
    from kernel_tuner.searchspace import Searchspace
    from kernel_tuner.strategies import diff_evo
    from kernel_tuner.strategies.common import CostFunc
    from kernel_tuner.interface import Options
    from kernel_tuner.util import TuningBudget
    ours,_,_=drive(KernelTunerDE(seed=23,population_size=4),12)
    domains={'x':list(range(8)),'y':list(range(8))}
    cache={str(i):{'x':x,'y':y} for i,(x,y) in enumerate(__import__('itertools').product(range(8),repeat=2))}
    direct=[];values={}
    class Done(Exception):pass
    class Runner:
        def add_strategy_time(self,value):pass
        def run(self,vectors,options):
            result=[]
            for vector in vectors:
                config=KernelConfig({k:int(v) for k,v in zip(domains,vector)})
                if config.id not in values:
                    if len(direct)==12:raise Done()
                    direct.append(config.id);values[config.id]=objective(config)
                result.append({**config.params,'time':values[config.id]})
            return result
    python_state=random.getstate();numpy_state=np.random.get_state()
    try:
        random.seed(23);np.random.seed(23)
        native=Searchspace(None,None,None,from_cache={'tune_params':domains,'cache':cache},neighbor_method='Hamming-adjacent')
        options=Options(tune_params=domains,objective=['time'],objective_higher_is_better=[False],
            strategy_options=Options(constraint_aware=True),budget=TuningBudget(time_limit=30))
        cost=CostFunc(native,options,Runner());rng=np.random.default_rng(23)
        with patch.object(qmc,'LatinHypercube',partial(qmc.LatinHypercube,rng=rng)):
            with pytest.raises(Done):diff_evo.differential_evolution(native,cost,cost.get_bounds(),4,512,.7,.6,'best1bin',True,False)
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)
    assert direct==[r.config.id for r in ours]
