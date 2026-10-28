"""Optional, native SMAC/Kernel Tuner adapters; the platform owns all measurements.

No target performance table is loaded. KT's synchronous optimizer is resumed by
seeded computation replay over revealed feedback, not by replacing its evolution
operator. Native optimizer files are algorithm state, not a second experiment DB.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import partial
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import tempfile
import time
import uuid

from .models import KernelConfig, canonical_json
from .sdk import SearchAlgorithmDefinition, SearchCapabilities, SpaceBuildBudget
from .space_tools import Relation, Forbidden


PINS = {'smac':'2.4.1', 'ConfigSpace':'1.2.2', 'kernel_tuner':'1.5.0'}


def _require(*names):
    for name in names:
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('Install the optional optimizer extra; missing ' + name) from exc
        if actual != PINS[name]:
            raise RuntimeError(f'{name}=={PINS[name]} required, found {actual}; no algorithm fallback')


def _source_hash(obj):
    return hashlib.sha256(Path(inspect.getsourcefile(obj)).read_bytes()).hexdigest()


def _plain(value):
    return value.item() if hasattr(value, 'item') else value


class _Audit:
    def __init__(self, seed):
        self.seed = seed; self.instance_id = uuid.uuid4().hex; self.output = None
        self.diagnostics = {'seed':seed,'new_physical_measurements_by_optimizer':0,
            'feedback_semantics':'verified latency or native failure encoding; never fabricated MeasurementRecords'}
        self.last_proposal_status = None

    def bind_trial(self, *, output_dir, identity):
        self.output = Path(output_dir); self.output.mkdir(parents=True, exist_ok=True)
        self._event('start', identity=identity, recovery='reconstruct from fixed seed and revealed compatible cache; do not load native state twice')

    def _event(self, kind, **values):
        if self.output is not None:
            with (self.output/'optimizer-events.jsonl').open('a') as stream:
                stream.write(canonical_json({'event':kind,'optimizer_instance':self.instance_id,**values})+'\n')


class SMACSearch(_Audit):
    name = 'smac_rf'

    def __init__(self, seed=0, initial_design_size=4, max_proposals=128, max_native_trials=10_000):
        super().__init__(seed)
        if any(type(v) is not int or v < 1 for v in (initial_design_size,max_proposals,max_native_trials)):
            raise ValueError('positive native optimizer limits required')
        self.initial_design_size = initial_design_size; self.max_proposals = max_proposals
        self.max_native_trials = max_native_trials; self.engine = None; self.pending = None
        self.told = set(); self.context = None; self.temporary = None
        self.diagnostics.update(engine='SMAC native ask/tell',versions={n:PINS[n] for n in ('smac','ConfigSpace')},
            initial_design_requested=initial_design_size,asks=0,tells=0,static_rejections=0,
            internal_cached_tells=0,proposals_with_fitted_model=0,native_proposal_origins={},failed_feedback=0)

    def _initialize(self, space, hardware, operator, shape):
        _require('smac','ConfigSpace')
        import ConfigSpace as CS
        from smac import Scenario, HyperparameterOptimizationFacade as Facade
        if not os.environ.get('PYTHONHASHSEED','').isdigit():
            raise RuntimeError('SMAC reproducible resume requires a fixed process PYTHONHASHSEED, e.g. PYTHONHASHSEED=0 ktlab research ...')
        self.space = space; self.context = (hardware,operator,shape)
        specs = space.parameter_specs; domains = dict(space.domains)
        if not domains:
            raise ValueError('SMAC adapter requires at least one declared parameter')
        cs = CS.ConfigurationSpace(seed=self.seed); hps = {}; encoders = {}; decoders = {}; states = {}; default_values = {}
        def activation(name):
            if name in states: return states[name]
            state = 'always'
            for parent, raw in specs[name].active_when.items():
                choices = raw if isinstance(raw,(tuple,list)) else (raw,)
                selected = {canonical_json(v) for v in choices}; available = {canonical_json(v) for v in domains[parent]}
                parent_state = activation(parent)
                if parent_state == 'never' or not selected & available:
                    state = 'never'; break
                if parent_state != 'always' or not available <= selected: state = 'conditional'
            states[name] = state; return state
        for name, values in domains.items():
            state = activation(name)
            if state != 'always' and canonical_json(specs[name].default) not in {canonical_json(v) for v in values}:
                raise ValueError('SMAC cannot losslessly map a restricted inactive default: '+name)
            actual = (specs[name].default,) if state == 'never' else values
            numeric = specs[name].kind in ('integer','real')
            encoded = list(actual) if numeric else [canonical_json(v) for v in actual]
            encoders[name] = {canonical_json(v):e for v,e in zip(actual,encoded)}
            decoders[name] = {canonical_json(e):v for v,e in zip(actual,encoded)}
            default = encoders[name].get(canonical_json(specs[name].default),encoded[0])
            default_values[name] = decoders[name][canonical_json(default)]
            if len(encoded) == 1: hp = CS.Constant(name,encoded[0])
            elif numeric: hp = CS.OrdinalHyperparameter(name,sequence=encoded,default_value=default)
            else: hp = CS.CategoricalHyperparameter(name,choices=encoded,default_value=default)
            hps[name] = hp
        cs.add(list(hps.values())); conditions = []
        for name,spec in specs.items():
            if states[name] != 'conditional': continue
            terms=[]
            for parent,raw in spec.active_when.items():
                choices = raw if isinstance(raw,(tuple,list)) else (raw,)
                selected=[encoders[parent][canonical_json(v)] for v in choices if canonical_json(v) in encoders[parent]]
                if states[parent]=='always' and len(selected)==len(domains[parent]):continue
                terms.append(CS.EqualsCondition(hps[name],hps[parent],selected[0]) if len(selected)==1 else CS.InCondition(hps[name],hps[parent],selected))
            if terms:conditions.append(terms[0] if len(terms)==1 else CS.AndConjunction(*terms))
        if conditions:cs.add(conditions)
        mapped=[]; final=[]
        for rule in space.rules:
            if any(states[n]!='always' for n in rule.dependencies):final.append(rule.to_dict());continue
            if rule.check_partial(default_values).status == 'invalid':
                # ConfigSpace requires a valid native default. Do not mutate it
                # and then swallow its validation exception; retain final checks.
                final.append(rule.to_dict());continue
            try:
                if isinstance(rule,Forbidden):
                    terms=[]
                    for name,raw in rule.values.items():
                        choices=raw if isinstance(raw,(tuple,list)) else (raw,)
                        vals=[encoders[name][canonical_json(v)] for v in choices if canonical_json(v) in encoders[name]]
                        if not vals:break
                        terms.append(CS.ForbiddenEqualsClause(hps[name],vals[0]) if len(vals)==1 else CS.ForbiddenInClause(hps[name],vals))
                    else:
                        cs.add(terms[0] if len(terms)==1 else CS.ForbiddenAndConjunction(*terms));mapped.append(rule.to_dict());continue
                if isinstance(rule,Relation):
                    a,b=hps[rule.left],hps[rule.right]
                    factories={'!=':[CS.ForbiddenEqualsRelation],'==':[CS.ForbiddenLessThanRelation,CS.ForbiddenGreaterThanRelation],
                        '<':[CS.ForbiddenGreaterThanEqualsRelation],'<=':[CS.ForbiddenGreaterThanRelation],
                        '>':[CS.ForbiddenLessThanEqualsRelation],'>=':[CS.ForbiddenLessThanRelation]}
                    same_order = domains[rule.left] == domains[rule.right]
                    if same_order and rule.op in factories and all(specs[n].kind in ('integer','real') for n in rule.dependencies):
                        cs.add([fn(a,b) for fn in factories[rule.op]]);mapped.append(rule.to_dict());continue
            except ValueError as exc:
                raise ValueError('SMAC cannot encode this structured rule without changing its semantics') from exc
            final.append(rule.to_dict())
        if self.output is not None:
            directory = self.output/'smac-state'
        else:
            self.temporary = tempfile.TemporaryDirectory(prefix='ktlab-smac-')
            directory = Path(self.temporary.name)
        scenario=Scenario(cs,deterministic=True,n_trials=self.max_native_trials,seed=self.seed,
                          n_workers=1,output_directory=directory,crash_cost=1e12)
        design=Facade.get_initial_design(scenario,n_configs=self.initial_design_size,max_ratio=1.0)
        self.engine=Facade(scenario,initial_design=design,intensifier=Facade.get_intensifier(scenario,max_config_calls=1),
                           target_function=None,logging_level=False,overwrite=True)
        # CPU resource control, retaining the upstream model and its seed.
        self.engine._model._rf_opts['n_jobs']=1
        self.decoders=decoders
        self.diagnostics.update(mapped_rules=mapped,final_check_rules=final,
            opaque_final_checks=space.capabilities['opaque_final_constraints'],
            upstream_source_sha256=_source_hash(Facade),model_class=type(self.engine._model).__name__,
            python_hash_seed=os.environ['PYTHONHASHSEED'],model_cpu_jobs=1)

    def _model_fitted(self):
        model=self.engine._model
        return bool(getattr(getattr(model,'_rf',None),'estimators_',[]))

    def _tell(self, info, record=None, *, reason=None, internal=False, cache_hit=False):
        from smac.runhistory.dataclasses import TrialValue
        from smac.runhistory.enumerations import StatusType
        good=record is not None and record.correctness_verified
        value=TrialValue(cost=record.median_ms if good else 1e12,
            time=0 if record is None or internal or cache_hit else record.wall_time_s,
            status=StatusType.SUCCESS if good else StatusType.CRASHED)
        self.engine.tell(info,value)
        self.diagnostics['tells']+=1;self.diagnostics['failed_feedback']+=int(not good)
        if internal:self.diagnostics['internal_cached_tells']+=1
        if record:self.told.add(record.id)
        self._event('tell',record_id=record.id if record else None,status=record.status.value if record else 'static_rejection',
            native_status=value.status.name,optimizer_internal_cost=value.cost,
            encoding='verified_latency_ms' if good else 'internal_failure_penalty_not_latency',reason=reason,internal_cached_feedback=internal)

    def observe(self, record, *, cache_hit=False):
        if record.id in self.told:return
        if self.pending is None or self.pending[0].id!=record.config.id:
            raise ValueError('SMAC feedback does not match its pending proposal')
        self._tell(self.pending[1],record,cache_hit=cache_hit);self.pending=None

    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0):return []
        if self.engine is None:self._initialize(space,hardware,operator,shape)
        if self.context[2].id!=shape.id:raise ValueError('one SMAC instance cannot mix target cases')
        by_config={r.config.id:r for r in history}
        if self.pending and self.pending[0].id in by_config:self.observe(by_config[self.pending[0].id],cache_hit=True)
        seen=set(history.attempted_config_ids)|set(by_config)
        for _ in range(self.max_proposals):
            try:info=self.engine.ask()
            except (StopIteration,RuntimeError) as exc:
                self.last_proposal_status='native_optimizer_stopped';self.diagnostics['stop']=str(exc);return []
            self.diagnostics['asks']+=1
            values={name:self.decoders[name][canonical_json(_plain(value))] for name,value in dict(info.config).items()}
            config=space.normalize(values);valid=space.validate(config,hardware,operator,shape)
            if not valid.valid:
                self.diagnostics['static_rejections']+=1;self._tell(info,reason='; '.join(valid.reasons));continue
            if config.id in seen:
                self._tell(info,by_config.get(config.id),reason='already revealed or attempted',internal=True);continue
            guided=self._model_fitted();self.diagnostics['model_fitted']=guided
            self.diagnostics['proposals_with_fitted_model']+=int(guided)
            origin=str(getattr(info.config,'origin',None) or 'unknown')
            origins=self.diagnostics['native_proposal_origins'];origins[origin]=origins.get(origin,0)+1
            self.pending=(config,info);self._event('ask',config=dict(config.params),model_fitted=guided,native_origin=origin)
            return [config]
        self.last_proposal_status='candidate_generation_budget_exhausted'
        return []


class _NeedObservation(Exception):
    def __init__(self,config):self.config=config


class KernelTunerDE(_Audit):
    name='kernel_tuner_constraint_de'

    def __init__(self,seed=0,population_size=8,max_generations=512,native_wall_s=10):
        super().__init__(seed)
        if type(population_size) is not int or population_size<1 or type(max_generations) is not int or max_generations<1:
            raise ValueError('positive population/generation limits required')
        self.population_size=population_size;self.max_generations=max_generations;self.native_wall_s=native_wall_s
        self.configs=None;self.known={};self.next_config=None;self.told=set()
        self.diagnostics.update(engine='Kernel Tuner native constraint-aware differential_evolution',
            versions={'kernel_tuner':PINS['kernel_tuner']},population_requested=population_size,
            computational_replays=0,internal_cached_evaluations=0,native_repair_calls=0,
            completed_generations=0,feedback_records=0,
            adaptation='Complete feasible Searchspace; seeded native QMC; fixed equal-size QMC bug; deterministic native computation replay, no benchmark replacement')

    def _initialize(self,space,hardware,operator,shape):
        _require('kernel_tuner')
        from kernel_tuner.strategies import diff_evo
        self.space=space;self.context=(hardware,operator,shape)
        result=space.build_feasible(hardware,operator,shape,budget=SpaceBuildBudget(max_nodes=500_000,max_wall_time_s=15,max_configs=50_000))
        if not result.complete:
            raise ValueError('Kernel Tuner DE requires a complete feasible set within its explicit build budget; '+result.status)
        if not result.configs:
            self.last_proposal_status='proven_empty';self.configs=();return
        self.configs=result.configs
        self.space_diagnostics={'strategy':'complete_feasible_native_searchspace',**result.diagnostics}
        self.diagnostics.update(feasible_count=len(self.configs),population_effective=min(self.population_size,len(self.configs)),
            upstream_source_sha256=_source_hash(diff_evo.differential_evolution))

    def _run_native(self):
        import numpy as np
        import scipy.stats.qmc as qmc
        from unittest.mock import patch
        from kernel_tuner.searchspace import Searchspace
        from kernel_tuner.interface import Options
        from kernel_tuner.strategies import diff_evo
        from kernel_tuner.strategies.common import CostFunc
        from kernel_tuner.util import TuningBudget,CompilationFailedConfig,StopCriterionReached
        domains={key:list(values) for key,values in self.space.domains.items()}
        # Upstream accepts space-only cached configurations; no timings or target
        # oracle are supplied. These are ALL feasible points, not a random pool.
        cache={str(i):{key:c.get(key) for key in domains} for i,c in enumerate(self.configs)}
        py_state=random.getstate();np_state=np.random.get_state();random.seed(self.seed);np.random.seed(self.seed)
        adapter=self;populations=0;began=time.perf_counter();self.diagnostics['computational_replays']+=1
        class SeededSpace(Searchspace):
            def get_LHS_sample_indices(self,num_samples):
                if num_samples==self.size:
                    return np.random.permutation(self.size).tolist()
                return super().get_LHS_sample_indices(num_samples)
        class RunnerBridge:
            def add_strategy_time(self,value):pass
            def run(self,vectors,options):
                output=[]
                for vector in vectors:
                    if time.perf_counter()-began>adapter.native_wall_s:raise StopCriterionReached('adapter CPU replay deadline')
                    config=adapter.space.normalize({k:_plain(v) for k,v in zip(domains,vector)})
                    result=adapter.space.validate(config,*adapter.context)
                    if not result.valid:raise ValueError('native repaired vector is illegal: '+'; '.join(result.reasons))
                    record=adapter.known.get(config.id)
                    if record is None:raise _NeedObservation(config)
                    adapter.diagnostics['internal_cached_evaluations']+=1
                    row=dict(config.params)
                    if record.correctness_verified:row['time']=record.median_ms
                    else:row['__error__']=CompilationFailedConfig()
                    output.append(row)
                nonlocal populations
                populations+=1
                return output
        original_repair=diff_evo.repair
        def trace_repair(vector,space):
            repaired=original_repair(vector,space);self.diagnostics['native_repair_calls']+=1
            if list(vector)!=list(repaired) and self.diagnostics['native_repair_calls']<=64:
                self._event('native_repair',original_proposal=[_plain(v) for v in vector],executed_proposal=[_plain(v) for v in repaired],feedback_object='executed native vector')
            return repaired
        try:
            native=SeededSpace(None,None,None,from_cache={'tune_params':domains,'cache':cache},neighbor_method='Hamming-adjacent')
            options=Options(tune_params=domains,objective=['time'],objective_higher_is_better=[False],
                strategy_options=Options(constraint_aware=True),budget=TuningBudget(time_limit=self.native_wall_s))
            cost=CostFunc(native,options,RunnerBridge())
            rng=np.random.default_rng(self.seed)
            with patch.object(qmc,'LatinHypercube',partial(qmc.LatinHypercube,rng=rng)),patch.object(diff_evo,'repair',trace_repair):
                diff_evo.differential_evolution(native,cost,cost.get_bounds(),min(self.population_size,len(self.configs)),
                    min(self.max_generations,len(self.configs)),0.7,0.6,'best1bin',True,False)
        except _NeedObservation as request:
            return request.config
        except StopCriterionReached as exc:
            self.last_proposal_status='candidate_generation_budget_exhausted';self.diagnostics['stop']=str(exc);return None
        finally:
            self.diagnostics['completed_generations']=max(self.diagnostics['completed_generations'],max(0,populations-1))
            self.diagnostics['model_or_evolution_entered']=self.diagnostics['completed_generations']>0
            random.setstate(py_state);np.random.set_state(np_state)
        self.last_proposal_status='native_optimizer_stopped';return None

    def observe(self,record,*,cache_hit=False):
        if record.id in self.told:return
        self.told.add(record.id);self.known[record.config.id]=deepcopy(record)
        self.diagnostics['feedback_records']+=1
        self._event('tell',record_id=record.id,config=dict(record.config.params),record_status=record.status.value,
            encoding='verified_latency_ms' if record.correctness_verified else 'native failure marker, not measured latency',cache_hit=cache_hit)
        # Flush the final population's feedback too. The resulting next request
        # is pending only; it never causes an unbudgeted measurement.
        self.next_config=self._run_native() if len(self.known)<len(self.configs) else None
        self.diagnostics['pending_unmeasured_proposal']=self.next_config is not None

    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0):return []
        if self.configs is None:self._initialize(space,hardware,operator,shape)
        self.known.update({r.config.id:deepcopy(r) for r in history})
        if not self.configs or len(self.known)>=len(self.configs):self.last_proposal_status='searcher_exhausted';return []
        candidate=self.next_config or self._run_native();self.next_config=None
        if candidate is None:return []
        if candidate.id in history.attempted_config_ids or any(r.config.id==candidate.id for r in history):
            raise ValueError('native DE reconstruction returned a revealed point; refuse silent substitution')
        self._event('ask',config=dict(candidate.params),generations_completed=self.diagnostics['completed_generations'])
        return [candidate]


ASSETS=(
    SearchAlgorithmDefinition('smac_rf','1.0.0',SMACSearch,
        metadata={'upstream':'https://github.com/automl/SMAC3','version':'2.4.1','license':'BSD-3-Clause'}),
    SearchAlgorithmDefinition('kernel_tuner_de','1.0.0',KernelTunerDE,
        capabilities=SearchCapabilities(max_cardinality=50_000),
        metadata={'upstream':'https://github.com/KernelTuner/kernel_tuner/tree/1.5.0','version':'1.5.0','license':'Apache-2.0'}),
)
