"""Bound existing searchers to the frozen cohort; prior labels are source-only.

This SDK plugin is for bounded online verification, not a new search algorithm.
The checked prior is immutable input, never a target measurement database.
"""
import hashlib
import json
from pathlib import Path

from kernel_tuning_lab.sdk import KernelConfig, SearchAlgorithmDefinition
from kernel_tuning_lab.search import make_searcher
from kernel_tuning_lab.space import ConfigSpace
from kernel_tuning_lab.models import ShapeSpec


class FiniteSearch:
    def __init__(self, seed=0, method='random', configs=None, prior_matrix=None, prior_sha256=None):
        if method not in ('random','heuristic','surrogate','source_static_priority'):
            raise ValueError('unsupported finite-cohort method')
        self.method=method;self.configs=tuple(KernelConfig(c) for c in configs or ())
        if not self.configs or len({c.id for c in self.configs})!=len(self.configs):
            raise ValueError('a nonempty distinct finite cohort is required')
        self.delegate=make_searcher('heuristic' if method=='source_static_priority' else method,seed=seed)
        self.orders={}
        if method=='source_static_priority':
            prior=Path(prior_matrix)
            if not prior.is_absolute():prior=Path(__file__).resolve().parent/prior
            blob=prior.read_bytes()
            if hashlib.sha256(blob).hexdigest()!=prior_sha256:
                raise ValueError('source-only prior content differs from its frozen SHA256')
            matrix=json.loads(blob);prior_configs=[KernelConfig(c) for c in matrix['configs']]
            if {c.id for c in prior_configs}!={c.id for c in self.configs}:
                raise ValueError('source prior and candidate cohort differ')
            for shape,values in zip(matrix['workload']['shapes'],matrix['latencies_ms']):
                if len(values)!=len(prior_configs) or any(v is None or v<=0 for v in values):
                    raise ValueError('source prior requires positive observed labels')
                self.orders[ShapeSpec(shape).id]=tuple(c for _,c in sorted(
                    zip(values,prior_configs),key=lambda pair:(pair[0],pair[1].id)))

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        # SDK contract acceptance uses unrelated spaces. Keep that check useful
        # without interpreting its synthetic Shapes as historical GPU evidence.
        if set(space.domains)!=set(self.configs[0].params):
            return self.delegate.suggest(hardware,operator,shape,space,history,budget,batch_size)
        ids={c.id for c in self.configs}
        def member(config,hw,op,sh):
            return None if config.id in ids and space.validate(config,hw,op,sh).valid else 'outside legal finite cohort'
        finite=ConfigSpace(space.domains,constraints=(member,),identity={'config_ids':sorted(ids)},name='frozen online cohort')
        if self.method!='source_static_priority':
            return self.delegate.suggest(hardware,operator,shape,finite,history,budget,batch_size)
        if budget.exhausted(0,0):return []
        if shape.id not in self.orders:raise ValueError('source prior lacks this Shape; do not read target labels')
        seen={r.config.id for r in history}|set(history.attempted_config_ids)
        return [c for c in self.orders[shape.id] if c.id not in seen and finite.validate(c,hardware,operator,shape).valid][:min(batch_size,budget.max_evaluations)]


ASSETS=(SearchAlgorithmDefinition('crossgpu/finite_search','1.0.0',FiniteSearch),)
