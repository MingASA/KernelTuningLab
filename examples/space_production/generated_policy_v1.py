"""Reviewed external coding-agent submission v1, before train/validation feedback.

This program proposes a Shape-dependent preference, then constructs actual legal
configurations. No latency table, learned optimum IDs or test files are loaded.
"""
import hashlib
from pathlib import Path
import random

from kernel_tuning_lab.sdk import SearchAlgorithmDefinition,SearchBudget,SpaceBuildBudget
from kernel_tuning_lab.policies import PolicyDecision


class GeneratedPolicy:
    name='generated_constructive_v1'
    version='1.0.0'
    training_records=()
    def __init__(self,space=None,seed=0):
        self.space=space;self.seed=seed;self.rng=random.Random(seed)
        self.diagnostics={'code_origin':'reviewed external coding agent','version':'v1',
            'development_feedback':'none yet','preference_revocations':0}
    @property
    def runtime_identity(self):
        return {'program_version':type(self).version,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'development_feedback':'none','test_labels_used':False}
    @property
    def specification(self):return {'program_version':type(self).version,'seed':self.seed,'sampling':'constructive nonuniform, no fixed random pool'}
    def _partial(self,hardware,shape,space):
        result={}
        if 'BLOCK_M' in space.domains:
            m=shape.get('M');result['BLOCK_M']=min(space.domains['BLOCK_M'],key=lambda v:(abs(v-min(m,64)),v))
        if 'BLOCK_N' in space.domains:result['BLOCK_N']=max(v for v in space.domains['BLOCK_N'] if v<=max(shape.get('N'),min(space.domains['BLOCK_N'])))
        return result
    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0):return []
        self.space=space;seen={r.config.id for r in history}|set(getattr(history,'attempted_config_ids',()))
        result=space.complete(self._partial(hardware,shape,space),hardware,operator,shape,count=min(batch_size,budget.max_evaluations),
            rng=self.rng,exclude_ids=seen,budget=SpaceBuildBudget(max_nodes=20_000,max_wall_time_s=.5))
        if result.configs:return list(result.configs)
        self.diagnostics['preference_revocations']+=1
        return list(space.complete({},hardware,operator,shape,count=min(batch_size,budget.max_evaluations),
            rng=self.rng,exclude_ids=seen,budget=SpaceBuildBudget(max_nodes=20_000,max_wall_time_s=.5)).configs)
    def select_config(self,hardware,operator,shape):
        return tuple(self.suggest(hardware,operator,shape,self.space,(),SearchBudget(1)))
    def search_config(self,hardware,operator,shape,history,budget):
        candidates=tuple(self.suggest(hardware,operator,shape,self.space,history,budget))
        valid=[r for r in history if r.correctness_verified]
        best=min(valid,key=lambda r:r.median_ms).config if valid else None
        return PolicyDecision(candidates,best)


def create_policy(space,seed=0):return GeneratedPolicy(space,seed)
def create_searcher(seed=0):return GeneratedPolicy(seed=seed)
ASSETS=(SearchAlgorithmDefinition('generated/constructive_v1','1.0.0',create_searcher,
    metadata={'route':'external code submission, contract, train/validation, freeze, test',
              'generated_code_is_trusted_reviewed_python':'not a malicious-code sandbox'}),)
