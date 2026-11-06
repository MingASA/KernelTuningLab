"""Reviewed external coding-agent revision after development-only GPU feedback.

No test records or oracle loaded. Balanced-by-shape relative scores produced
marginal value rankings; these are revocable preferences, not legality rules.
Every fourth step explores globally; other steps construct or jointly mutate.
"""
import hashlib
from pathlib import Path
import random
from kernel_tuning_lab.sdk import SearchAlgorithmDefinition,SearchBudget,SpaceBuildBudget
from kernel_tuning_lab.policies import PolicyDecision

DEVELOPMENT_FEEDBACK_SHA256='3b23865971225e86d3c63b64a972e79f549d64d3a8093a33f8b550f74bbd9824'
DEVELOPMENT_RECORD_IDS_SHA256='d68bee121cfa191c546ed89a6ebb169771cbc459d696859ef43917a3f6ec5445'
PREFERENCES={'BLOCK_M': [32, 16, 64, 128], 'BLOCK_N': [32, 64, 128], 'BLOCK_K': [64, 32], 'num_warps': [8, 4], 'num_stages': [2, 4, 3, 1], 'GROUP_M': [2, 8, 1, 4], 'SPLIT_K': [16, 8, 4, 2, 1]}

class GeneratedPolicy:
    name='generated_feedback_v2'
    version='2.0.0'
    training_records=()
    def __init__(self,space=None,seed=0):
        self.space=space;self.seed=seed;self.rng=random.Random(seed)
        self.diagnostics={'code_origin':'reviewed external coding agent','version':'v2',
            'development_feedback_sha256':DEVELOPMENT_FEEDBACK_SHA256,'preference_revocations':0,
            'global_explorations':0,'joint_mutations':0,'test_labels_used_for_development':False}
    @property
    def runtime_identity(self):
        return {'program_version':type(self).version,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'development_feedback_sha256':DEVELOPMENT_FEEDBACK_SHA256,
            'development_record_ids_sha256':DEVELOPMENT_RECORD_IDS_SHA256,
            'development_splits':['train','validation'],'test_labels_used_for_development':False}
    @property
    def specification(self):
        return {'program_version':type(self).version,'seed':self.seed,'proposal':'marginal development preferences + online joint mutation + global exploration',
            'development_feedback_sha256':DEVELOPMENT_FEEDBACK_SHA256,'preferences':PREFERENCES}
    def _ranked(self,key,space):
        domain=list(space.domains[key]);return [v for v in PREFERENCES.get(key,[]) if v in domain]+[v for v in domain if v not in PREFERENCES.get(key,[])]
    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0):return []
        self.space=space;seen={r.config.id for r in history}|set(getattr(history,'attempted_config_ids',()));step=len(history)
        partial={};keys=[k for k in ('BLOCK_N','SPLIT_K','num_stages','BLOCK_K','num_warps') if k in space.domains]
        good=[r for r in history if r.correctness_verified]
        if step%4==3:
            self.diagnostics['global_explorations']+=1
        elif step%4==2 and good:
            partial=dict(min(good,key=lambda r:r.median_ms).config.params)
            movable=[k for k in space.domains if len(space.domains[k])>1]
            for key in self.rng.sample(movable,min(2,len(movable))):
                choices=[v for v in self._ranked(key,space) if v!=partial[key]];partial[key]=self.rng.choice(choices)
            self.diagnostics['joint_mutations']+=1
        else:
            # Only three selected coordinates are preferred; the remaining
            # coordinates are constructed from the full authoritative space.
            for index,key in enumerate(keys[:3]):
                ranked=self._ranked(key,space);partial[key]=ranked[(step//(2**index))%min(2,len(ranked))]
        result=space.complete(partial,hardware,operator,shape,count=min(batch_size,budget.max_evaluations),
            rng=self.rng,exclude_ids=seen,budget=SpaceBuildBudget(max_nodes=20000,max_wall_time_s=.5))
        if result.configs:return list(result.configs)
        self.diagnostics['preference_revocations']+=1
        return list(space.complete({},hardware,operator,shape,count=min(batch_size,budget.max_evaluations),
            rng=self.rng,exclude_ids=seen,budget=SpaceBuildBudget(max_nodes=20000,max_wall_time_s=.5)).configs)
    def select_config(self,hardware,operator,shape):return tuple(self.suggest(hardware,operator,shape,self.space,(),SearchBudget(1)))
    def search_config(self,hardware,operator,shape,history,budget):
        candidates=tuple(self.suggest(hardware,operator,shape,self.space,history,budget));good=[r for r in history if r.correctness_verified]
        return PolicyDecision(candidates,min(good,key=lambda r:r.median_ms).config if good else None)


def create_policy(space,seed=0):return GeneratedPolicy(space,seed)
def create_searcher(seed=0):return GeneratedPolicy(seed=seed)
ASSETS=(SearchAlgorithmDefinition('generated/feedback_v2','2.0.0',create_searcher,
    metadata={'route':'reviewed external program revision','feedback_sha256':DEVELOPMENT_FEEDBACK_SHA256,
        'development_splits':['train','validation'],'code_is_trusted_reviewed_python':'not a malicious-code sandbox'}),)
