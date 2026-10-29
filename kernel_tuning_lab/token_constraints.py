"""Finite output-contract grammar for HF prefix_allowed_tokens_fn.

Only complete *statically* feasible configurations are encoded, without any
latency labels. Model logits still select every branch. This is not a repair,
candidate ranking model, random pool, or proof of CUDA feasibility.
"""
from __future__ import annotations
from .models import canonical_json


class ConfigTokenTrie:
    def __init__(self,configs,tokenizer,*,max_nodes=1_000_000):
        self.edges=[{}];self.live=[0];self.terminal={};self.paths={};self.blocked=set()
        self.max_output_tokens=0
        configs=tuple(configs)
        texts=[canonical_json({'config':dict(c.params)}) for c in configs]
        batches=tokenizer(texts,add_special_tokens=False)['input_ids']
        for config,tokens in zip(configs,batches):
            if not tokens:raise ValueError('tokenizer produced an empty config representation')
            self.max_output_tokens=max(self.max_output_tokens,len(tokens)+1)
            node=0;path=[0]
            for token in tokens:
                if token not in self.edges[node]:
                    if len(self.edges)>=max_nodes:raise ValueError('complete config grammar exceeds its bounded node allowance')
                    self.edges[node][token]=len(self.edges);self.edges.append({});self.live.append(0)
                node=self.edges[node][token];path.append(node)
            if node in self.terminal:raise ValueError('distinct configs have the same token representation')
            self.terminal[node]=config.id;self.paths[config.id]=tuple(path)
            for index in path:self.live[index]+=1

    def exclude(self,identities):
        for identity in set(identities)-self.blocked:
            path=self.paths.get(identity)
            if path is not None:
                for index in path:self.live[index]-=1
                self.blocked.add(identity)

    def allowed(self,generated,eos_ids):
        node=0
        for token in generated:
            if token not in self.edges[node]:raise ValueError('generated prefix left the authoritative grammar')
            node=self.edges[node][token]
        if node in self.terminal and self.live[node]>0:return list(eos_ids)
        result=[token for token,target in self.edges[node].items() if self.live[target]>0]
        if not result:raise ValueError('no unrevealed completion for this output prefix')
        return result

    @property
    def remaining(self):return self.live[0]

