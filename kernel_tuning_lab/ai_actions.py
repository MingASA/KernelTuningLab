"""Shared model task/action protocol over the existing authoritative SpaceView."""
from __future__ import annotations
import json
import random
from .models import canonical_json
from .sdk import SpaceBuildBudget


def parse_action(text,parameter_keys=None):
    text=text.strip()
    if '</think>' in text:text=text.rsplit('</think>',1)[1].strip()
    if text.startswith('```') and text.endswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    value=json.loads(text)
    if parameter_keys is not None and isinstance(value,dict) and set(value)==set(parameter_keys):
        return {'config':value,'normalization':'lossless bare-config envelope'}
    if not isinstance(value,dict) or set(value)-{'config','partial','region','query','rationale','prediction'}:
        raise ValueError('one config/partial/advisory region/space query required; no code or file actions')
    if len(set(value)&{'config','partial','region','query'})!=1:raise ValueError('exactly one action selector required')
    return value


def space_query(query,hardware,operator,shape,space):
    if not isinstance(query,dict):raise ValueError('structured space query required')
    kind=query.get('kind');partial=query.get('partial',{});limit=query.get('limit',8)
    if type(limit) is not int or not 1<=limit<=32:raise ValueError('query result limit1..32 required')
    if kind=='validate_partial':
        result=space.validate_partial(partial,hardware,operator,shape)
        return {'kind':kind,'status':result.status,'reasons':list(result.reasons)}
    if kind=='values':
        name=query['parameter'];domain=space.domains.get(name)
        if domain is None:raise ValueError('unknown queried parameter')
        offset=query.get('offset',0)
        if type(offset) is not int or not 0<=offset<=len(domain):raise ValueError('query offset outside domain')
        spec=space.parameter_specs[name]
        return {'kind':kind,'parameter':name,'values':list(domain[offset:offset+limit]),'total':len(domain),
            'next_offset':offset+limit if offset+limit<len(domain) else None,'active_when':dict(spec.active_when),
            'scope':'declaration values only; constraints still authoritative'}
    if kind=='complete':
        result=space.complete(partial,hardware,operator,shape,count=limit,rng=random.Random(query.get('seed',0)),
            budget=SpaceBuildBudget(max_nodes=10000,max_wall_time_s=.5,max_configs=32,max_bytes=1024**2))
        return {'kind':kind,'status':result.status,'configs':[dict(c.params) for c in result.configs],
            'diagnostics':dict(result.diagnostics),'scope':'static constructive completions only; no CUDA/compiler/performance labels'}
    raise ValueError('query kind is values/validate_partial/complete')


def resolve_action(action,hardware,operator,shape,space,history,*,rng,allow_region_fallback=False):
    seen=set(history.attempted_config_ids)|{r.config.id for r in history}
    trace={'original_action':action,'authority':'unchanged platform space','platform_completed_fields':[],'region_revoked':False}
    if 'query' in action:return None,{'query_result':space_query(action['query'],hardware,operator,shape,space),**trace}
    if 'config' in action:
        config=space.normalize(action['config']);valid=space.validate(config,hardware,operator,shape)
        if not valid.valid:raise ValueError('illegal configuration: '+';'.join(valid.reasons))
        if config.id in seen:raise ValueError('configuration already revealed; an incumbent recommendation is not a new search observation')
    else:
        partial=action.get('partial')
        if 'region' in action:
            region=action['region']
            if not isinstance(region,dict) or any(not isinstance(v,list) or not v for v in region.values()):raise ValueError('finite nonempty advisory region required')
            partial={name:rng.choice(values) for name,values in region.items()}
        if not isinstance(partial,dict):raise ValueError('partial assignment must be an object')
        result=space.complete(partial,hardware,operator,shape,count=1,rng=rng,exclude_ids=seen,
            budget=SpaceBuildBudget(max_nodes=10000,max_wall_time_s=.5,max_configs=1))
        trace['construction']=dict(result.diagnostics);trace['construction_status']=result.status
        if not result.configs and 'region' in action and allow_region_fallback:
            trace['region_revoked']=True
            result=space.complete({},hardware,operator,shape,count=1,rng=rng,exclude_ids=seen,
                budget=SpaceBuildBudget(max_nodes=10000,max_wall_time_s=.5,max_configs=1))
        if not result.configs:raise ValueError('no completion within model action/budget: '+result.status+'; not proof the whole space is empty')
        config=result.configs[0];trace['platform_completed_fields']=sorted(set(config.params)-set(partial))
    trace['executed_config']=dict(config.params)
    trace['representation_normalized']=canonical_json(action.get('config',{}))!=canonical_json(config.params) if 'config' in action else True
    return config,trace
