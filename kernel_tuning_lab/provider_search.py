"""B/A decisions through public API or an existing CPU process, same actions.

Providers propose. Existing Runner owns compilation/correctness/timing/budgets.
No model training, automatic credential use, or malicious-code sandbox claim.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

from .sdk import SearchAlgorithmDefinition
from .api_inference import APIProvider,APIAuthorization,PublicModelClient
from .external_search import describe_task
from .ai_actions import parse_action,resolve_action
from .models import canonical_json
from .process_environment import reviewed_environment


def cpu_provider_environment():
    # Trusted code still receives no inherited credentials or private config.
    # This is hygiene, not a sandbox: users review commands before running them.
    return reviewed_environment(cpu_only=True)


class ProviderSearch:
    def __init__(self,seed=0,*,provider,route='B',authorization=None,max_calls_per_proposal=4,allow_region_fallback=False):
        if route not in ('A','B'):raise ValueError('C produces reviewed/frozen programs via artifact workflow, not a decision-search alias')
        if type(max_calls_per_proposal) is not int or not 1<=max_calls_per_proposal<=8:raise ValueError('bounded query/repair loop required')
        self.provider=dict(provider);self.route=route;self.rng=random.Random(seed);self.seed=seed;self.limit=max_calls_per_proposal;self.fallback=allow_region_fallback;self.output=None
        self.diagnostics={'route':route,'provider_kind':provider['kind'],'calls':0,'queries':0,'repairs':0,'model_wall_s':0.,'platform_completion_count':0}
        if provider['kind']=='public_api':
            self.client=PublicModelClient(APIProvider(**provider['configuration']),authorization=APIAuthorization(**(authorization or {})))
        elif provider['kind']=='local_model':
            if route!='B':raise ValueError('local LLM decision provider serves B; dedicated trained A uses a reviewed model process')
            from .local_inference import LocalModelClient
            self.client=LocalModelClient(**provider['configuration'],seed=seed)
        elif provider['kind']=='cpu_process':
            command=provider.get('command')
            if not isinstance(command,list) or not command or any(not isinstance(x,str) for x in command):raise ValueError('trusted CPU provider command must be argv, not shell')
            self.command=[sys.executable if command[0] in ('python','python3') else command[0],*command[1:]]
            self.timeout=provider.get('timeout_s',30)
            if not 0<self.timeout<=300:raise ValueError('finite CPU provider timeout required')
        else:raise ValueError('provider kind is public_api, local_model, or cpu_process')
    @property
    def runtime_identity(self):return {'route':self.route,'provider_sha256':hashlib.sha256(canonical_json(self.provider).encode()).hexdigest(),
        'provider_kind':self.provider['kind'],'feature_version':'shape-space-v1+bounded-queries','model_id':self.provider.get('model_id'),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'authorization':'never part of model prompt; secret values never in options'}
    def bind_trial(self,*,output_dir,identity):
        self.output=Path(output_dir)
        if hasattr(self,'client'):self.client.log_path=self.output/'model-calls.jsonl'
    def close(self):
        if hasattr(self,'client') and callable(getattr(self.client,'close',None)):self.client.close()
    def _request(self,task):
        if hasattr(self,'client'):
            return self.client.invoke(task,route=self.route,schema=self.provider.get('output_schema'),
                max_output_tokens=self.provider.get('max_output_tokens',1024),
                system='Return exactly ONE JSON OBJECT. Its sole action key must be config, partial, region, or query, and its value must be an OBJECT, not an action name. For example {"partial":{"PARAMETER_NAME":SCALAR_VALUE_FROM_TASK}} proposes parameter values; {"action":"config"} is INVALID. Replace the placeholders with actual task parameter names and legal scalar values. Use only declared choices and revealed feedback. Never invent measured performance.')['text']
        with tempfile.TemporaryFile() as stdout,tempfile.TemporaryFile() as stderr:
            subprocess.run(self.command,input=canonical_json(task).encode(),stdout=stdout,stderr=stderr,
                timeout=self.timeout,check=True,env=cpu_provider_environment())
            stdout.seek(0);blob=stdout.read(1024**2+1)
        if len(blob)>1024**2:raise ValueError('CPU provider output exceeds1MiB')
        return blob.decode()
    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0):return []
        if self.output is None:
            # Synthetic SDK contract, explicitly not provider/model evidence.
            seen=set(history.attempted_config_ids)|{r.config.id for r in history}
            return list(space.complete({},hardware,operator,shape,count=min(batch_size,budget.max_evaluations),
                exclude_ids=seen,rng=self.rng).configs)
        task=describe_task(hardware,operator,shape,space,history,budget)
        task['response_contract']={'actions':['config','partial','region','query'],
            'envelope':'{"config":{parameter:scalar,...}} or {"partial":{parameter:scalar,...}}; never {"action":"config"}',
            'queries':{'values':'parameter/offset/limit<=32','validate_partial':'partial','complete':'partial/count<=32'},
            'notice':'partial/region completions are platform-selected and separately logged, not entirely model decisions'}
        for index in range(self.limit):
            began=time.perf_counter();self.diagnostics['calls']+=1;raw=None;event={'attempt':index}
            try:
                raw=self._request(task);action=parse_action(raw,space.domains)
                config,trace=resolve_action(action,hardware,operator,shape,space,history,rng=self.rng,allow_region_fallback=self.fallback)
                event.update(trace=trace,raw_response=raw)
                if config is None:
                    self.diagnostics['queries']+=1;task['space_query_result']=trace['query_result'];continue
                self.diagnostics['platform_completion_count']+=int(bool(trace['platform_completed_fields']))
                return [config]
            except Exception as exc:
                # Authorization is an execution boundary, not a model repair or
                # a configuration failure. Never retry it as a billable call.
                from .api_inference import APIAuthorizationRequired,ProviderResponseError
                import urllib.error
                if isinstance(exc,(APIAuthorizationRequired,ProviderResponseError,urllib.error.URLError,
                    TimeoutError,subprocess.TimeoutExpired,subprocess.CalledProcessError)):raise
                event.update(error=type(exc).__name__+': '+str(exc),raw_response=raw)
                task['space_query_result']={'kind':'static_action_error','message':str(exc)};self.diagnostics['repairs']+=1
            finally:
                event['wall_s']=time.perf_counter()-began;self.diagnostics['model_wall_s']+=event['wall_s']
                with (self.output/'provider-actions.jsonl').open('a') as stream:stream.write(canonical_json(event)+'\n')
        self.last_proposal_status='provider_query_or_repair_budget_exhausted';return []


ASSETS=(SearchAlgorithmDefinition('ai/standard_provider','1.0.0',ProviderSearch,
    source_files=('api_inference.py','ai_actions.py','external_search.py','local_inference.py','live_llm.py','local_llm.py','inference_resources.py'),
    metadata={'routes':'A external existing model or B LLM; API/local process are orthogonal provider deployments','training':'none','trust':'CPU processes are reviewed trusted code, not a sandbox'}),)
