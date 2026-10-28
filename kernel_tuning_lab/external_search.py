"""Model/process/endpoint proposals through the normal SDK; no Agent framework.

Commands and model artifacts are trusted user code/data, not a malicious-code
sandbox. Examples enforce CPU inference; HTTP device placement is a declared
external contract and is never claimed to be verified by this client.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.request

from .sdk import SearchAlgorithmDefinition,KernelConfig,SpaceBuildBudget
from .models import canonical_json
from .datasets import case_id


def _sha(data):return hashlib.sha256(data).hexdigest()


def describe_task(hardware,operator,shape,space,history,budget):
    parameters={}
    for name,spec in space.parameter_specs.items():
        domain=space.domains[name]
        parameters[name]={'kind':spec.kind,'size':len(domain),'default':spec.default,
            'active_when':dict(spec.active_when),
            'values':list(domain) if len(domain)<=32 else None,
            'numeric_bounds':[min(domain),max(domain)] if spec.kind in ('integer','real') else None}
    return {'schema_version':1,'feature_version':'shape-space-v1','operator':{'kind':operator.kind,
        'implementation':operator.implementation,'dtype':operator.dtype,'parameters':dict(operator.parameters)},
        'hardware':{'name':hardware.name,'architecture':hardware.architecture,'sm_count':hardware.sm_count,
                    'l2_bytes':hardware.l2_bytes},'shape':dict(shape.dims),
        'space':{'fingerprint':space.fingerprint,'declared_cardinality':space.cardinality,
                 'parameters':parameters,'structured_rules':space.capabilities['partial_rules']},
        'history':[{'config':dict(r.config.params),'status':r.status.value,
            'verified':r.correctness_verified,'latency_ms':r.median_ms if r.correctness_verified else None} for r in history],
        'attempted_config_ids':sorted(history.attempted_config_ids),
        'remaining_budget':{'observations':budget.max_evaluations,'wall_s':budget.max_wall_time_s},
        'allowed_responses':['config','partial','region'],'notice':'No oracle, filesystem path, frozen labels, or predictions-as-measurements are provided.'}


class ExternalSearch:
    name='external_proposal'
    def __init__(self,seed=0,*,model_artifact=None,model_sha256=None,command=None,endpoint=None,
                 model_id=None,provider_kind='external_declared',timeout_s=10,max_retries=2,max_response_bytes=65536):
        if sum(value is not None for value in (model_artifact,command,endpoint))!=1:
            raise ValueError('exactly one model artifact, process command, or endpoint is required')
        if timeout_s<=0 or type(max_retries) is not int or not 0<=max_retries<=4:raise ValueError('finite bounded external calls required')
        self.seed=seed;self.rng=random.Random(seed);self.command=command;self.endpoint=endpoint
        self.timeout_s=timeout_s;self.max_retries=max_retries;self.max_response_bytes=max_response_bytes
        self.output=None;self.card=None
        self.diagnostics={'model_id':model_id,'calls':0,'retries':0,'inference_wall_ms':0.0,
            'invalid_responses':0,'region_revocations':0,'feedback_records':0,
            'transport':'artifact' if model_artifact else 'process' if command else 'HTTP',
            'model_execution':'CPU reference policy' if model_artifact else 'process CUDA hidden; trusted code, not sandbox' if command else 'external device placement declared by caller, unverified',
            'evidence_level':'unverified_external_provider' if not model_artifact else 'interface_fixture_not_trained'}
        if provider_kind=='contract_fixture' and not model_artifact:self.diagnostics['evidence_level']='protocol_fixture_not_a_real_model'
        if model_artifact:
            path=Path(model_artifact)
            if path.stat().st_size>1048576:raise ValueError('reference JSON artifact is limited to1MiB; use an external process for larger models')
            blob=path.read_bytes()
            if _sha(blob)!=model_sha256:raise ValueError('model artifact differs from frozen SHA256')
            self.card=json.loads(blob)
            if self.card.get('kind')!='interface_linear_policy' or self.card.get('feature_version')!='shape-space-v1':
                raise ValueError('unsupported reference model/feature version')
            self.diagnostics.update(model_id=self.card['model_id'],model_sha256=model_sha256,
                evidence_level=self.card.get('evidence_level','unknown_training_provenance'),
                training_record_ids=self.card.get('training_record_ids',[]))
        if command:
            if not isinstance(command,list) or not command or any(not isinstance(v,str) for v in command):raise ValueError('process command is an argv list; no shell interpolation')
            self.diagnostics['command_sha256']=_sha(canonical_json(command).encode())
            files=[Path(v) for v in command if Path(v).is_file()]
            self.diagnostics['process_code_sha256']={f.name:_sha(f.read_bytes()) for f in files if f.stat().st_size<1048576}
            self.command=[sys.executable if command[0] in ('python','python3') else command[0],*command[1:]]

    @property
    def runtime_identity(self):
        return {key:value for key,value in self.diagnostics.items() if key in
            ('model_id','model_sha256','command_sha256','process_code_sha256','transport','evidence_level','training_record_ids')}

    def bind_trial(self,*,output_dir,identity):self.output=Path(output_dir)

    def observe(self,record,*,cache_hit=False):self.diagnostics['feedback_records']+=1

    def _artifact_action(self,task,space):
        shape=task['shape'];features={'bias':1.,'log_M':math.log2(max(1,shape.get('M',1))),
            'log_N':math.log2(max(1,shape.get('N',1))),'log_K':math.log2(max(1,shape.get('K',1))),
            'observations':len(task['history'])}
        weights=self.card.get('parameter_weights',{});matches=[k for k in weights if k in space.domains]
        names=matches or list(space.domains)[:1];partial={}
        for name in names:
            coefficients=weights.get(name,self.card.get('default_weights',{'bias':0}))
            value=sum(float(coefficients.get(key,0))*number for key,number in features.items())
            domain=space.domains[name];index=int((math.tanh(value)+1)/2*len(domain))
            # The observed-count feature provides sequential actions; other
            # fields are completed by the authoritative dependency constructor.
            partial[name]=domain[min(len(domain)-1,index)]
        return {'partial':partial,'rationale':'reference artifact policy, not a trained performance predictor'}

    def _call(self,task,space):
        request=canonical_json(task).encode();start=time.perf_counter();response=None;error=None
        try:
            if self.card is not None:response=self._artifact_action(task,space)
            elif self.command:
                result=subprocess.run(self.command,input=request,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                    timeout=self.timeout_s,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
                if len(result.stdout)>self.max_response_bytes:raise ValueError('external response too large')
                response=json.loads(result.stdout)
            else:
                req=urllib.request.Request(self.endpoint,data=request,headers={'Content-Type':'application/json'},method='POST')
                with urllib.request.urlopen(req,timeout=self.timeout_s) as stream:blob=stream.read(self.max_response_bytes+1)
                if len(blob)>self.max_response_bytes:raise ValueError('external response too large')
                response=json.loads(blob)
            return response
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc);raise
        finally:
            elapsed=(time.perf_counter()-start)*1000;self.diagnostics['calls']+=1;self.diagnostics['inference_wall_ms']+=elapsed
            event={'request_sha256':_sha(request),'response':response,'response_sha256':_sha(canonical_json(response).encode()) if response is not None else None,
                'inference_wall_ms':elapsed,'error':error,'model_identity':self.runtime_identity}
            if self.output:
                with (self.output/'external-calls.jsonl').open('a') as out:out.write(canonical_json(event)+'\n')

    def _executed(self,action,config,*,revoked=False):
        self.diagnostics['last_executed_config']=dict(config.params)
        if self.output:
            with (self.output/'external-decisions.jsonl').open('a') as out:
                out.write(canonical_json({'original_action':action,'executed_config':dict(config.params),
                    'preference_revoked':revoked,'authority':'unchanged platform space; declared inactive normalization only'})+'\n')
        return [config]

    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0):return []
        if self.card and case_id(hardware,operator,shape) in self.card.get('training_case_ids',[]):
            raise ValueError('model training manifest overlaps the target evaluation case')
        task=describe_task(hardware,operator,shape,space,history,budget)
        seen={r.config.id for r in history}|set(history.attempted_config_ids)
        for attempt in range(self.max_retries+1):
            try:
                action=self._call(task,space)
                if not isinstance(action,dict) or set(action)-{'config','partial','region','rationale','prediction'}:
                    raise ValueError('unexpected decision schema; code and file requests are not accepted')
                selectors=set(action)&{'config','partial','region'}
                if len(selectors)!=1:raise ValueError('one config, partial, or advisory region required')
                if 'config' in action:
                    config=space.normalize(action['config']);valid=space.validate(config,hardware,operator,shape)
                    if not valid.valid or config.id in seen:raise ValueError('invalid/repeated complete configuration')
                    return self._executed(action,config)
                region=action.get('region');values=action.get('partial')
                if region is not None:
                    if not isinstance(region,dict) or any(not isinstance(v,list) or not v for v in region.values()):raise ValueError('region requires finite nonempty value lists')
                    values={key:self.rng.choice(choices) for key,choices in region.items()}
                if not isinstance(values,dict):raise ValueError('partial assignment must be an object')
                generated=space.complete(values,hardware,operator,shape,count=1,rng=self.rng,exclude_ids=seen,
                    budget=SpaceBuildBudget(max_nodes=10_000,max_wall_time_s=.5,max_configs=1))
                if generated.configs:return self._executed(action,generated.configs[0])
                # A region is an exploration preference, never authority. Its
                # revocation and global constructive exploration are explicit.
                self.diagnostics['region_revocations']+=1
                task['rejection_feedback']={'kind':'no_new_feasible_completion','status':generated.status,'preference_revoked':True}
                if attempt==self.max_retries:
                    global_result=space.complete({},hardware,operator,shape,count=1,rng=self.rng,exclude_ids=seen,
                        budget=SpaceBuildBudget(max_nodes=10_000,max_wall_time_s=.5,max_configs=1))
                    self.diagnostics['last_action']='recorded preference revocation; structured exploration outside requested region'
                    return self._executed(action,global_result.configs[0],revoked=True) if global_result.configs else []
            except Exception as exc:
                self.diagnostics['invalid_responses']+=1;task['rejection_feedback']={'kind':'invalid_external_response','message':str(exc)}
            self.diagnostics['retries']+=int(attempt<self.max_retries)
        self.last_proposal_status='external_proposal_budget_exhausted';return []


ASSETS=(SearchAlgorithmDefinition('external/model_or_process','1.0.0',ExternalSearch,
    metadata={'scope':'reference artifact / local process / HTTP structured proposals; no model training or Agent framework',
              'code_trust':'reviewed trusted Python; not a malicious-code sandbox'}),)
