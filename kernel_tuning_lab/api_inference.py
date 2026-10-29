"""Official Chat Completions / Responses wire protocols, optional and CPU-safe.

Building/testing a client grants no paid-call or data-egress authorization. No
credential is inspected until an explicitly authorized live invocation. Injected
transports are always labelled protocol fixtures, never real API evidence.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import hashlib
import ipaddress
import json
import math
import os
import socket
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from .models import canonical_json


class APIAuthorizationRequired(RuntimeError):pass
class ProviderResponseError(RuntimeError):pass


def require_public_service_location(endpoint,*,resolver=socket.getaddrinfo):
    """HTTP alone is no GPU-location proof. Private/local services need the
    separately verified worker protocol, which this public client does not own.
    """
    host=urllib.parse.urlsplit(endpoint).hostname
    try:addresses=[ipaddress.ip_address(host)]
    except ValueError:
        if host=='localhost' or host.endswith('.localhost'):raise APIAuthorizationRequired('local inference service requires a verified cooperative worker')
        try:addresses=[ipaddress.ip_address(row[4][0]) for row in resolver(host,443,type=socket.SOCK_STREAM)]
        except OSError as exc:raise APIAuthorizationRequired('cannot verify public inference-service placement') from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise APIAuthorizationRequired('private/local endpoint is not verified separate inference; use the cooperative local provider')


@dataclass(frozen=True)
class APIProvider:
    provider: str
    endpoint: str
    model: str
    protocol: str='chat_completions'
    structured_output: str='json_object'
    supported_sampling_parameters: tuple[str,...]=()
    inference_location: str='remote_public_api'
    timeout_s: float=30
    max_retries: int=1
    max_response_bytes: int=2*1024**2
    key_env: str='OPENAI_API_KEY'
    input_usd_per_million: float | None=None
    output_usd_per_million: float | None=None

    def __post_init__(self):
        url=urllib.parse.urlsplit(self.endpoint)
        if url.scheme!='https' or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('provider endpoint must be a credential-free HTTPS URL')
        if self.protocol not in ('chat_completions','responses'):raise ValueError('unsupported official wire protocol')
        if not self.model or not self.provider:raise ValueError('explicit provider and model are required')
        if self.structured_output not in ('none','json_object','json_schema'):raise ValueError('declare supported output format')
        if self.inference_location not in ('remote_public_api','cpu_service','separate_device','single_card_resident'):
            raise ValueError('explicit inference placement required')
        if not 0<self.timeout_s<=300 or type(self.max_retries) is not int or not 0<=self.max_retries<=3:
            raise ValueError('finite API request/retry limits required')
        if not 1<=self.max_response_bytes<=16*1024**2:raise ValueError('bounded provider response required')
        if any(p is not None and (not math.isfinite(p) or p<0) for p in (self.input_usd_per_million,self.output_usd_per_million)):
            raise ValueError('configured token prices cannot be negative')


@dataclass(frozen=True)
class APIAuthorization:
    allow_paid_requests: bool=False
    allow_task_metadata_egress: bool=False
    allowed_endpoint: str | None=None
    allowed_model: str | None=None
    max_requests: int=1
    authorization_ref: str | None=None

    def __post_init__(self):
        if type(self.allow_paid_requests) is not bool or type(self.allow_task_metadata_egress) is not bool:
            raise ValueError('authorization flags must be explicit JSON booleans, not strings')
        if type(self.max_requests) is not int or not 1<=self.max_requests<=10000:
            raise ValueError('explicit finite integer request allowance required')

    def check(self,spec,count):
        if (not self.allow_paid_requests or not self.allow_task_metadata_egress
                or self.allowed_endpoint!=spec.endpoint or self.allowed_model!=spec.model
                or not self.authorization_ref or count>=self.max_requests):
            raise APIAuthorizationRequired('explicit endpoint/model, paid-call, data-egress and request-count authorization required')


def outbound_task(task):
    """Allowlist only task metadata/revealed data; never serialize a whole spec."""
    allowed={'schema_version','feature_version','route','operator','hardware','shape','space','history',
        'attempted_config_ids','remaining_budget','response_contract','allowed_responses','notice',
        'public_search_interface','development_feedback','previous_program','space_query_result'}
    if set(task)-allowed:raise ValueError('unapproved outbound task fields: '+','.join(sorted(set(task)-allowed)))
    forbidden={'api_key','authorization','credentials','private_source','source_code','weights','test_labels',
        'device_uuid','hostname','private_path','training_database','output_dir'}
    def visit(value):
        if isinstance(value,dict):
            if set(k.lower() for k in value)&forbidden:raise ValueError('sensitive field in model task')
            for child in value.values():visit(child)
        elif isinstance(value,(tuple,list)):
            for child in value:visit(child)
        elif isinstance(value,str) and any(x in value for x in ('/export/home/','/mnt/cfs/','-----BEGIN PRIVATE KEY-----')):
            raise ValueError('private machine/source material is not authorized for API egress')
    visit(task)
    feedback=task.get('development_feedback')
    if feedback and (feedback.get('test_labels_read') is not False or any(r.get('split')=='test' for r in feedback.get('observations',[]))):
        raise ValueError('C development cannot send held-out labels')
    return task


def request_body(spec,messages,*,sampling=None,schema=None,max_output_tokens=1024):
    if type(max_output_tokens) is not int or not 1<=max_output_tokens<=32768:raise ValueError('bounded integer output tokens required')
    sampling=dict(sampling or {})
    if set(sampling)-set(spec.supported_sampling_parameters):raise ValueError('sampling parameter unsupported by provider capabilities')
    if spec.structured_output=='json_schema' and not isinstance(schema,dict):raise ValueError('explicit supported JSON schema required')
    body={'model':spec.model,**sampling}
    if spec.protocol=='chat_completions':
        body.update(messages=messages,max_completion_tokens=max_output_tokens)
        if spec.structured_output=='json_object':body['response_format']={'type':'json_object'}
        if spec.structured_output=='json_schema':body['response_format']={'type':'json_schema','json_schema':{'name':'ktlab_output','strict':True,'schema':schema}}
    else:
        body.update(input=messages,max_output_tokens=max_output_tokens,store=False)
        if spec.structured_output=='json_object':body['text']={'format':{'type':'json_object'}}
        if spec.structured_output=='json_schema':body['text']={'format':{'type':'json_schema','name':'ktlab_output','strict':True,'schema':schema}}
    return body


def parse_response(spec,value):
    if spec.protocol=='chat_completions':
        choices=value.get('choices',[])
        if len(choices)!=1:raise ProviderResponseError('one provider completion required')
        choice=choices[0];message=choice.get('message',{})
        if message.get('refusal'):raise ProviderResponseError('provider refused the request')
        if choice.get('finish_reason')!='stop':raise ProviderResponseError('provider output incomplete or unsupported tool response')
        text=message.get('content');usage=value.get('usage',{})
        normalized={'input_tokens':usage.get('prompt_tokens'),'output_tokens':usage.get('completion_tokens')}
    else:
        if value.get('status')!='completed':raise ProviderResponseError('provider response is not completed')
        parts=[]
        for item in value.get('output',[]):
            if item.get('type')!='message':continue
            for content in item.get('content',[]):
                if content.get('type')=='refusal':raise ProviderResponseError('provider refused the request')
                if content.get('type')=='output_text':parts.append(content['text'])
        text=''.join(parts);usage=value.get('usage',{})
        normalized={'input_tokens':usage.get('input_tokens'),'output_tokens':usage.get('output_tokens')}
    if not isinstance(text,str) or not text:raise ProviderResponseError('provider returned no text output')
    return {'text':text,'response_id':value.get('id'),'returned_model':value.get('model'),
        'usage':normalized,'raw_usage':usage,'remote_determinism':'not guaranteed; provider version/weights may be unavailable'}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ProviderResponseError('provider redirects are not followed with credentials')


class PublicModelClient:
    def __init__(self,spec,*,authorization=None,log_path=None,transport=None,resource_token=None):
        self.spec=spec;self.authorization=authorization or APIAuthorization();self.log_path=Path(log_path) if log_path else None
        self.transport=transport;self.requests=0;self.resource_token=resource_token
    def invoke(self,task,*,system,route='B',sampling=None,schema=None,max_output_tokens=1024):
        task=outbound_task({**task,'route':route})
        outbound_task({'notice':system})
        messages=[{'role':'system','content':system},{'role':'user','content':canonical_json(task)}]
        body=request_body(self.spec,messages,sampling=sampling,schema=schema,max_output_tokens=max_output_tokens)
        if self.transport is None:
            self.authorization.check(self.spec,self.requests)
            host=urllib.parse.urlsplit(self.spec.endpoint).hostname
            if host in ('localhost','127.0.0.1','::1') and self.spec.inference_location=='remote_public_api':
                raise APIAuthorizationRequired('HTTP transport does not prove inference uses a separate GPU; local placement/quiescence required')
            if self.spec.inference_location!='remote_public_api':
                # This client cannot synchronize an external service's CUDA
                # context. A string declaration must never whitelist its PID.
                raise APIAuthorizationRequired('external local/private service needs a verified worker placement/quiescence protocol; use the cooperative in-process local provider meanwhile')
            require_public_service_location(self.spec.endpoint)
            secret=os.environ.get(self.spec.key_env)
            if not secret:raise APIAuthorizationRequired('authorized credential environment variable is unavailable')
        else:secret=None
        started=time.perf_counter();events=[];result=None;error=None
        def redact(v):
            if isinstance(v,str):return v.replace(secret,'[REDACTED]') if secret else v
            if isinstance(v,dict):return {k:redact(x) for k,x in v.items()}
            if isinstance(v,list):return [redact(x) for x in v]
            return v
        try:
            for attempt in range(self.spec.max_retries+1):
                if self.transport is None:self.authorization.check(self.spec,self.requests)
                self.requests+=1;before=time.perf_counter();request_id=None;value=None
                try:
                    payload=canonical_json(body).encode()
                    if self.transport is not None:value,headers=self.transport(payload)
                    else:
                        request=urllib.request.Request(self.spec.endpoint,data=payload,
                            headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'},method='POST')
                        with urllib.request.build_opener(_NoRedirect()).open(request,timeout=self.spec.timeout_s) as response:
                            blob=response.read(self.spec.max_response_bytes+1);headers=dict(response.headers)
                        if len(blob)>self.spec.max_response_bytes:raise ProviderResponseError('response exceeds byte allowance')
                        value=json.loads(blob)
                    request_id=next((v for k,v in headers.items() if k.lower()=='x-request-id'),None)
                    result=parse_response(self.spec,value);result['request_id']=request_id
                    events.append({'attempt':attempt,'network_and_service_wall_s':time.perf_counter()-before,
                        'request_id':request_id,'response':redact(value),'status':'completed'})
                    break
                except urllib.error.HTTPError as exc:
                    events.append({'attempt':attempt,'status':'http_error','http_status':exc.code,
                        'network_and_service_wall_s':time.perf_counter()-before,'billing':'unknown unless provider usage supplied'})
                    if exc.code not in (429,500,502,503,504) or attempt==self.spec.max_retries:raise ProviderResponseError('provider HTTP'+str(exc.code)) from None
                    time.sleep(min(2.,.25*2**attempt))
                except Exception as exc:
                    events.append({'attempt':attempt,'status':'error','error':redact(type(exc).__name__+': '+str(exc)),
                        'request_id':request_id,'response':redact(value),
                        'usage_when_available':value.get('usage') if isinstance(value,dict) else None,
                        'network_and_service_wall_s':time.perf_counter()-before,'billing':'unknown; ambiguous timeout is not retried'})
                    raise
            usage=result['usage'];price=None
            if (self.spec.input_usd_per_million is not None and self.spec.output_usd_per_million is not None
                    and all(type(usage[x]) is int for x in ('input_tokens','output_tokens'))):
                price=(usage['input_tokens']*self.spec.input_usd_per_million+usage['output_tokens']*self.spec.output_usd_per_million)/1e6
            result.update(estimated_usd=price,price_basis='successful response only; caller-configured estimate, not provider invoice; failed/ambiguous request charges unknown',
                total_wall_s=time.perf_counter()-started,evidence_level='protocol_fixture_not_real_API' if self.transport else 'authorized_remote_API_response')
            return result
        except Exception as exc:error=redact(type(exc).__name__+': '+str(exc));raise
        finally:
            event={'provider':self.spec.provider,'endpoint':self.spec.endpoint,'requested_model':self.spec.model,
                'protocol':self.spec.protocol,'request':body,'request_sha256':hashlib.sha256(canonical_json(body).encode()).hexdigest(),
                'events':events,'result':result,'error':error,'total_wall_s':time.perf_counter()-started,
                'evidence_level':'protocol_fixture_not_real_API' if self.transport else 'authorized_remote_API_call',
                'authorization_ref':self.authorization.authorization_ref}
            if self.log_path:
                self.log_path.parent.mkdir(parents=True,exist_ok=True)
                with self.log_path.open('a') as stream:stream.write(canonical_json(redact(event))+'\n')
