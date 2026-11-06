"""Existing local checkpoints behind the same B/C task protocol as the API.

No downloads/training; lazy optional dependencies. A resident client loads once
per trial/client lifetime, uses owned cooperative leases, and closes explicitly
at a phase boundary. This does not authorize an additional inference GPU.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
from .api_inference import outbound_task
from .models import canonical_json


class LocalModelClient:
    def __init__(self,*,model_manifest,manifest_sha256,resource_mode='single_card_resident',seed=0,log_path=None,inference_device_uuid=None):
        blob=Path(model_manifest).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=manifest_sha256:raise ValueError('local model manifest identity mismatch')
        self.manifest=json.loads(blob)
        if not self.manifest.get('full_weight_hashes'):raise ValueError('existing model needs full weight identities')
        if resource_mode not in ('single_card_resident','stage_offload','separate_device'):raise ValueError('unknown cooperative local resource mode')
        if (resource_mode=='separate_device')!=(inference_device_uuid is not None):
            raise ValueError('separate_device requires explicit inference_device_uuid; same-card modes do not accept it')
        self.inference_device_uuid=inference_device_uuid
        self.sha=manifest_sha256;self.mode=resource_mode;self.seed=seed;self.session=None;self.calls=0
        self.log_path=Path(log_path) if log_path else None
    @property
    def runtime_identity(self):return {'model_id':self.manifest['model_id'],'manifest_sha256':self.sha,'resource_mode':self.mode,
        'inference_device_uuid':self.inference_device_uuid,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'downloads':False,'training':False}
    def invoke(self,task,*,system,route='B',schema=None,max_output_tokens=1024,sampling=None):
        if route not in ('B','C'):raise ValueError('a dedicated trained A model uses its own external process; no implicit training')
        if type(max_output_tokens) is not int or not 1<=max_output_tokens<=8192:raise ValueError('finite local output cap1..8192 required')
        if sampling:raise ValueError('this local provider declares greedy only; no provider capability guessing')
        task=outbound_task({**task,'route':route});outbound_task({'notice':system})
        messages=[{'role':'system','content':system},{'role':'user','content':canonical_json(task)}]
        self.calls+=1;started=time.perf_counter();row={'messages':messages,'route':route,'model_identity':self.runtime_identity,
            'evidence_level':'local_provider_attempt; model_execution_unconfirmed'}
        try:
            if self.session is None:
                from .live_llm import _Session
                self.session=_Session(self.manifest,self.mode,self.sha,**({'inference_device_uuid':self.inference_device_uuid} if self.inference_device_uuid else {}))
            row.update(self.session.infer(messages,self.seed+self.calls-1,max_output_tokens))
            row.update(evidence_level='real_local_model_inference',
                schema_enforcement='JSON schema supplied as a contract only; final action/code checks remain mandatory',response_id=None,
                estimated_usd=None,price_basis='local inference costs/time only; no cloud price fabricated')
            return row
        except Exception as exc:
            row.update(error=type(exc).__name__+': '+str(exc),candidate_measurement_admitted=False);raise
        finally:
            row['client_total_wall_s']=time.perf_counter()-started
            if self.log_path:
                self.log_path.parent.mkdir(parents=True,exist_ok=True)
                with self.log_path.open('a') as stream:stream.write(canonical_json(row)+'\n')
    def close(self):
        if self.session is not None:self.session.close();self.session=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
