"""Cooperative in-process inference/measurement leases and resource identity.

This gate coordinates registered trusted producers; it is not a sandbox for
arbitrary CUDA code. External same-card services are refused until their own
worker provides an authenticated stop-submission/quiescence protocol. A sync
in the benchmark process cannot prove another process is quiescent.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import os
import hashlib
from pathlib import Path
import threading
import time
import uuid

from .models import stable_id


MODES=('remote_api','separate_device','single_card_resident','stage_offload')


class ResourceIsolationError(RuntimeError):pass
class ResourceContextLimit(RuntimeError):pass


@dataclass
class Participant:
    registration_id: str
    logical_id: str
    mode: str
    device_uuid: str | None
    identity: dict
    synchronize: object
    owner_pid: int
    resident_bytes: int=0


class ProcessCoordinator:
    def __init__(self):
        self._lock=threading.RLock();self.participants={};self.phase='idle';self.epoch=0;self.last_proof=None
    def register(self,logical_id,*,mode,device_uuid,identity,synchronize,owner_pid=None,resident_bytes=0):
        if mode not in MODES:raise ValueError('explicit inference deployment mode required')
        if mode!='remote_api' and (not isinstance(device_uuid,str) or not device_uuid):
            raise ValueError('physical inference device identity required')
        owner_pid=os.getpid() if owner_pid is None else owner_pid
        if owner_pid!=os.getpid():
            raise ResourceIsolationError('external same-card process needs its own verified cooperative worker; caller-side synchronize is insufficient')
        if not callable(synchronize):raise ValueError('participant-owned completion synchronization required')
        if type(resident_bytes) is not int or resident_bytes<0:raise ValueError('known nonnegative resident byte budget required')
        with self._lock:
            if self.phase!='idle':raise ResourceIsolationError('register outside active GPU phases')
            token=uuid.uuid4().hex
            self.participants[token]=Participant(token,logical_id,mode,device_uuid,dict(identity),synchronize,owner_pid,resident_bytes)
            return token
    def unregister(self,token):
        with self._lock:
            if self.phase!='idle':raise ResourceIsolationError('cannot unregister an active GPU producer')
            self.participants.pop(token,None)
    def resource_identity(self):
        # PIDs, nonce and dynamic allocator statistics are proof/cost metadata,
        # not stable measurement-cache identities.
        items=[{'logical_id':p.logical_id,'mode':p.mode,'device_uuid':p.device_uuid,
            'model_identity':p.identity,'resident_bytes':p.resident_bytes}
            for p in self.participants.values()]
        items.sort(key=lambda p:(p['logical_id'],p['mode'],p['device_uuid'] or ''))
        source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return {'protocol':'cooperative_serial_v1','participants':items,'context_id':stable_id(items),'gate_source_sha256':source_sha} if items else {'protocol':'exclusive_no_registered_inference_v1','gate_source_sha256':source_sha}
    @contextmanager
    def inference_window(self,token):
        with self._lock:
            if token not in self.participants:raise ResourceIsolationError('unregistered inference producer')
            if self.phase!='idle':raise ResourceIsolationError('inference cannot submit while another GPU phase holds the lease')
            self.phase='inference';self.epoch+=1
            try:yield
            finally:
                try:self.participants[token].synchronize()
                finally:self.phase='idle'
    @contextmanager
    def measurement_window(self,hardware):
        with self._lock:
            if self.phase!='idle':raise ResourceIsolationError('measurement cannot overlap registered inference')
            device=hardware.metadata.get('device_uuid','').lower().removeprefix('gpu-')
            local=[]
            for p in self.participants.values():
                if p.owner_pid!=os.getpid():raise ResourceIsolationError('participant owner process changed')
                same=(p.device_uuid or '').lower().removeprefix('gpu-')==device
                if p.mode=='separate_device' and same:raise ResourceIsolationError('declared separate inference device is the measurement device')
                if p.mode in ('single_card_resident','stage_offload') and not same:
                    raise ResourceIsolationError('same-card mode is bound to another physical device')
                if same:local.append(p)
            self.phase='measurement';start=time.monotonic();self.epoch+=1
            try:
                # The lease stops new submissions; synchronization executes in
                # each registered producer's actual process/context.
                for p in local:p.synchronize()
                proof={'protocol':'cooperative_serial_v1','epoch':self.epoch,'owner_pid':os.getpid(),
                    'registered_local_producers':[p.registration_id for p in local],
                    'new_submissions_blocked':True,'producer_completion_observed':True,
                    'resident_weights_may_remain':True,'resource_identity':self.resource_identity()}
                self.last_proof=proof;yield proof
            finally:
                try:
                    for p in local:p.synchronize()
                finally:
                    if self.last_proof:self.last_proof['whole_measurement_phase_wall_s']=time.monotonic()-start
                    self.phase='idle'
    def check_known_allocation(self,hardware,bytes_requested,current_declared=0):
        reserved=sum(p.resident_bytes for p in self.participants.values() if p.mode=='single_card_resident')
        if hardware.memory_bytes and reserved+current_declared+bytes_requested>hardware.memory_bytes:
            raise ResourceContextLimit('known allocation exceeds this resident-model resource context; not GPU-architecture unsupported')


COORDINATOR=ProcessCoordinator()
