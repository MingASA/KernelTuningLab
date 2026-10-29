"""Persistent callback bridge for the unchanged Kernel Tuner1.5 DE routine.

Native code suspends at a missing real observation. Its global RNG state is
suspended/restored at those synchronous boundaries, as in the former replay
adapter; native computation never runs during a Kernel evaluation. No GPU
runner or alternative differential-evolution implementation lives here.
"""
from __future__ import annotations
from copy import deepcopy
from functools import partial
import queue
import random
import threading
from types import FunctionType
import time


_RNG_LOCK=threading.RLock()
_STOP=object()


class _Closed(Exception):pass


class NativeDECallback:
    def __init__(self,adapter):
        self.adapter=adapter;self.outbox=queue.Queue();self.inbox=queue.Queue();self.native_rng=None
        self.thread=None;self.done=False;self.pending=None;self.stop=threading.Event()
    def advance(self,feedback=None):
        if self.done:return None
        import numpy as np
        with _RNG_LOCK:
            host=(random.getstate(),np.random.get_state())
            try:
                if self.native_rng is None:random.seed(self.adapter.seed);np.random.seed(self.adapter.seed)
                else:random.setstate(self.native_rng[0]);np.random.set_state(self.native_rng[1])
                if self.thread is None:
                    self.thread=threading.Thread(target=self._run,daemon=True,name='native-KT-DE-callback');self.thread.start()
                else:self.inbox.put(feedback)
                try:message=self.outbox.get(timeout=self.adapter.native_wall_s)
                except queue.Empty:
                    self.stop.set();self.inbox.put(_STOP)
                    self.thread.join(timeout=self.adapter.native_wall_s)
                    raise TimeoutError('native DE CPU callback deadline; no unbudgeted candidate substitution')
                self.native_rng=(random.getstate(),np.random.get_state())
            finally:random.setstate(host[0]);np.random.set_state(host[1])
        if message['kind']=='error':self.done=True;raise RuntimeError(message['error'])
        if message['kind']=='done':self.done=True;self.pending=None;return None
        self.pending=message['config'];return self.pending
    def close(self):
        if self.thread is None or self.done:return
        self.stop.set();self.advance(_STOP);self.thread.join(timeout=self.adapter.native_wall_s)
        if self.thread.is_alive():raise RuntimeError('native callback failed to stop; do not admit another GPU measurement')
    def _run(self):
        import numpy as np
        import scipy.stats.qmc as qmc
        from unittest.mock import patch
        from kernel_tuner.searchspace import Searchspace
        from kernel_tuner.interface import Options
        from kernel_tuner.strategies import diff_evo
        from kernel_tuner.strategies.common import CostFunc
        from kernel_tuner.util import TuningBudget,CompilationFailedConfig
        from .models import KernelConfig
        adapter=self.adapter;domains={k:list(v) for k,v in adapter.space.domains.items()};known=deepcopy(adapter.known)
        cache={str(i):dict(c.params) for i,c in enumerate(adapter.configs)};qmc_rng=np.random.default_rng(adapter.seed)
        bridge=self;populations=0
        class SeededSpace(Searchspace):
            def get_LHS_sample_indices(self,num_samples):
                if num_samples==self.size:return np.random.permutation(self.size).tolist()
                # Scope the upstream RNG compatibility hook to this CPU method,
                # never leave scipy patched while waiting for a GPU result.
                with patch.object(qmc,'LatinHypercube',partial(qmc.LatinHypercube,rng=qmc_rng)):
                    return super().get_LHS_sample_indices(num_samples)
        class RunnerBridge:
            def add_strategy_time(self,value):pass
            def run(self,vectors,options):
                nonlocal populations
                rows=[]
                for vector in vectors:
                    if bridge.stop.is_set():raise _Closed()
                    config=adapter.space.normalize({k:(v.item() if hasattr(v,'item') else v) for k,v in zip(domains,vector)})
                    validity=adapter.space.validate(config,*adapter.context)
                    if not validity.valid:raise ValueError('native repaired vector is outside the authoritative space')
                    record=known.get(config.id)
                    if record is None:
                        adapter._event('ask',config=dict(config.params),generations_completed=adapter.diagnostics['completed_generations'])
                        bridge.outbox.put({'kind':'ask','config':config})
                        feedback=bridge.inbox.get()
                        if feedback is _STOP or bridge.stop.is_set():raise _Closed()
                        if feedback.config.id!=config.id:raise ValueError('native DE feedback does not match its pending vector')
                        record=feedback;known[config.id]=record
                    else:adapter.diagnostics['internal_cached_evaluations']+=1
                    row=dict(config.params)
                    if record.correctness_verified:row['time']=record.median_ms
                    else:row['__error__']=CompilationFailedConfig()
                    rows.append(row)
                populations+=1;adapter.diagnostics['completed_generations']=max(0,populations-1)
                adapter.diagnostics['model_or_evolution_entered']=populations>1
                return rows
        original_repair=diff_evo.repair
        def trace_repair(vector,space):
            repaired=original_repair(vector,space);adapter.diagnostics['native_repair_calls']+=1
            if list(vector)!=list(repaired) and adapter.diagnostics['native_repair_calls']<=64:
                plain=lambda values:[v.item() if hasattr(v,'item') else v for v in values]
                adapter._event('native_repair',original_proposal=plain(vector),executed_proposal=plain(repaired),feedback_object='executed native vector')
            return repaired
        try:
            native=SeededSpace(None,None,None,from_cache={'tune_params':domains,'cache':cache},neighbor_method='Hamming-adjacent')
            options=Options(tune_params=domains,objective=['time'],objective_higher_is_better=[False],
                strategy_options=Options(constraint_aware=True),budget=TuningBudget())
            cost=CostFunc(native,options,RunnerBridge())
            # The upstream numeric routine and helpers stay unchanged. Only a
            # private repair trace binding replaces the old global monkeypatch.
            scope=dict(diff_evo.differential_evolution.__globals__);scope['repair']=trace_repair
            run=FunctionType(diff_evo.differential_evolution.__code__,scope)
            adapter.diagnostics['native_invocations']+=1
            run(native,cost,cost.get_bounds(),min(adapter.population_size,len(adapter.configs)),
                min(adapter.max_generations,len(adapter.configs)),.7,.6,'best1bin',True,False)
            self.outbox.put({'kind':'done'})
        except _Closed:self.outbox.put({'kind':'done'})
        except Exception as exc:self.outbox.put({'kind':'error','error':type(exc).__name__+': '+str(exc)})
