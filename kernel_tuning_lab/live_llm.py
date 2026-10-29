"""Real local LLM next-action decisions through the existing SDK/Runner.

The default keeps weights resident, with inference and measurement serialized
by a cooperative lease. Stage offload is an explicit resource fallback. Neither
allocation statistics nor empty_cache are used as concurrency proof.
"""
from __future__ import annotations
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

from .sdk import SearchAlgorithmDefinition,SearchCapabilities,SpaceBuildBudget
from .external_search import describe_task
from .local_llm import digest_file,verify_manifest,loading_metadata
from .models import canonical_json
from .ai_actions import parse_action,resolve_action
from .inference_resources import COORDINATOR

_SESSIONS={}


def decision_task(hardware,operator,shape,space,history,budget):
    """Lossless finite domains with an unambiguous scalar-output contract.

    Small models copied the old nested parameter-description objects instead
    of choosing a value in a real H800 pilot. Flatten only the presentation;
    the authoritative space/constraints and final validation are unchanged.
    """
    task=describe_task(hardware,operator,shape,space,history,budget)
    specs=task['space'].pop('parameters')
    task['space']['choices']={name:(list(space.domains[name]) if len(space.domains[name])<=32 else
        {'size':len(space.domains[name]),'query':{'kind':'values','parameter':name,'offset':0,'limit':16}}) for name in specs}
    task['space']['active_when']={name:spec['active_when'] for name,spec in specs.items() if spec['active_when']}
    task['history']=[{**value,'failure_reason':record.failure_reason} for value,record in zip(task['history'],history)]
    task.pop('allowed_responses')
    task['response_contract']={'config':'one JSON object: every choice name maps to ONE scalar from its allowed choices; never an array or a domain-description object',
        'required_keys':list(specs),'no_repeated_configuration':True}
    return task


def parse_decision(text,parameter_keys=None):
    text=text.strip()
    if '</think>' in text:text=text.rsplit('</think>',1)[1].strip()
    if text.startswith('```') and text.endswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    value=json.loads(text)
    if (parameter_keys is not None and isinstance(value,dict) and set(value)==set(parameter_keys)
            and all(not isinstance(v,(dict,list)) for v in value.values())):
        # A real 0.8B pilot returned every legal scalar but omitted the outer
        # envelope. This changes no proposed parameter and is logged explicitly.
        return {'config':value,'response_form':'bare_complete_config_lossless_envelope'}
    if not isinstance(value,dict) or set(value)-{'config','rationale'} or not isinstance(value.get('config'),dict):
        raise ValueError('a complete {"config": {...}} decision is required')
    return value


class _Session:
    def __init__(self,manifest,resource_mode='single_card_resident',manifest_sha=None):
        import torch,transformers
        verify_manifest(manifest);self.torch=torch;self.manifest=manifest
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not visible.startswith('GPU-') or ',' in visible or torch.cuda.device_count()!=1:
            raise ValueError('live inference requires one resource-admitted physical GPU')
        actual=str(getattr(torch.cuda.get_device_properties(0),'uuid',''))
        if actual.lower().removeprefix('gpu-')!=visible.lower().removeprefix('gpu-'):
            raise ValueError('LLM/Kernel GPU UUID mismatch')
        self.mode=resource_mode;before=time.monotonic()
        self.lease_token=COORDINATOR.register(manifest['model_id'],mode=resource_mode,device_uuid=actual,
            identity={'model_id':manifest['model_id'],'manifest_sha256':manifest_sha},
            synchronize=lambda:torch.cuda.synchronize(),resident_bytes=manifest['weights_bytes'] if resource_mode=='single_card_resident' else 0)
        try:
            self.tokenizer=transformers.AutoTokenizer.from_pretrained(manifest['local_path'],local_files_only=True,trust_remote_code=False)
            cls=transformers.AutoModelForImageTextToText if manifest['model_type']=='qwen3_5' else transformers.AutoModelForCausalLM
            with COORDINATOR.inference_window(self.lease_token):
                self.model,info=cls.from_pretrained(manifest['local_path'],local_files_only=True,trust_remote_code=False,
                    dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',output_loading_info=True)
                if any('mtp' not in k.lower() for k in info.get('missing_keys',[])) or any('mtp' not in k.lower() for k in info.get('unexpected_keys',[])) or info.get('mismatched_keys'):
                    raise ValueError('local checkpoint loading mismatch')
                self.model.eval();self.parameter_count=sum(p.numel() for p in self.model.parameters())
                if self.parameter_count<100_000_000:raise ValueError('not the expected complete local language model')
        except BaseException:
            if hasattr(self,'model'):del self.model
            COORDINATOR.unregister(self.lease_token);raise
        self.load_s=time.monotonic()-before;self.loading_info=loading_metadata(info)
    def close(self):
        with COORDINATOR.inference_window(self.lease_token):
            del self.model;gc.collect();self.torch.cuda.synchronize()
        COORDINATOR.unregister(self.lease_token)
    def infer(self,messages,seed,max_tokens,decoding='greedy',constraint=None):
        with COORDINATOR.inference_window(self.lease_token):
            return self._infer_serial(messages,seed,max_tokens,decoding,constraint)

    def _infer_serial(self,messages,seed,max_tokens,decoding='greedy',constraint=None):
        torch=self.torch;before_bytes=torch.cuda.memory_allocated();before=time.monotonic()
        prompt=self.tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,
            enable_thinking=False,preserve_thinking=False,reasoning_effort='low')
        cpu_inputs=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
        if cpu_inputs['input_ids'].shape[-1]>8192:raise ValueError('live decision prompt exceeds context budget')
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);inputs=result=None
        row={'messages':messages,'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'seed':seed,'model_id':self.manifest['model_id'],'input_token_ids':cpu_inputs['input_ids'][0].tolist(),
            'model_load_wall_s':self.load_s,'initial_load_device':'admitted CUDA device',
            'resource_mode':self.mode,'parameter_count':self.parameter_count,'loading_info':self.loading_info}
        params={'max_new_tokens':max_tokens,'do_sample':decoding=='sampled','max_time':90}
        if decoding=='sampled':params.update(temperature=.7,top_p=.8,top_k=20,repetition_penalty=1.)
        prefix_fn=None
        if constraint is not None:
            eos=self.model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            offset=cpu_inputs['input_ids'].shape[-1]
            prefix_fn=lambda batch,ids:constraint.allowed(ids[offset:].tolist(),eos)
        try:
            torch.cuda.synchronize();start=time.monotonic();self.model.to('cuda:0');torch.cuda.synchronize()
            row['upload_wall_s']=time.monotonic()-start;inputs=cpu_inputs.to('cuda:0')
            start=time.monotonic()
            with torch.inference_mode():result=self.model.generate(**inputs,**params,
                **({'prefix_allowed_tokens_fn':prefix_fn} if prefix_fn is not None else {}))
            torch.cuda.synchronize();row['generation_wall_s']=time.monotonic()-start
            ids=result[0,inputs['input_ids'].shape[-1]:].tolist()
            eos=self.model.generation_config.eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos])
            row.update(output_token_ids=ids,input_tokens=inputs['input_ids'].shape[-1],output_tokens=len(ids),
                text=self.tokenizer.decode(ids,skip_special_tokens=True),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                finish_reason='eos' if ids and ids[-1] in eos else 'token_or_time_limit',
                generation_parameters=params,
                chat_template_options={'enable_thinking':False,'preserve_thinking':False,'reasoning_effort':'low'})
        finally:
            start=time.monotonic();inputs=result=None
            if self.mode=='stage_offload':self.model.to('cpu')
            if hasattr(self.model,'_cache'):self.model._cache=None
            if hasattr(self.model,'rope_deltas'):self.model.rope_deltas=None
            del cpu_inputs;gc.collect();torch.cuda.synchronize()
            if self.mode=='stage_offload':torch.cuda.empty_cache()
            row['offload_and_release_wall_s']=time.monotonic()-start
            row['CUDA_allocated_before_bytes']=before_bytes;row['CUDA_allocated_after_bytes']=torch.cuda.memory_allocated()
            row['all_parameters_on_CPU_before_Kernel']=all(p.device.type=='cpu' for p in self.model.parameters())
            row['all_parameters_resident_on_GPU']=all(p.device.type=='cuda' for p in self.model.parameters())
            row['CUDA_reserved_after_bytes']=torch.cuda.memory_reserved()
            row['quiescence_authority']='same-process cooperative lease + producer-owned completion sync; allocation statistics are resource metadata, not concurrency proof'
            if self.mode=='stage_offload' and (not row['all_parameters_on_CPU_before_Kernel'] or row['CUDA_allocated_after_bytes']>before_bytes+1024**2):
                raise RuntimeError('model is not quiescent/offloaded; refuse Kernel measurement')
        row['total_inference_proposal_wall_s']=time.monotonic()-before
        return row


class LiveDecision:
    def __init__(self,seed=0,*,model_manifest,manifest_sha256,max_new_tokens=192,max_retries=2,decoding='greedy',structured_output=False,resource_mode='single_card_resident'):
        blob=Path(model_manifest).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=manifest_sha256:raise ValueError('frozen local model manifest mismatch')
        self.manifest=json.loads(blob)
        if not self.manifest.get('full_weight_hashes'):raise ValueError('full existing-weight fingerprints required')
        self.sha=manifest_sha256;self.seed=seed;self.max_tokens=max_new_tokens;self.retries=max_retries;self.output=None
        if decoding not in ('greedy','sampled'):raise ValueError('decoding is greedy or sampled; no implicit profile fallback')
        self.decoding=decoding
        if type(structured_output) is not bool:raise ValueError('structured_output is boolean')
        self.structured=structured_output;self.grammar=None;self.grammar_context=None
        if resource_mode not in ('single_card_resident','stage_offload'):raise ValueError('local in-process mode is single_card_resident or explicit stage_offload fallback')
        self.resource_mode=resource_mode;self.rng=random.Random(seed)
        if type(max_retries) is not int or not 0<=max_retries<=2 or not 64<=max_new_tokens<=512:raise ValueError('bounded live inference settings required')
        self.diagnostics={'model_id':self.manifest['model_id'],'actual_model_calls':0,'model_retries':0,
            'upload_wall_s':0.,'generation_wall_s':0.,'offload_wall_s':0.,'output_tokens':0,
            'evidence_level':'real_local_model_next_action','resource_mode':resource_mode,
            'isolation':'cooperative same-process serial gate; resident weights may remain','queries':0,'platform_completions':0}
    @property
    def runtime_identity(self):
        return {'manifest_sha256':self.sha,'model_id':self.manifest['model_id'],
            'source_sha256':digest_file(__file__),'feature_version':'generic-action-v4+bounded-space-queries',
            'inference_backend':'Transformers5.8.1/sdpa/bfloat16','model_context':'target revealed history only',
            'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','accelerate','safetensors')},
            'isolation':'cooperative serial lease with owned completion synchronization','resource_mode':self.resource_mode,'max_new_tokens':self.max_tokens,
            'decoding':self.decoding,'structured_output':self.structured,
            'grammar':'complete static feasible set; no performance labels; exclude revealed configs' if self.structured else None,
            'sampling_profile':{'temperature':.7,'top_p':.8,'top_k':20,'repetition_penalty':1.} if self.decoding=='sampled' else None}
    def bind_trial(self,*,output_dir,identity):self.output=Path(output_dir)
    def close(self):
        session=_SESSIONS.pop((self.sha,self.resource_mode),None)
        if session is not None and hasattr(session,'close'):session.close()
    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0):return []
        seen={r.config.id for r in history}|set(history.attempted_config_ids)
        if self.output is None:
            # Explicit synthetic SDK contract check, never model-quality evidence.
            return list(space.complete({},hardware,operator,shape,count=min(batch_size,budget.max_evaluations),
                rng=random.Random(self.seed+len(history)),exclude_ids=seen,
                budget=SpaceBuildBudget(max_nodes=20000,max_wall_time_s=1)).configs)
        task=decision_task(hardware,operator,shape,space,history,budget)
        system='Choose the next operator configuration using only the declared task and revealed history. Return JSON with exactly one selector: config (complete configuration), partial (some fields to be constructively completed by the platform), region (advisory finite value lists), or query (kind values/validate_partial/complete). Do not repeat attempted configurations or invent measured performance. Large domains may be queried in bounded pages. Never return code or copy the domain description.'
        messages=[{'role':'system','content':system},{'role':'user','content':canonical_json(task)}]
        session_key=(self.sha,self.resource_mode)
        if session_key not in _SESSIONS:_SESSIONS[session_key]=_Session(self.manifest,self.resource_mode,self.sha)
        session=_SESSIONS[session_key]
        if self.structured:
            context=(hardware.id,operator.id,shape.id,space.fingerprint)
            if self.grammar is None:
                from .token_constraints import ConfigTokenTrie
                began=time.monotonic()
                built=space.build_feasible(hardware,operator,shape,
                    budget=SpaceBuildBudget(max_nodes=500000,max_wall_time_s=15,max_configs=50000))
                if not built.complete:raise ValueError('full static output grammar cannot be built within its declared allowance: '+built.status)
                self.grammar=ConfigTokenTrie(built.configs,session.tokenizer);self.grammar_context=context
                if self.grammar.max_output_tokens>self.max_tokens:
                    raise ValueError('declared output token cap cannot cover the full config grammar')
                self.diagnostics['grammar']={'mode':'HF prefix_allowed_tokens_fn; actual model logits choose every branch',
                    'full_static_configs':len(built.configs),'token_nodes':len(self.grammar.edges),
                    'max_output_tokens':self.grammar.max_output_tokens,
                    'build_wall_s':time.monotonic()-began,'space_build':dict(built.diagnostics)}
            if context!=self.grammar_context:raise ValueError('one live decoder cannot mix task contexts')
            self.grammar.exclude(seen)
            if not self.grammar.remaining:self.last_proposal_status='complete_static_set_exhausted';return []
            self.diagnostics['grammar']['remaining_configs']=self.grammar.remaining
        for attempt in range(self.retries+1):
            self.diagnostics['actual_model_calls']+=1
            call_started=time.monotonic()
            try:row=session.infer(messages,self.seed+len(history)*1009+attempt,self.max_tokens,self.decoding,self.grammar)
            except Exception as exc:
                failed={'model_manifest_sha256':self.sha,'observation_index':len(history),'attempt':attempt,
                    'messages':messages,'error':type(exc).__name__+': '+str(exc),
                    'failed_inference_wall_s':time.monotonic()-call_started,
                    'no_Kernel_evaluation_admitted':True}
                with (self.output/'llm-live-calls.jsonl').open('a') as f:f.write(canonical_json(failed)+'\n')
                raise
            row.update(observation_index=len(history),attempt=attempt,model_manifest_sha256=self.sha)
            self.diagnostics['output_tokens']+=row['output_tokens']
            self.diagnostics['upload_wall_s']+=row['upload_wall_s'];self.diagnostics['generation_wall_s']+=row['generation_wall_s'];self.diagnostics['offload_wall_s']+=row['offload_and_release_wall_s']
            try:
                action=parse_action(row['text'],space.domains)
                config,trace=resolve_action(action,hardware,operator,shape,space,history,rng=self.rng)
                row['action_trace']=trace;row['static_valid']=True
                if config is None:
                    self.diagnostics['queries']+=1;row['query_only']=True
                else:
                    row['accepted_config']=dict(config.params);self.diagnostics['platform_completions']+=int(bool(trace['platform_completed_fields']))
                row['response_form']='bare_complete_config_lossless_envelope' if action.get('normalization') else 'config_envelope'
            except Exception as exc:
                row['static_valid']=False;row['rejection']=type(exc).__name__+': '+str(exc)
            with (self.output/'llm-live-calls.jsonl').open('a') as f:f.write(canonical_json(row)+'\n')
            if row['static_valid'] and row.get('query_only'):
                messages.extend([{'role':'assistant','content':row['text']},{'role':'user','content':canonical_json(trace['query_result'])}]);continue
            if row['static_valid']:return [config]
            self.diagnostics['model_retries']+=1
            messages.extend([{'role':'assistant','content':row['text']},{'role':'user','content':'Static contract error: '+row['rejection']+'. Return a corrected complete JSON configuration. No new performance feedback is supplied.'}])
        raise ValueError('real local LLM exhausted static-output repairs; no heuristic substitution')


ASSETS=(SearchAlgorithmDefinition('llm/live_local_decision','1.0.0',LiveDecision,
    SearchCapabilities(),
    source_files=('local_llm.py','external_search.py','token_constraints.py','ai_actions.py','inference_resources.py'),
    metadata={'scope':'B: generic config/partial/region/space-query decisions; cooperative resident default; optional finite trie and offload fallback; no model training'}),)
