"""Real local LLM next-action decisions through the existing SDK/Runner.

The model stays in CPU memory between decisions, moves onto the admitted GPU
only for inference, then moves back and releases CUDA tensors before returning
a candidate. Upload/generation/offload costs are distinct and included in the
proposal wall. No Kernel measurement overlaps model inference.
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

_SESSIONS={}


def decision_task(hardware,operator,shape,space,history,budget):
    """Lossless finite domains with an unambiguous scalar-output contract.

    Small models copied the old nested parameter-description objects instead
    of choosing a value in a real H800 pilot. Flatten only the presentation;
    the authoritative space/constraints and final validation are unchanged.
    """
    task=describe_task(hardware,operator,shape,space,history,budget)
    specs=task['space'].pop('parameters')
    task['space']['choices']={name:list(space.domains[name]) for name in specs}
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
    def __init__(self,manifest):
        import torch,transformers
        verify_manifest(manifest);self.torch=torch;self.manifest=manifest
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not visible.startswith('GPU-') or ',' in visible or torch.cuda.device_count()!=1:
            raise ValueError('live inference requires one resource-admitted physical GPU')
        actual=str(getattr(torch.cuda.get_device_properties(0),'uuid',''))
        if actual.lower().removeprefix('gpu-')!=visible.lower().removeprefix('gpu-'):
            raise ValueError('LLM/Kernel GPU UUID mismatch')
        before=time.monotonic()
        self.tokenizer=transformers.AutoTokenizer.from_pretrained(manifest['local_path'],local_files_only=True,trust_remote_code=False)
        cls=transformers.AutoModelForImageTextToText if manifest['model_type']=='qwen3_5' else transformers.AutoModelForCausalLM
        self.model,info=cls.from_pretrained(manifest['local_path'],local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='sdpa',output_loading_info=True)
        if any('mtp' not in k.lower() for k in info.get('missing_keys',[])) or any('mtp' not in k.lower() for k in info.get('unexpected_keys',[])) or info.get('mismatched_keys'):
            raise ValueError('local checkpoint loading mismatch')
        self.model.eval();self.load_s=time.monotonic()-before;self.loading_info=loading_metadata(info)
        self.parameter_count=sum(p.numel() for p in self.model.parameters())
        if self.parameter_count<100_000_000:raise ValueError('not the expected complete local language model')
    def infer(self,messages,seed,max_tokens):
        torch=self.torch;before_bytes=torch.cuda.memory_allocated();before=time.monotonic()
        prompt=self.tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,
            enable_thinking=False,preserve_thinking=False,reasoning_effort='low')
        cpu_inputs=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
        if cpu_inputs['input_ids'].shape[-1]>8192:raise ValueError('live decision prompt exceeds context budget')
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);inputs=result=None
        row={'messages':messages,'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'seed':seed,'model_id':self.manifest['model_id'],'input_token_ids':cpu_inputs['input_ids'][0].tolist(),
            'CPU_model_load_wall_s':self.load_s,'parameter_count':self.parameter_count,'loading_info':self.loading_info}
        try:
            torch.cuda.synchronize();start=time.monotonic();self.model.to('cuda:0');torch.cuda.synchronize()
            row['upload_wall_s']=time.monotonic()-start;inputs=cpu_inputs.to('cuda:0')
            start=time.monotonic()
            with torch.inference_mode():result=self.model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,max_time=90)
            torch.cuda.synchronize();row['generation_wall_s']=time.monotonic()-start
            ids=result[0,inputs['input_ids'].shape[-1]:].tolist()
            eos=self.model.generation_config.eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos])
            row.update(output_token_ids=ids,input_tokens=inputs['input_ids'].shape[-1],output_tokens=len(ids),
                text=self.tokenizer.decode(ids,skip_special_tokens=True),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                finish_reason='eos' if ids and ids[-1] in eos else 'token_or_time_limit',
                generation_parameters={'max_new_tokens':max_tokens,'do_sample':False,'max_time':90},
                chat_template_options={'enable_thinking':False,'preserve_thinking':False,'reasoning_effort':'low'})
        finally:
            start=time.monotonic();inputs=result=None
            self.model.to('cpu')
            if hasattr(self.model,'_cache'):self.model._cache=None
            if hasattr(self.model,'rope_deltas'):self.model.rope_deltas=None
            del cpu_inputs;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
            row['offload_and_release_wall_s']=time.monotonic()-start
            row['CUDA_allocated_before_bytes']=before_bytes;row['CUDA_allocated_after_bytes']=torch.cuda.memory_allocated()
            row['all_parameters_on_CPU_before_Kernel']=all(p.device.type=='cpu' for p in self.model.parameters())
            if not row['all_parameters_on_CPU_before_Kernel'] or row['CUDA_allocated_after_bytes']>before_bytes+1024**2:
                raise RuntimeError('model is not quiescent/offloaded; refuse Kernel measurement')
        row['total_inference_proposal_wall_s']=time.monotonic()-before
        return row


class LiveDecision:
    def __init__(self,seed=0,*,model_manifest,manifest_sha256,max_new_tokens=192,max_retries=2):
        blob=Path(model_manifest).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=manifest_sha256:raise ValueError('frozen local model manifest mismatch')
        self.manifest=json.loads(blob)
        if not self.manifest.get('full_weight_hashes'):raise ValueError('full existing-weight fingerprints required')
        self.sha=manifest_sha256;self.seed=seed;self.max_tokens=max_new_tokens;self.retries=max_retries;self.output=None
        if type(max_retries) is not int or not 0<=max_retries<=2 or not 64<=max_new_tokens<=512:raise ValueError('bounded live inference settings required')
        self.diagnostics={'model_id':self.manifest['model_id'],'actual_model_calls':0,'model_retries':0,
            'upload_wall_s':0.,'generation_wall_s':0.,'offload_wall_s':0.,'output_tokens':0,
            'evidence_level':'real_local_model_next_action','isolation':'model fully onCPU before every Kernel call'}
    @property
    def runtime_identity(self):
        return {'manifest_sha256':self.sha,'model_id':self.manifest['model_id'],
            'source_sha256':digest_file(__file__),'feature_version':'scalar-choice-v3+lossless-envelope+observed-failures',
            'inference_backend':'Transformers5.8.1/sdpa/bfloat16','model_context':'target revealed history only',
            'isolation':'sequential same GPU, full weight offload before Kernel','max_new_tokens':self.max_tokens}
    def bind_trial(self,*,output_dir,identity):self.output=Path(output_dir)
    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0):return []
        seen={r.config.id for r in history}|set(history.attempted_config_ids)
        if self.output is None:
            # Explicit synthetic SDK contract check, never model-quality evidence.
            return list(space.complete({},hardware,operator,shape,count=min(batch_size,budget.max_evaluations),
                rng=random.Random(self.seed+len(history)),exclude_ids=seen,
                budget=SpaceBuildBudget(max_nodes=20000,max_wall_time_s=1)).configs)
        task=decision_task(hardware,operator,shape,space,history,budget)
        system='Choose the next GPU GEMM configuration. Return only JSON with a config object. For each parameter name in choices, choose exactly ONE scalar number from that list. Do not return lists or copy the space description. Include every parameter key. Do not repeat an attempted configuration. You may use only the supplied revealed history; no unobserved performance or speedup is known. Fused W4 unpacks inside GEMM; pipeline W4 dequantizes then GEMMs and optionally reduces Split-K on every invocation. Do not return predicted latency.'
        messages=[{'role':'system','content':system},{'role':'user','content':canonical_json(task)}]
        if self.sha not in _SESSIONS:_SESSIONS[self.sha]=_Session(self.manifest)
        session=_SESSIONS[self.sha]
        for attempt in range(self.retries+1):
            self.diagnostics['actual_model_calls']+=1
            call_started=time.monotonic()
            try:row=session.infer(messages,self.seed+len(history)*1009+attempt,self.max_tokens)
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
                action=parse_decision(row['text'],space.domains);config=space.normalize(action['config'])
                if set(action['config'])!=set(space.domains):raise ValueError('all declared fields must be explicitly predicted')
                valid=space.validate(config,hardware,operator,shape)
                if not valid.valid or config.id in seen:raise ValueError('illegal/repeated decision: '+str(valid.reasons))
                row['accepted_config']=dict(config.params);row['static_valid']=True
                row['response_form']=action.get('response_form','config_envelope')
            except Exception as exc:
                row['static_valid']=False;row['rejection']=type(exc).__name__+': '+str(exc)
            with (self.output/'llm-live-calls.jsonl').open('a') as f:f.write(canonical_json(row)+'\n')
            if row['static_valid']:return [config]
            self.diagnostics['model_retries']+=1
            messages.extend([{'role':'assistant','content':row['text']},{'role':'user','content':'Static contract error: '+row['rejection']+'. Return a corrected complete JSON configuration. No new performance feedback is supplied.'}])
        raise ValueError('real local LLM exhausted static-output repairs; no heuristic substitution')


ASSETS=(SearchAlgorithmDefinition('llm/live_local_decision','1.0.0',LiveDecision,
    SearchCapabilities(required_parameters=('BLOCK_M','BLOCK_N','BLOCK_K','num_warps','num_stages','GROUP_M')),
    metadata={'scope':'B: real small-model next-step decisions; weights offloaded between actions; no dedicated performance-model training'}),)
