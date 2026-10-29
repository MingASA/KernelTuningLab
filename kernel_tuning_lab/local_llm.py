"""Bounded offline inference using an existing local checkpoint, not an Agent.

Torch/Transformers are optional and imported only inside run(). Model weights
are never downloaded, changed, quantized or trained. Run under the ordinary
exclusive-GPU guard, and let this process exit before Kernel measurement.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from kernel_tuning_lab.models import canonical_json


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def checkpoint_manifest(directory,model_id,*,hash_weights=False):
    root=Path(directory).resolve()
    if not root.is_dir():raise ValueError('an existing local model directory is required; downloads are disabled')
    config=root/'config.json'
    if not config.is_file():raise ValueError('local model config missing')
    index=root/'model.safetensors.index.json'
    if index.exists():
        names=sorted(set(json.loads(index.read_text())['weight_map'].values()))
    else:names=[p.name for p in root.glob('*.safetensors')]
    if not names or any(not n.endswith('.safetensors') or Path(n).is_absolute() or '..' in Path(n).parts for n in names):
        raise ValueError('complete local safetensors checkpoints required')
    missing=[n for n in names if not (root/n).is_file()]
    if missing:raise ValueError('local checkpoint is incomplete: '+str(missing[:8]))
    weights=[]
    for name in names:
        file=root/name;stat=file.stat()
        entry={'name':name,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'ctime_ns':stat.st_ctime_ns}
        if hash_weights:entry['sha256']=digest_file(file)
        weights.append(entry)
    meta={p.name:digest_file(p) for p in root.iterdir() if p.is_file() and
          (p.name in ('config.json','generation_config.json','chat_template.jinja','model.safetensors.index.json') or p.name.startswith(('tokenizer','vocab','merges')))}
    return {'schema_version':1,'kind':'existing_local_transformers_checkpoint','model_id':model_id,
        'model_type':json.loads(config.read_text())['model_type'],'local_path':str(root),
        'metadata_sha256':meta,'weights':weights,'weights_bytes':sum(x['bytes'] for x in weights),
        'full_weight_hashes':hash_weights,'downloads':False,'training':False}


def verify_manifest(manifest):
    root=Path(manifest['local_path'])
    for item in manifest['weights']:
        stat=(root/item['name']).stat()
        if any(stat_value!=item[key] for key,stat_value in [('bytes',stat.st_size),('mtime_ns',stat.st_mtime_ns),('ctime_ns',stat.st_ctime_ns)]):
            raise ValueError('model weight changed since fingerprinting: '+item['name'])
    for name,sha in manifest['metadata_sha256'].items():
        if digest_file(root/name)!=sha:raise ValueError('model metadata/tokenizer changed: '+name)


def _write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(canonical_json(value)+'\n');temp.replace(path)


def run(manifest_file,queries_file,output,*,max_wall_s=1800):
    started=time.monotonic();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'generations.jsonl').exists():raise ValueError('inference output already exists; use a fresh attempt and retain previous evidence')
    manifest=json.loads(Path(manifest_file).read_text());verify_manifest(manifest)
    if not manifest.get('full_weight_hashes'):raise ValueError('actual experiments require full weight-file fingerprints')
    queries=json.loads(Path(queries_file).read_text())['queries']
    if not queries or len(queries)>64 or len({q['id'] for q in queries})!=len(queries):raise ValueError('bounded distinct query IDs required')
    for q in queries:
        if not isinstance(q.get('messages'),list) or not 1<=q.get('generation',{}).get('max_new_tokens',1024)<=8192:
            raise ValueError('structured messages and finite output tokens required')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    import torch
    import transformers
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible.startswith('GPU-') or ',' in visible or torch.cuda.device_count()!=1:
        raise ValueError('run inference under the resource guard with one physical GPU UUID')
    actual=str(getattr(torch.cuda.get_device_properties(0),'uuid',''))
    normalize=lambda x:x.lower().removeprefix('gpu-')
    if normalize(actual)!=normalize(visible):raise ValueError('visible CUDA ordinal does not match the admitted physical GPU')
    identity={'model_id':manifest['model_id'],'manifest_sha256':digest_file(manifest_file),
        'query_bundle_sha256':digest_file(queries_file),'worker_source_sha256':digest_file(__file__),
        'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','accelerate','safetensors')},
        'dtype':'bfloat16','attention_backend':'sdpa','device_map':'single admitted CUDA device; no CPU/disk/TP/DP offload',
        'evidence_level':'real_local_model_inference','physical_gpu_uuid':actual}
    _write(out/'run.json',{'status':'loading','identity':identity,'query_count':len(queries)})
    load_started=time.monotonic()
    tokenizer=transformers.AutoTokenizer.from_pretrained(manifest['local_path'],local_files_only=True,trust_remote_code=False)
    cls=transformers.AutoModelForImageTextToText if manifest['model_type']=='qwen3_5' else transformers.AutoModelForCausalLM
    model,info=cls.from_pretrained(manifest['local_path'],local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',output_loading_info=True)
    critical_missing=[k for k in info.get('missing_keys',[]) if 'mtp' not in k.lower()]
    unexpected=[k for k in info.get('unexpected_keys',[]) if 'mtp' not in k.lower()]
    if critical_missing or unexpected or info.get('mismatched_keys'):
        raise ValueError('checkpoint/model implementation mismatch: '+str({'missing':critical_missing[:8],'unexpected':unexpected[:8]}))
    if any(p.device.type!='cuda' or p.device.index!=0 for p in model.parameters()):raise ValueError('all model parameters must reside on the one admitted GPU')
    model.eval();torch.cuda.synchronize();load_s=time.monotonic()-load_started
    verify_manifest(manifest)
    count=sum(p.numel() for p in model.parameters())
    if count<1_000_000_000:raise ValueError('unexpectedly small model; refuse fixture/submodel evidence')
    _write(out/'run.json',{'status':'generating','identity':identity,'load_wall_s':load_s,
        'parameter_count':count,'loading_info':info,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    done=0
    for q in queries:
        if time.monotonic()-started>=max_wall_s:raise TimeoutError('finite whole-inference wall allowance exhausted')
        seed=q.get('seed',0);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        template={'add_generation_prompt':True,'tokenize':False,**q.get('template',{})}
        prompt=tokenizer.apply_chat_template(q['messages'],**template)
        inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to('cuda:0')
        if inputs['input_ids'].shape[-1]>16384:raise ValueError('prompt exceeds the frozen inference context budget')
        params={'max_new_tokens':1024,'do_sample':False,**q.get('generation',{})}
        allowed={'max_new_tokens','do_sample','temperature','top_p','top_k','repetition_penalty'}
        if set(params)-allowed:raise ValueError('unsupported generation parameter')
        if tokenizer.pad_token_id is not None:params['pad_token_id']=tokenizer.pad_token_id
        torch.cuda.synchronize();before=time.monotonic()
        with torch.inference_mode():
            result=model.generate(**inputs,**params,max_time=min(q.get('max_wall_s',300),max_wall_s-(time.monotonic()-started)))
        torch.cuda.synchronize();elapsed=time.monotonic()-before
        generated=result[0,inputs['input_ids'].shape[-1]:].tolist();text=tokenizer.decode(generated,skip_special_tokens=True)
        eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        finished=bool(generated and generated[-1] in eos)
        row={'query_id':q['id'],'messages':q['messages'],'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'template':template,'generation_parameters':params,'seed':seed,
            'input_token_ids':inputs['input_ids'][0].tolist(),'output_token_ids':generated,'text':text,
            'input_tokens':inputs['input_ids'].shape[-1],'output_tokens':len(generated),'generation_wall_s':elapsed,
            'finish_reason':'eos' if finished else 'token_or_time_limit','identity':identity,'load_wall_s':load_s,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        with (out/'generations.jsonl').open('a') as f:f.write(canonical_json(row)+'\n')
        done+=1;_write(out/'progress.json',{'completed':done,'queries':len(queries),'last_query_id':q['id']})
        del inputs,result
    verify_manifest(manifest)
    _write(out/'run.json',{'status':'passed','identity':identity,'query_count':len(queries),'completed':done,
        'parameter_count':count,'load_wall_s':load_s,'total_wall_s':time.monotonic()-started,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'scope':'Model inference only. Process exit releases weights before any Kernel benchmark; no tuning feedback was generated here.'})
    return done


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('manifest');a.add_argument('--model',required=True);a.add_argument('--model-id',required=True);a.add_argument('--hash-weights',action='store_true');a.add_argument('--output',required=True)
    a=sub.add_parser('run');a.add_argument('--manifest',required=True);a.add_argument('--queries',required=True);a.add_argument('--output',required=True);a.add_argument('--max-wall-s',type=float,default=1800)
    args=p.parse_args()
    if args.command=='manifest':_write(Path(args.output),checkpoint_manifest(args.model,args.model_id,hash_weights=args.hash_weights))
    else:print(json.dumps({'completed':run(args.manifest,args.queries,args.output,max_wall_s=args.max_wall_s)}))


if __name__=='__main__':main()
