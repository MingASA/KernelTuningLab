"""C-route client entry: API/local model -> retained original -> review workflow.

This never executes returned code. API auth/data-egress is explicit; absent
authorization denies before inspecting a key. No models are downloaded.
"""
import argparse
import json
from pathlib import Path
from kernel_tuning_lab.api_inference import APIProvider,APIAuthorization,PublicModelClient
from kernel_tuning_lab.local_inference import LocalModelClient
from kernel_tuning_lab.program_artifacts import generate


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--provider',required=True)
    parser.add_argument('--task',required=True);parser.add_argument('--output',required=True);parser.add_argument('--authorization')
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise ValueError('new output per model attempt; preserve earlier responses/failures')
    out.mkdir(parents=True,mode=0o700);provider=json.loads(Path(args.provider).read_text())
    if provider['kind']=='public_api':
        authorization=APIAuthorization(**json.loads(Path(args.authorization).read_text())) if args.authorization else APIAuthorization()
        client=PublicModelClient(APIProvider(**provider['configuration']),authorization=authorization,log_path=out/'calls.jsonl')
    elif provider['kind']=='local_model':client=LocalModelClient(**provider['configuration'],log_path=out/'calls.jsonl')
    else:raise ValueError('program generation provider is public_api or an existing local_model')
    try:
        response=generate(client,json.loads(Path(args.task).read_text()))
        (out/'model-response.txt').write_text(response['text'])
        value=json.loads(response['text'])
        if set(value)!={'program','algorithm_ref','rationale'} or not all(isinstance(v,str) for v in value.values()):
            raise ValueError('program/ref/rationale string output contract failed; raw response retained')
        (out/'model-program.py').write_text(value['program'])
        (out/'generation.json').write_text(json.dumps({**response,'algorithm_ref':value['algorithm_ref']},indent=2)+'\n')
        print(json.dumps({'status':'generated_not_executed','algorithm_ref':value['algorithm_ref'],'next':'review source; ai-program submit then explicit review/check/development/freeze'}))
    finally:
        if callable(getattr(client,'close',None)):client.close()


if __name__=='__main__':main()
