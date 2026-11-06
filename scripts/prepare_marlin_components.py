"""Verify/extract the fixed official vLLM0.8.5 operator, without installing a server.

Download the pinned wheel separately using the machine's approved network path.
This command is CPU-only and never changes a Python environment or system CUDA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

WHEEL_SHA='74bfe92953bee1269c1e1c27827bc156777751cdd6a3457ee8e27dd8ebf1e247'
BINARY_SHA='bf47ca452c7cf18cab47b651b29d9c1a703d32149e20139d92a472726aba6ccb'
MEMBERS=('vllm/_C.abi3.so','vllm/_custom_ops.py','vllm/scalar_type.py',
 'vllm/model_executor/layers/quantization/utils/marlin_utils.py',
 'vllm/model_executor/layers/quantization/utils/marlin_utils_test.py',
 'vllm/model_executor/layers/quantization/utils/quant_utils.py')


def extract(wheel,output):
    wheel,output=Path(wheel),Path(output);digest=hashlib.sha256()
    with wheel.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    if digest.hexdigest()!=WHEEL_SHA:raise ValueError('not the pinned official vLLM0.8.5 wheel; no components extracted')
    output.mkdir(mode=0o700,parents=True,exist_ok=True);files=[]
    with zipfile.ZipFile(wheel) as archive:
        for name in MEMBERS:
            blob=archive.read(name);path=output/name;path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            if path.exists():
                if path.read_bytes()!=blob:raise ValueError('refuse to overwrite a different component: '+name)
            else:path.write_bytes(blob);path.chmod(0o600)
            files.append({'file':name,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
    if next(f['sha256'] for f in files if f['file']=='vllm/_C.abi3.so')!=BINARY_SHA:raise ValueError('native binary identity mismatch')
    manifest={'version':'0.8.5','required_torch':'2.6.0','wheel_sha256':WHEEL_SHA,
              'binary_sha256':BINARY_SHA,'files':files,'gpu_validation':'not_executed_by_this_CPU_command'}
    (output/'component-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wheel',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=extract(args.wheel,args.output)
    print(json.dumps({'status':'components_verified_cpu_only','version':result['version'],'binary_sha256':BINARY_SHA}))
