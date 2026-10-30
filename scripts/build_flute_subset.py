"""CPU-only private build of an explicit FLUTE0.4.2 template subset.

Numerical headers/kernel bodies stay upstream. Only generated template dispatch
cases and the private CUTLASS include path change. GPU execution is never used
to detect an architecture; target architectures are explicit.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from kernel_tuning_lab.production import flute,native_sources


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source,cutlass,work,ids,accumulation='mixed'):
    if accumulation not in ('mixed','high'):raise ValueError('explicit original accumulation mode required')
    source,cutlass,work=map(lambda p:Path(p).resolve(),(source,cutlass,work))
    if work==source or source in work.parents:raise ValueError('build in a separate private copy, never modify the source checkout')
    original=native_sources.native_component(source,'flute',['flute/codegen_utils.py','flute/csrc/qgemm.cpp','flute/csrc/qgemm_kernel_raw_generated.cu','flute/utils.py'])
    if not (cutlass/'include/cute/tensor.hpp').is_file():raise ValueError('fixed CUTLASS headers missing')
    if work.exists():raise ValueError('build work directory already exists; preserve it, use a fresh attempt or its native Ninja continuation')
    shutil.copytree(source,work,ignore=shutil.ignore_patterns('.git','build','*.egg-info','__pycache__'))
    path=native_sources.source('flute','flute/codegen_utils.py');tree=ast.parse(path.read_text())
    original_codegen=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='codegen_raw')
    nodes=[]
    for node in original_codegen.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='names' for t in node.targets):break
        nodes.append(node)
    function=ast.FunctionDef('_cases',original_codegen.args,[*nodes,ast.Return(ast.Name('cases',ast.Load()))],[],None)
    ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(path),'exec'),ns)
    all_cases=ns['_cases']();selected={(3,i):all_cases[(3,i)] for i in ids}
    generate={};native_sources.host_functions('flute','flute/codegen_utils.py',{'generate_nested_switch'},generate)
    switch=generate['generate_nested_switch'](['NumBits','template_id'],selected,[True,False],'    ')
    if accumulation=='high':switch=switch.replace('kMixed','kHigh')
    target=work/'flute/csrc/qgemm_kernel_raw_generated.cu';text=target.read_text();before=digest(target)
    marker=text.index('    // Generated Code Below');start=text.index('\n',marker)+1
    # The generated dispatch ends at the next function-closing brace. Strings
    # and comments cannot contribute braces to the match.
    stripped=re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"',lambda m:' '*len(m.group()),text,flags=re.S)
    opening=stripped.index('{',stripped.index('void\n_qgemm_raw'))
    depth=0;closing=None
    for index in range(opening,len(stripped)):
        if stripped[index]=='{':depth+=1
        elif stripped[index]=='}':
            depth-=1
            if depth==0:closing=index;break
    if closing is None or closing<=start:raise ValueError('cannot identify original generated dispatch boundary')
    target.write_text(text[:start]+switch+'\n'+text[closing:])
    setup=work/'setup.py';setup_text=setup.read_text();needle='CUTLASS_PATH = "/workspace/cutlass/"'
    if needle not in setup_text:raise ValueError('unexpected upstream build include-path declaration')
    setup.write_text(setup_text.replace(needle,'CUTLASS_PATH = os.environ["FLUTE_CUTLASS_PATH"]'))
    plan={'source_revision':'9eb83a12d56949bbe7fe9c836ba97a67bd1e3761','source_identity':original,
        'compiled_templates':{'3':ids},'source_expressible_templates':{'3':36,'2':36,'4':144},
        'accumulation_mode':'kHigh' if accumulation=='high' else 'kMixed',
        'numeric_profile':'original FP32 MMA and FP32 StreamK reduction branch' if accumulation=='high' else 'original mixed reduction',
        'patches':[{'file':'setup.py','kind':'private include-path configuration, no numeric source changes'},
            {'file':'flute/csrc/qgemm_kernel_raw_generated.cu','kind':'original RUN_QGEMM geometry IDs with explicit '+accumulation+' accumulation argument; headers and numerical bodies unchanged',
             'original_sha256':before,'patched_sha256':digest(target)}],
        'native_kernel_headers_modified':False,'cutlass_tensor_header_sha256':digest(cutlass/'include/cute/tensor.hpp'),
        'unknown_metrics':['per-template cubin identities','per-template code bytes','GPU correctness/performance']}
    (work/'build-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--cutlass',required=True);parser.add_argument('--work',required=True)
    parser.add_argument('--templates',default='0,1,2,8,9,10');parser.add_argument('--architectures',default='8.9;9.0');parser.add_argument('--cuda-home');parser.add_argument('--build',action='store_true');parser.add_argument('--timeout',type=int,default=2400)
    parser.add_argument('--accumulation',choices=('mixed','high'),default='high',help='default: original FP32 reduction branch; mixed is an explicit reproduction profile, never relabel an existing binary')
    parser.add_argument('--resource-facts',help='CPU sizeof facts from probe_flute_resources.py using these exact headers')
    args=parser.parse_args();ids=[int(x) for x in args.templates.split(',')]
    if not ids or len(ids)!=len(set(ids)) or any(i not in range(36) for i in ids):raise ValueError('original int3 IDs must be distinct0..35')
    if not all(x in ('8.0','8.6','8.9','9.0') for x in args.architectures.split(';')):raise ValueError('explicit audited target architecture list required')
    plan=prepare(args.source,args.cutlass,args.work,ids,args.accumulation);work=Path(args.work).resolve()
    if args.resource_facts:
        facts=json.loads(Path(args.resource_facts).read_text())
        if (facts['config_header_sha256']!=digest(work/'flute/csrc/config.hpp')
                or facts['cutlass_tensor_header_sha256']!=plan['cutlass_tensor_header_sha256']):raise ValueError('resource facts came from different build headers')
        plan.update(template_resources=facts['template_resources'],resource_facts={k:v for k,v in facts.items() if k!='template_resources'})
        (work/'build-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.build:
        if not args.cuda_home:raise ValueError('explicit existing CUDA toolkit required; no system install')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','TORCH_CUDA_ARCH_LIST':args.architectures,'CUDA_HOME':args.cuda_home,
            'FLUTE_CUTLASS_PATH':str(Path(args.cutlass).resolve()),'MAX_JOBS':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        began=time.monotonic()
        with (work/'native-build.log').open('w') as log:
            result=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
        plan.update(build_returncode=result.returncode,build_wall_s=time.monotonic()-began,compiled_architectures=['sm_'+a.replace('.','') for a in args.architectures.split(';')])
        binaries=list((work/'flute').glob('_C*.so'))
        if result.returncode==0 and len(binaries)==1:
            import torch
            binary=binaries[0];plan.update(binary_sha256=digest(binary),binary_bytes=binary.stat().st_size,
                torch=torch.__version__,torch_cuda_runtime=torch.version.cuda,build_status='passed',GPU_status='not_executed')
        else:plan['build_status']='failed'
        (work/'component-manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
        if result.returncode:raise SystemExit(result.returncode)
    print(json.dumps(plan,indent=2))


if __name__=='__main__':main()
