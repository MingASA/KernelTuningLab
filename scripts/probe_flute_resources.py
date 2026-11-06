"""Compile/run a CPU-only sizeof probe of original FLUTE template storage.

No CUDA kernel or device query is called. This reports the exact host launch
request Config::kSmemSize, not a performance-based shared-memory estimate.
"""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
from kernel_tuning_lab.production import flute,native_sources

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--cutlass',required=True);p.add_argument('--nvcc',required=True);p.add_argument('--compiler-bindir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);source=Path(a.source);cutlass=Path(a.cutlass)
    header=source/'flute/csrc/config.hpp';assert header.read_bytes()==native_sources.source('flute','flute/csrc/config.hpp').read_bytes()
    lines=['#include <cstdio>','#include "config.hpp"','using namespace cute;','int main() {']
    for candidate in flute.candidates():
        c=candidate.parameters
        if c['TileP']!=32:continue
        for group in [64,128]:
            for dtype,ctype in [('float16','half_t'),('bfloat16','bfloat16_t')]:
                args=f"{ctype}, uint16_t, Int<{c['Threads']}>, Int<{c['TileM']}>, Int<{c['TileK']}>, Int<{c['TileP']}>, Int<{c['Stages']}>, Int<3>, Int<{group}>, config::QuantMapModeEnum::Vectorized, config::AccumulationModeEnum::High, config::DecompositionModeEnum::StreamK, Int<{c['G2STiledCopySizeS']}>, Int<{c['MmaPrmK']}>"
                name=f"C_{candidate.native_id}_{group}_{dtype}"
                lines.extend([f'using {name} = config::GemmConfig<{args}>;',
                    f'static_assert({name}::kSmemSize == config::GemmConfig<{args.replace("::High","::Mixed")}>::kSmemSize);',
                    f'std::printf("{candidate.native_id} {group} {dtype} %d\\n", {name}::kSmemSize);'])
    lines.append('return 0; }');cpp=root/'resources.cu';cpp.write_text('\n'.join(lines)+'\n');binary=root/'resources-cpu'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
    command=[a.nvcc,'-std=c++17','-ccbin',a.compiler_bindir,'-I'+str(cutlass/'include'),'-I'+str(source/'flute/csrc'),str(cpp),'-o',str(binary)]
    with (root/'compile.log').open('wb') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
    raw=subprocess.check_output([str(binary)],env=env,text=True,timeout=10);(root/'stdout.txt').write_text(raw);resources={}
    for line in raw.splitlines():
        tid,group,dtype,size=line.split();resources.setdefault('3',{}).setdefault(group,{}).setdefault(tid,{})[dtype]=int(size)
    result={'kind':'original Config::kSmemSize CPU sizeof','source_revision':'9eb83a12d56949bbe7fe9c836ba97a67bd1e3761',
        'config_header_sha256':hashlib.sha256(header.read_bytes()).hexdigest(),'cutlass_tensor_header_sha256':hashlib.sha256((cutlass/'include/cute/tensor.hpp').read_bytes()).hexdigest(),
        'probe_source_sha256':hashlib.sha256(cpp.read_bytes()).hexdigest(),'mixed_and_high_same_storage':True,
        'template_resources':resources,'GPU_kernel_calls':0}
    (root/'resources.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
