"""CPU-build a registration-only subset of unchanged vLLM0.8.5 Marlin sources.

No GPU is executed. Uses an existing CUDA toolkit/compiler and the current
Torch2.6 environment. The result is a native SOURCE BUILD, not the stock wheel;
all numerical bodies and internal template selectors remain upstream bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
PINS=ROOT/'kernel_tuning_lab/production/upstream/NATIVE_SOURCES.json'
SCHEMA='''// Registration only; all numerical code remains unchanged vLLM0.8.5.
#include <torch/library.h>
TORCH_LIBRARY_FRAGMENT(_C, ops) {
 at::Tag stride_tag = at::Tag::needs_fixed_stride_order;
 ops.def("gptq_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, Tensor b_zeros, Tensor g_idx, Tensor perm, Tensor workspace, int b_q_type, SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, bool has_zp, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor", {stride_tag});
 ops.def("gptq_marlin_repack(Tensor b_q_weight, Tensor perm, SymInt size_k, SymInt size_n, int num_bits) -> Tensor");
}
'''


def prepare_source(archive,output):
    archive,output=Path(archive),Path(output);pins=json.loads(PINS.read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=pins['source_archive_sha256']:
        raise ValueError('source archive differs from pinned vLLM0.8.5 sdist')
    selected={item['file']:item for item in pins['files']};found=set()
    output.mkdir(mode=0o700,parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as source:
        for member in source:
            if not member.isfile():continue
            name=str(Path(*Path(member.name).parts[1:]))
            if name not in selected:continue
            blob=source.extractfile(member).read();assert hashlib.sha256(blob).hexdigest()==selected[name]['sha256']
            target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=blob:raise ValueError('refuse changed existing numerical source')
            target.write_bytes(blob);found.add(name)
    if found!=set(selected):raise ValueError('pinned source member missing')
    return pins


def build(archive,output,*,cuda_home,cxx,arch):
    out=Path(output);source=out/'source';pins=prepare_source(archive,source);start=time.perf_counter()
    if not Path(cxx).is_file() or not (Path(cuda_home)/'bin/nvcc').is_file():raise ValueError('existing C++/CUDA compiler required; no system installation attempted')
    if not arch.startswith('sm_') or not arch[3:].isdigit():raise ValueError('explicit target architecture sm_XX required')
    digits=arch[3:];cuda_arch=digits[:-1]+'.'+digits[-1]
    # These variables affect this one build process, not a shared shell/config.
    os.environ.update(CUDA_VISIBLE_DEVICES='',CUDA_HOME=str(cuda_home),CXX=str(cxx),
        TORCH_CUDA_ARCH_LIST=cuda_arch,MAX_JOBS='1')
    import torch
    if not torch.__version__.startswith('2.6.0'):raise ValueError('this source-build profile is validated with Torch2.6.0')
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only builder must not initialize CUDA')
    from torch.utils.cpp_extension import load
    bindings=source/'schema_subset.cpp';bindings.write_text(SCHEMA);directory=out/'build';directory.mkdir(parents=True,exist_ok=True)
    flags=['-O3','-DTORCH_EXTENSION_NAME=_C'];cuda_flags=[*flags,'-ccbin='+str(cxx),'--expt-relaxed-constexpr',
        '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__']
    library=load(name='_C',sources=[str(bindings),str(source/'csrc/quantization/gptq_marlin/gptq_marlin.cu'),
        str(source/'csrc/quantization/gptq_marlin/gptq_marlin_repack.cu')],extra_cflags=flags,extra_cuda_cflags=cuda_flags,
        extra_include_paths=[str(source/'csrc'),str(source/'csrc/quantization/gptq_marlin')],
        build_directory=str(directory),verbose=True,with_cuda=True,is_python_module=False)
    if torch.cuda.is_initialized():raise RuntimeError('unexpected GPU initialization during CPU build')
    target=out/'components/vllm/_C.abi3.so';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(library,target)
    result={'version':'0.8.5','kind':'native_source_subset','required_torch':'2.6.0',
        'source_archive_sha256':pins['source_archive_sha256'],'source_files':pins['files'],
        'binary_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'binary_bytes':target.stat().st_size,
        'compiled_architectures':[arch],'registration':'two native schemas only; numerical source and internal template selectors unchanged',
        'registration_source_sha256':hashlib.sha256(SCHEMA.encode()).hexdigest(),
        'compiler':subprocess.check_output([str(cxx),'--version'],text=True).splitlines()[0],
        'cuda_toolkit':subprocess.check_output([str(Path(cuda_home)/'bin/nvcc'),'--version'],text=True),
        'compile_flags':flags,'cuda_flags':cuda_flags,'torch':torch.__version__,'build_wall_s':time.perf_counter()-start,
        'gpu_validation':'not_executed_by_CPU_builder','python_abi':'Torch dispatcher library; not a Python module entrypoint'}
    (out/'components/component-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-sdist',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--cuda-home',required=True);parser.add_argument('--cxx',required=True);parser.add_argument('--arch',required=True)
    args=parser.parse_args();result=build(args.source_sdist,args.output,cuda_home=args.cuda_home,cxx=args.cxx,arch=args.arch)
    print(json.dumps({'status':'built_cpu_only','binary_sha256':result['binary_sha256'],'build_wall_s':result['build_wall_s']}))
