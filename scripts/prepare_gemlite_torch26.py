"""Copy a verified GemLite package into a new private optional-E8M0 profile.

This does not install/upgrade Torch/Triton, import GemLite, access a GPU or edit
the original component. Only three eager dtype registrations are conditional.
"""
import argparse
import json
import shutil
from pathlib import Path
from kernel_tuning_lab.production import native_sources as sources,gemlite_compat


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--raw-a16',action='store_true',help='omit optional Dynamo/custom-op registration while preserving native function bodies')
    args=parser.parse_args();source=Path(args.source).resolve();target=Path(args.output).resolve()
    if target.exists():raise ValueError('use a new private component; never overwrite an existing environment')
    files=[name for name in sources.manifest()['gemlite']['files'] if name.startswith('gemlite/') and name.endswith('.py')]
    identity=sources.native_component(source,'gemlite',files)
    target.mkdir(parents=True,mode=0o700);shutil.copytree(source/'gemlite',target/'gemlite',ignore=shutil.ignore_patterns('__pycache__'))
    (target/'gemlite/dtypes.py').write_bytes(gemlite_compat.optional_e8m0_source())
    interop={}
    if args.raw_a16:
        for relative in gemlite_compat.RAW_FILES:
            (target/relative).write_bytes(gemlite_compat.raw_a16_source(relative))
            interop[relative]=gemlite_compat.verify_runtime_source(target,relative)
    metadata={'source_identity':identity,'patch':gemlite_compat.verify_dtype_source(target),'GPU_execution':False,
        'raw_native_A16_interop':interop,
        'scope':'GemLite W2/W4 with A16 FP16/BF16 and FP32 dot accumulator; no E8M0/MX profile claim'}
    (target/'component-patch.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata))


if __name__=='__main__':main()
