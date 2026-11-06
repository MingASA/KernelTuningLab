"""Create a new private environment from a core wheel/source and a fixed profile."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernel_tuning_lab.portable import profiles, sha256


def plan(args):
    profile = profiles()[args.profile]
    probe = subprocess.check_output([args.python, '-c', 'import json,sys;print(json.dumps(list(sys.version_info[:2])))'], text=True, timeout=15)
    minor = tuple(json.loads(probe));expected = tuple(profile['python'])
    if (minor < expected if profile.get('python_is_minimum') else minor != expected):
        raise ValueError(f'{args.profile} needs Python {expected[0]}.{expected[1]}; select --python explicitly')
    target = Path(args.env).resolve()
    if target.exists():raise ValueError('create a new private environment; do not overwrite or copy an old venv')
    python = target/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    core = str(Path(args.wheel).resolve()) if args.wheel else str(ROOT)
    pip = [str(python), '-m', 'pip', 'install']
    source_options = []
    if args.wheelhouse:
        source_options = ['--no-index', '--find-links', str(Path(args.wheelhouse).resolve())]
    elif profile.get('extra_index_url'):
        source_options = ['--extra-index-url', profile['extra_index_url']]
    constraints = ['-c', str(Path(args.constraints).resolve())] if args.constraints else []
    commands = [[args.python, '-m', 'venv', str(target)], pip+source_options+['--no-deps', core]]
    if profile['packages']:commands.append(pip+source_options+constraints+profile['packages'])
    return target, python, commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', required=True)
    parser.add_argument('--profile', choices=tuple(profiles()), default='core')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--wheel', help='verified core release wheel for offline installation')
    parser.add_argument('--wheel-sha256', help='expected core wheel SHA256 from the release receipt')
    parser.add_argument('--wheelhouse', help='private matching wheel cache; disables network package resolution')
    parser.add_argument('--wheelhouse-manifest', help='JSON files[{file,sha256}] for the selected private wheelhouse')
    parser.add_argument('--constraints', help='matching full environment constraints, excluding the core package')
    parser.add_argument('--components', help='restored FLUTE package root or fixed vLLM0.8.5 component directory')
    parser.add_argument('--flute-manifest', help='binary-bound High build/resource manifest')
    parser.add_argument('--raw-gemlite', action='store_true', help='create the audited Torch2.6 A16 overlay in the new environment')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.wheel_sha256 and (not args.wheel or sha256(args.wheel) != args.wheel_sha256):
        raise ValueError('core wheel SHA256 mismatch')
    if args.raw_gemlite and args.profile != 'gemlite-cu124':
        raise ValueError('--raw-gemlite belongs to the GemLite Torch2.6 profile')
    target, python, commands = plan(args)
    if args.profile == 'flute-cu124' and not (args.components and args.flute_manifest):
        raise ValueError('FLUTE requires restored components and their exact build manifest, or a new private native build')
    if args.profile == 'moe-cu124' and not args.components:
        raise ValueError('MoE requires the fixed original official component directory')
    if args.wheelhouse_manifest:
        if not args.wheelhouse:raise ValueError('--wheelhouse-manifest requires --wheelhouse')
        for row in json.loads(Path(args.wheelhouse_manifest).read_text())['files']:
            path = Path(args.wheelhouse)/row['file']
            if not path.resolve().is_relative_to(Path(args.wheelhouse).resolve()) or sha256(path) != row['sha256']:
                raise ValueError('wheelhouse SHA256 mismatch: '+row['file'])
    if args.dry_run:
        print(json.dumps({'commands': commands, 'environment': str(target), 'GPU_work': False}, indent=2));return
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONNOUSERSITE': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1'}
    for key in ('PYTHONPATH', 'PYTHONHOME'):env.pop(key, None)
    for command in commands:
        subprocess.run(command, env=env, check=True, timeout=1800)
    overrides = {}
    if args.raw_gemlite:
        source = subprocess.check_output([str(python), '-c', 'import importlib.util;from pathlib import Path;print(Path(importlib.util.find_spec("gemlite").origin).parent.parent)'], env=env, text=True).strip()
        overlay = target/'native-a16'
        subprocess.run([str(python), str(ROOT/'scripts/prepare_gemlite_torch26.py'), '--source', source, '--output', str(overlay), '--raw-a16'], env={**env, 'PYTHONPATH': str(ROOT)}, check=True, timeout=60)
        overrides['PYTHONPATH'] = str(overlay)
    if args.profile == 'moe-cu124':overrides['KT_VLLM085_MOE_COMPONENTS'] = str(Path(args.components).resolve())
    if args.profile == 'flute-cu124':
        overrides.update(PYTHONPATH=str(Path(args.components).resolve()), KT_FLUTE_BUILD_MANIFEST=str(Path(args.flute_manifest).resolve()))
    qualification = subprocess.check_output([str(python), '-m', 'kernel_tuning_lab', 'doctor'], env={**env, **overrides}, text=True, timeout=45)
    component_evidence = None
    if args.profile in ('moe-cu124', 'flute-cu124'):
        code = ('from kernel_tuning_lab.production.vllm_moe_compat import execution_identity;identity=execution_identity()'
                if args.profile == 'moe-cu124' else
                'from kernel_tuning_lab.production.flute import native_api;identity=native_api()[3]')
        code += ';import json;print(json.dumps(identity))'
        raw = subprocess.check_output([str(python), '-c', code], env={**env, **overrides}, text=True, timeout=90)
        component_evidence = json.loads(raw.splitlines()[-1])
    runtime = {'profile': args.profile, 'python': str(python), 'commands': commands, 'environment': overrides,
               'doctor': json.loads(qualification), 'component_evidence': component_evidence,
               'core_wheel_sha256': sha256(args.wheel) if args.wheel else None,
               'constraints_sha256': sha256(args.constraints) if args.constraints else None,
               'GPU_validation': 'not performed by bootstrap'}
    (target/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
    activate = 'source '+shlex.quote(str(target/'bin/activate'))+'\n'
    for key, value in overrides.items():activate += 'export '+key+'='+shlex.quote(value)+'\n'
    (target/'activate-ktlab.sh').write_text(activate)
    print(json.dumps({'status': 'installed_CPU_diagnosed', 'environment': str(target), 'activate': str(target/'activate-ktlab.sh'), 'GPU_validation': 'pending admitted target-runtime acceptance'}, indent=2))


if __name__ == '__main__':
    try:main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr);raise SystemExit(2)
