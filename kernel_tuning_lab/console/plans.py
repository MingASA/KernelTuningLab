"""Native-schema plans and immutable execution inputs; no uploaded imports here."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

from ..asset_experiments import AssetExperiment
from ..research import expand_research
from .projects import verify_project
from .store import beneath, digest, read_json, write_json


def plan_spec(raw, action, limits):
    if action not in ('run', 'check', 'research'):
        raise ValueError('choose run, check or research')
    if action == 'research':
        expanded = expand_research({**raw, 'max_jobs': min(raw.get('max_jobs', 1024), limits.get('max_jobs', 128))})
        specs = [AssetExperiment.from_dict(j['experiment']) for j in expanded['jobs']]
    else:
        specs = [AssetExperiment.from_dict(raw)]
    if any(spec.validation.trust_claims for spec in specs):
        raise ValueError('Web experiments require independent correctness: validation.trust_claims=false')
    count = sum(len(s.kernels) * len(s.shapes) * len(s.algorithms) for s in specs)
    observations = 0 if action == 'check' else sum(len(s.kernels) * len(s.shapes) * len(s.algorithms)
                                                      * s.budget.max_evaluations for s in specs)
    validation = sum(s.validation.max_configs for s in specs)
    wall = sum(2 * s.preflight_seconds + s.validation.max_wall_time_s + 5 +
               (0 if action == 'check' else len(s.kernels) * len(s.shapes) * len(s.algorithms)
                * s.budget.max_wall_time_s) for s in specs)
    if len(specs) > limits.get('max_jobs', 128) or count > limits.get('max_jobs', 128):
        raise ValueError('expanded task count exceeds the server limit')
    if observations + validation > limits.get('max_observations', 100000):
        raise ValueError('expanded measurement budget exceeds the server limit')
    if wall > limits.get('max_wall_time_s', 86400):
        raise ValueError('serial wall allowance exceeds the server limit')
    return {'execution_jobs': len(specs), 'search_trials': count,
            'max_observations': observations, 'max_validation_evaluations': validation,
            'max_physical_measurements': observations + validation, 'serial_wall_allowance_s': wall,
            'workspace_memory_bytes': max(s.workspace_memory_limit_bytes for s in specs),
            'protocols': [{'warmup': s.warmup, 'repetitions': s.repetitions, 'timing_mode': s.timing_mode,
                           'cuda_graph_batch_size': s.cuda_graph_batch_size} for s in specs]}, specs


def source_snapshot(target):
    import kernel_tuning_lab
    source = Path(kernel_tuning_lab.__file__).parent
    files = []
    for file in sorted(source.rglob('*')):
        if file.is_symlink():
            raise ValueError('platform source snapshot cannot contain symbolic links')
        if not file.is_file() or '__pycache__' in file.parts or file.suffix in ('.pyc', '.pyo'):
            continue
        relative = file.relative_to(source)
        output = target / 'kernel_tuning_lab' / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, output)
        files.append({'file': str(Path('kernel_tuning_lab') / relative),
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
    write_json(target / 'SOURCE.json', {'files': files, 'sha256': digest(files)})
    return digest(files)


def freeze_plan(store, job):
    env = store.environment(job['environment'])
    planned, _ = plan_spec(job['spec'], job.get('action', 'run'), store.config())
    source = store.directory(job['id']) / 'plans' / str(job['revision'])
    if source.exists():
        # Never overwrite an inspected input version.
        raise ValueError('this revision was already frozen; edit or duplicate the draft')
    source.mkdir(parents=True, mode=0o700)
    platform_sha = source_snapshot(source / 'platform')
    uploaded = None
    if job.get('project'):
        uploaded = verify_project(store, job['project']['id'], job['project']['version'])
        original = store.root / 'projects' / uploaded['id'] / uploaded['version'] / 'files'
        shutil.copytree(original, source / 'project')
        write_json(source / 'project-manifest.json', uploaded)
    raw = dict(job['spec'])
    plugins = []
    for ref in [*env.get('plugins', []), *raw.get('plugins', [])]:
        if ref in env.get('plugins', []):
            plugins.append(ref)
        else:
            if uploaded is None or not isinstance(ref, str) or not ref.endswith('.py'):
                raise ValueError('plugins must be registered in this environment or part of the trusted upload')
            plugin = beneath(source / 'project', ref)
            if not plugin.is_file():
                raise ValueError('uploaded plugin file was not found')
            plugins.append(str(plugin))
    raw['plugins'] = list(dict.fromkeys(plugins))
    if env['device'] == 'cuda':
        allocation = read_json(store.root / 'worker/resource-context.json', {})
        if any(key in allocation for key in ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID')):
            raise ValueError('server inherited a GPU allocation; the idle-GPU guard cannot override it')
        gpu = job.get('gpu')
        if not isinstance(gpu, str) or not gpu.startswith('GPU-'):
            raise ValueError('select a physical GPU UUID before prechecking a GPU experiment')
        allowed = env.get('gpu_uuids', [])
        if allowed and gpu not in allowed:
            raise ValueError('GPU is not authorized for this execution environment')
    elif job.get('gpu') not in (None, '', 'cpu'):
        raise ValueError('CPU environments cannot claim a GPU')
    # The original request and the exact resolved native specification are separate.
    write_json(source / 'request.json', job['spec'])
    write_json(source / 'experiment.json', raw)
    write_json(source / 'environment.json', env)
    plan = {'revision': job['revision'], 'action': job.get('action', 'run'), 'environment': env,
            'workspace_root': str(store.root),
            'resource_context': read_json(store.root / 'worker/resource-context.json', {}),
            'device': env['device'], 'gpu': job.get('gpu'), 'planned': planned,
            'platform_sha256': platform_sha, 'request_sha256': digest(job['spec']),
            'spec_sha256': digest(raw), 'environment_sha256': digest(env),
            'project_version': uploaded['version'] if uploaded else None,
            'uploaded_code': bool(uploaded and any(f['path'].endswith('.py') for f in uploaded['files'])),
            'project_trust': uploaded['trusted'] if uploaded else None}
    plan['sha256'] = digest(plan)
    write_json(source / 'plan.json', plan)
    return source, plan


def verify_plan(source):
    source = Path(source)
    plan = read_json(source / 'plan.json')
    if digest({k: v for k, v in plan.items() if k != 'sha256'}) != plan['sha256']:
        raise ValueError('frozen plan identity changed')
    if digest(read_json(source / 'experiment.json')) != plan['spec_sha256']:
        raise ValueError('frozen experiment identity changed')
    if digest(read_json(source / 'request.json')) != plan['request_sha256']:
        raise ValueError('frozen original request changed')
    if digest(read_json(source / 'environment.json')) != plan['environment_sha256']:
        raise ValueError('frozen environment identity changed')
    manifest = read_json(source / 'platform/SOURCE.json')
    if manifest['sha256'] != plan['platform_sha256'] or digest(manifest['files']) != manifest['sha256']:
        raise ValueError('platform source manifest changed')
    for row in manifest['files']:
        if hashlib.sha256(beneath(source / 'platform', row['file']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('frozen platform source changed')
    uploaded = read_json(source / 'project-manifest.json')
    if uploaded:
        for row in uploaded['files']:
            if hashlib.sha256(beneath(source / 'project', row['path']).read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('frozen plugin identity changed')
    return plan


def templates():
    cpu = {'schema_version': 2, 'name': 'CPU GEMM', 'operator': 'gemm@1.0.0', 'dtype': 'float32',
           'kernels': [{'asset': 'python/gemm@1.0.0'}], 'algorithms': ['random@1.0.0', 'grid@1.0.0'],
           'shapes': [{'M': 3, 'N': 5, 'K': 7}], 'warmup': 1, 'repetitions': 5, 'seed': 17,
           'budget': {'max_evaluations': 4, 'max_wall_time_s': 15},
           'validation': {'max_configs': 1, 'max_cases': 0, 'max_wall_time_s': 15, 'trust_claims': False}}
    moe = {**cpu, 'name': 'Production MoE', 'operator': 'production/routed_moe_silu@1.0.0',
           'kernels': [{'asset': 'production/vllm085_moe_explicit@1.0.0'}],
           'algorithms': ['random@1.0.0', {'asset': 'smac_rf@1.0.0', 'options': {'initial_design_size': 2}}],
           'parameters': {'route_distribution': 'balanced'}, 'dtype': 'float16',
           'shapes': [{'M': 17, 'H': 128, 'I': 256, 'E': 4, 'TOP_K': 2}],
           'timing_mode': 'cuda_graph', 'cuda_graph_batch_size': 16,
           'budget': {'max_evaluations': 4, 'max_wall_time_s': 90}}
    return [{'id': 'cpu-gemm', 'name': 'CPU · GEMM', 'spec': cpu},
            {'id': 'production-moe', 'name': 'H800 · MoE', 'profile': 'moe-cu124', 'spec': moe}]
