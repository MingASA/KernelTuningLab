"""A serial, independently supervised queue worker around the existing CLI/Runner."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

from ..gpu_admission import proc_identity, collect_owned, stop_owned, signal_owned
from .isolation import clean_environment, apply_limits, protect_supervisor, restriction_capabilities, bubblewrap_command
from .plans import freeze_plan, verify_plan
from .store import Store, digest, read_json, write_json
from .telemetry import harvest

BOOTSTRAP = ("import os,sys,runpy; p=sys.argv.pop(1); "
             "sys.path[:0]=[p]+[x for x in os.environ.get('PYTHONPATH','').split(os.pathsep) if x]; "
             "runpy.run_module('kernel_tuning_lab.console.worker',run_name='__main__')")


def command(python, source, *arguments):
    return [python, '-I', '-c', BOOTSTRAP, str(source), *map(str, arguments)]


def environment_identity():
    versions = {}
    metadata = {}
    for name in ('torch', 'triton', 'numpy', 'smac', 'kernel_tuner', 'gemlite', 'flute-kernel', 'ConfigSpace',
                 'scipy', 'scikit-learn', 'pandas', 'pymoo', 'jaxtyping', 'wadler-lindig', 'transformers'):
        try:
            versions[name] = importlib.metadata.version(name)
            distribution = importlib.metadata.distribution(name)
            metadata[name] = hashlib.sha256(((distribution.read_text('METADATA') or '') +
                                             (distribution.read_text('RECORD') or '')).encode()).hexdigest()
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    binary = Path(sys.executable).resolve()
    import kernel_tuning_lab
    return {'python': sys.executable, 'python_realpath': str(binary), 'python_version': sys.version,
            'python_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'packages': versions,
            'distribution_metadata_sha256': metadata,
            'framework_version': kernel_tuning_lab.__version__,
            'framework_import_root': str(Path(kernel_tuning_lab.__file__).parent),
            'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'host_id': host_identity(),
            'kernel': platform.release(), 'architecture': platform.machine(),
            'driver_identity': hashlib.sha256(Path('/proc/driver/nvidia/version').read_bytes()).hexdigest()
                if Path('/proc/driver/nvidia/version').is_file() else None}


def host_identity():
    machine = Path('/etc/machine-id')
    return hashlib.sha256((platform.node() + ':' + (machine.read_text().strip() if machine.exists() else '')).encode()).hexdigest()


def catalogue(plugins):
    from ..catalog import get_catalog
    from ..plugins import load_plugin, discover_entry_points
    catalog = get_catalog()
    discover_entry_points(catalog)
    for plugin in plugins:
        load_plugin(plugin, catalog)
    return catalog


def describe(catalog):
    return [{'ref': a.ref, 'fingerprint': a.fingerprint, **a.manifest()} for a in catalog.list()]


def inspect_native(plan_dir, plan, runtime):
    from ..asset_experiments import AssetExperiment
    from .plans import plan_spec
    raw = read_json(plan_dir / 'experiment.json')
    _, specs = plan_spec(raw, plan['action'], {})
    catalog = catalogue(raw.get('plugins', []))
    spaces, selected, components = [], set(), {}
    from ..portable import profiles
    profile = profiles().get(plan['environment'].get('profile', 'core'))
    if profile:
        expected_python = tuple(profile['python'])
        actual_python = sys.version_info[:2]
        if (actual_python < expected_python if profile.get('python_is_minimum') else actual_python != expected_python):
            raise ValueError('Python minor does not match the registered runtime profile')
        for requirement in profile['packages']:
            name, _, version = requirement.partition('==')
            if version and runtime['packages'].get(name) != version:
                raise ValueError(f'{requirement} required by this profile; found {runtime["packages"].get(name)}')
    for spec in specs:
        if plan['device'] == 'cpu' and spec.timing_mode == 'cuda_graph':
            raise ValueError('CUDA Graph timing requires a GPU execution environment')
        operator = catalog.resolve('operator', spec.operator)
        if spec.dtype not in operator.dtypes:
            raise ValueError('dtype is not supported by the selected operator')
        if any(set(s.dims) != set(operator.shape_dims) for s in spec.shapes):
            raise ValueError('shape dimensions do not match the selected operator')
        selected.add(('operator', operator.ref))
        for choice in spec.kernels:
            kernel = catalog.resolve('kernel', choice.asset)
            if kernel.runtime not in ('python', 'cuda'):
                raise ValueError('this Worker supports the native Python and CUDA execution paths')
            if catalog.resolve('operator', kernel.operator).ref != operator.ref:
                raise ValueError('Kernel implements a different operator')
            if (kernel.runtime == 'python') != (plan['device'] == 'cpu'):
                raise ValueError('Kernel runtime is incompatible with this execution environment')
            if kernel.dtypes and spec.dtype not in kernel.dtypes:
                raise ValueError('dtype is not supported by the selected Kernel')
            for key, values in choice.space.items():
                if key not in kernel.space.parameters or any(v not in kernel.space.domains[key] for v in values):
                    raise ValueError('space restriction is outside the native Kernel domain')
            selected.add(('kernel', kernel.ref))
            if kernel.execution_identity:
                from ..assets import _jsonable
                components[kernel.ref] = _jsonable(kernel.execution_identity())
            from ..asset_space import build_effective_space, check_algorithm_compatibility
            from ..assets import SpaceContext
            from ..models import HardwareSpec, OperatorSpec
            from ..acceptance import check_operator_request, inspect_kernel_entrypoint, inspect_signature
            from ..kernel_runtime import AssetKernelAdapter
            if plan['device'] == 'cpu':
                hardware = AssetKernelAdapter(kernel, operator, force_correctness=True).probe_hardware()
            elif spec.hardware or plan['environment'].get('hardware_profile'):
                from ..hardware import get_hardware_profile
                hardware = get_hardware_profile(spec.hardware or plan['environment']['hardware_profile'])
            else:
                hardware = HardwareSpec('GPU pending physical admission', 'unknown', metadata={'probe_pending': True})
            for shape in spec.shapes:
                parameters = check_operator_request(operator, spec.dtype, spec.parameters, shape)
                bound = OperatorSpec(operator.id, kind=operator.id, implementation=kernel.ref,
                    dtype=spec.dtype, parameters=parameters, correctness_baseline=operator.ref, version=operator.version,
                    metadata={'operator_asset_fingerprint': operator.fingerprint, 'kernel_asset_fingerprint': kernel.fingerprint})
                context = SpaceContext(hardware, bound, shape, environment=runtime['packages'],
                                       workspace_memory_limit_bytes=spec.workspace_memory_limit_bytes)
                space = build_effective_space(kernel, hardware, bound, shape, choice.space, context=context)
                inspect_kernel_entrypoint(kernel, space)
                for algorithm in spec.algorithms:
                    definition = catalog.resolve('algorithm', algorithm.asset)
                    inspect_signature(definition.factory, seed=spec.seed, **algorithm.options)
                    compatible = check_algorithm_compatibility(definition, space)
                    if not compatible.valid:
                        raise ValueError('; '.join(compatible.reasons))
                spaces.append({'kernel': kernel.ref, 'shape': dict(shape.dims), 'declared': kernel.space.to_dict(),
                    'restrictions': dict(choice.space), 'resolved_domains': space.domains,
                    'cardinality_upper_bound': space.cardinality, 'resolution': space.resolution,
                    'evidence': 'CPU domain resolution; native runtime preparation and numerical checks follow admission'})
        from ..acceptance import check_operator_request
        for shape in spec.shapes:
            check_operator_request(operator, spec.dtype, spec.parameters, shape)
        for choice in spec.algorithms:
            catalog.resolve('algorithm', choice.asset)
            selected.add(('algorithm', choice.asset))
            required = {'smac_rf@1.0.0': ('smac', 'ConfigSpace'), 'kernel_tuner_de@1.0.0': ('kernel_tuner',)}.get(choice.asset, ())
            if any(runtime['packages'].get(name) is None for name in required):
                raise ValueError('selected optimizer is not installed in this environment: ' + ', '.join(required))
    if plan['device'] == 'cuda':
        if not runtime['packages'].get('torch') or not runtime['packages'].get('triton'):
            raise ValueError('GPU environment is missing Torch or Triton')
        if not plan['environment'].get('gpu_lock'):
            raise ValueError('GPU environments require a configured cooperative lock path')
    return {'status': 'passed', 'runtime': runtime, 'runtime_components': components, 'effective_spaces': spaces,
            'assets': [a for a in describe(catalog) if (a['kind'], a['ref']) in selected],
            'warnings': ['GPU numerical validation runs after resource admission'] if plan['device'] == 'cuda' else []}


def execute(plan_dir, attempt, operation, expected_sha, *, namespace_inner=False):
    """Only the trusted launcher reads a plan. Plugins run after filesystem restriction."""
    plan_dir, attempt = Path(plan_dir), Path(attempt)
    plan = verify_plan(plan_dir)
    if plan['sha256'] != expected_sha:
        raise ValueError('plan differs from the durably approved queue identity')
    env = plan['environment']
    output = plan_dir.parents[1] / 'output'
    output.mkdir(exist_ok=True)
    attempt.mkdir(parents=True, exist_ok=True)
    runtime = environment_identity()
    if operation == 'run' and env['device'] == 'cuda':
        from .. import gpu_admission as guard
        lock_path = Path(env['gpu_lock']).expanduser()
        if lock_path.is_symlink():
            raise ValueError('GPU cooperative lock cannot be a symlink')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {'status': 'resource_unavailable', 'reason': 'cooperative GPU lock is busy', 'exit_code': 75}
            args = argparse.Namespace(gpu=plan['gpu'], expected_model=env.get('expected_model', ''),
                receipt=str(attempt / 'resource-guard.json'), timeout=plan['planned']['serial_wall_allowance_s'] + 30,
                ready_samples=3, interval=1, select_only=False, cwd=str(attempt),
                command=command(env['python'], plan_dir / 'platform', 'execute', '--plan', plan_dir,
                                '--attempt', attempt / 'native', '--operation', 'native', '--plan-sha', plan['sha256']))
            code = guard.run(args)
        native = read_json(attempt / 'native/result.json', {})
        return {**native, 'status': ('resource_unavailable' if code == 75 and not native else native.get('status', 'failed')),
                'exit_code': code, 'guard': read_json(attempt / 'resource-guard.json', {})}
    capabilities = restriction_capabilities() if not namespace_inner else {}
    preferred = env.get('isolation', 'auto')
    use_bwrap = not namespace_inner and (preferred == 'bubblewrap' or
                 (preferred == 'auto' and capabilities.get('bubblewrap')))
    if use_bwrap:
        native_command = command(env['python'], plan_dir / 'platform', 'execute', '--plan', plan_dir,
            '--attempt', attempt, '--operation', operation, '--plan-sha', expected_sha, '--namespace-inner')
        args = bubblewrap_command(native_command, attempt, env, source=plan_dir, output=output, runtime=runtime,
                                  gpu=plan['gpu'] if operation == 'native' and plan['device'] == 'cuda' else None)
        run = subprocess.run(args, stdin=subprocess.DEVNULL, check=False)
        return read_json(attempt / 'result.json', {'status': 'failed', 'exit_code': run.returncode,
                                                   'error': 'restricted Worker exited without a result'})
    # Freeze exact runtime at inspect time, and refuse a changed installation before execution.
    approved = read_json(plan_dir / 'precheck.json', {})
    if operation in ('run', 'native') and approved.get('runtime') != runtime:
        raise ValueError('execution environment changed since precheck; duplicate and precheck again')
    os.chdir(attempt)
    os.environ['PYTHONPATH'] = os.pathsep.join([str(plan_dir / 'platform'), str(plan_dir / 'project'),
                                               env.get('env', {}).get('PYTHONPATH', '')]).rstrip(os.pathsep)
    sys.path.insert(1, str(plan_dir / 'project'))
    restrictions = apply_limits(attempt, env, source=plan_dir, output=output,
                                uploaded=plan.get('uploaded_code', bool(plan.get('project_version'))),
                                hard_wall_s=plan['planned']['serial_wall_allowance_s'], namespace_inner=namespace_inner)
    write_json(attempt / 'isolation.json', restrictions)
    if operation == 'inspect':
        result = inspect_native(plan_dir, plan, runtime)
        result['isolation'] = restrictions
        return result
    if approved.get('isolation', {}).get('mode') != restrictions['mode'] or (
            restrictions['mode'] == 'bubblewrap' and
            approved['isolation'].get('binary_sha256') != restrictions['binary_sha256']):
        raise ValueError('execution restriction changed since precheck; create a new precheck')
    fresh = inspect_native(plan_dir, plan, runtime)
    if digest(fresh['runtime_components']) != digest(approved.get('runtime_components', {})):
        raise ValueError('native components changed since precheck; duplicate and precheck again')
    from ..asset_runner import execute_asset_experiment
    if plan['action'] == 'research':
        from ..research import run_research
        result = run_research(plan_dir / 'experiment.json', output)
    else:
        result = execute_asset_experiment(plan_dir / 'experiment.json', output,
                                          check_only=plan['action'] == 'check')
    return {**result, 'runtime': runtime, 'isolation': restrictions}


def alive(identity):
    if not identity or identity.get('host_id') != host_identity():
        return False
    current = proc_identity(identity['pid'])
    return bool(current is not None and current[1] == identity['start_ticks'])


def terminate(identity):
    if not alive(identity):
        return
    # pid + kernel start_ticks is required; same UID alone never grants ownership.
    class Owned:
        pid = identity['pid']
        def poll(self):
            return None if alive(identity) else 0
        def wait(self, timeout=None):
            deadline = time.monotonic() + (timeout or 0)
            while self.poll() is None and time.monotonic() < deadline:
                time.sleep(.025)
            if self.poll() is None:
                raise subprocess.TimeoutExpired('owned Worker', timeout)
            return 0
    known = {identity['pid']: identity['start_ticks']}
    collect_owned(identity['pid'], known)
    stop_owned(Owned(), known)


def probe_environments(store):
    """Only explicitly registered, administrator-trusted providers are imported."""
    platform = Path(__file__).parents[2]
    for env in store.config()['environments']:
        target = store.root / 'worker' / ('environment-' + env['id'] + '.json')
        cached = read_json(target, {})
        if cached.get('configuration_sha256') == digest(env):
            continue
        directory = store.root / 'worker' / ('probe-' + env['id'])
        directory.mkdir(exist_ok=True)
        command_args = command(env['python'], platform, 'probe', '--plugins', json.dumps(env.get('plugins', [])))
        try:
            run = subprocess.run(command_args, cwd=directory, env=clean_environment(directory, env, cpu_only=True),
                                 capture_output=True, text=True, timeout=30)
            value = json.loads(run.stdout) if run.returncode == 0 else {'status': 'unavailable', 'error': run.stderr[-3000:]}
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            value = {'status': 'unavailable', 'error': str(exc)}
        write_json(target, {**value, 'configuration_sha256': digest(env), 'at': time.time()})


def launch(store, job, operation):
    if operation == 'inspect':
        plan_dir, plan = freeze_plan(store, job)
        store.update(job['id'], plan_dir=str(plan_dir.relative_to(store.root)),
                     plan_sha=plan['sha256'], plan=plan)
    else:
        plan_dir = store.root / job['plan_dir']
        plan = verify_plan(plan_dir)
        if plan['sha256'] != job['plan_sha']:
            raise ValueError('frozen plan differs from the approved task identity')
        if plan.get('workspace_root', str(store.root)) != str(store.root):
            raise ValueError('workspace moved; duplicate this configuration and precheck in the restored environment')
        if digest(store.environment(job['environment'])) != plan['environment_sha256']:
            raise ValueError('registered environment changed since precheck')
    attempt = store.directory(job['id']) / 'attempts' / uuid.uuid4().hex
    attempt.mkdir(parents=True, mode=0o700)
    env = clean_environment(attempt, plan['environment'], cpu_only=operation == 'inspect' or plan['device'] == 'cpu')
    if operation != 'inspect' and plan['device'] == 'cuda':
        # The existing admission guard sets the fixed physical UUID itself.
        env.pop('CUDA_VISIBLE_DEVICES', None)
    with (attempt / 'launcher.log').open('wb') as log:
        process = subprocess.Popen(command(plan['environment']['python'], plan_dir / 'platform',
            'execute', '--plan', plan_dir, '--attempt', attempt, '--operation', operation, '--plan-sha', plan['sha256']),
            cwd=attempt, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            close_fds=True, start_new_session=True)
    identity = proc_identity(process.pid)
    if identity is None:
        process.wait(timeout=5)
        raise RuntimeError('Worker exited before its process identity was recorded')
    execution = {'pid': process.pid, 'start_ticks': identity[1], 'operation': operation, 'plan_sha': plan['sha256'],
                 'host_id': host_identity(),
                 'attempt': str(attempt.relative_to(store.root)), 'started': time.time(),
                 'deadline_s': 45 if operation == 'inspect' else plan['planned']['serial_wall_allowance_s'] + 45}
    write_json(attempt / 'process.json', execution)
    expected_state = 'prechecking' if operation == 'inspect' else 'queued'
    try:
        store.update(job['id'], expected={expected_state},
                     state='prechecking' if operation == 'inspect' else 'running', execution=execution,
                     emit=('execution', execution))
    except ValueError:
        # A cancelled request cannot be revived by a concurrent slow source freeze.
        terminate(execution)
        process.wait(timeout=5)
        return None
    write_json(attempt / 'launch-admitted.json', execution)
    return process


def settle(store, job):
    execution = job['execution']
    store.update(job['id'], last_execution=execution)
    job = store.get(job['id'])
    attempt = store.root / execution['attempt']
    result = read_json(attempt / 'result.json')
    if job['state'] == 'cancelling':
        terminate(execution)
        store.update(job['id'], expected={job['state']}, state='cancelled', execution=None, last_result=result)
    elif result is None:
        timed_out = execution.get('termination') == 'timeout'
        store.update(job['id'], expected={job['state']},
                     state=('invalid' if execution['operation'] == 'inspect' else 'failed') if timed_out else 'interrupted',
                     execution=None, error=('Worker exceeded its frozen wall allowance' if timed_out else
                     'Worker exited without a completion receipt; native records are retained'))
    elif execution['operation'] == 'inspect':
        if result.get('status') == 'passed':
            write_json(store.root / job['plan_dir'] / 'precheck.json', result)
        store.update(job['id'], expected={job['state']}, state='ready' if result.get('status') == 'passed' else 'invalid',
                     execution=None, precheck=result, error=result.get('error'))
    elif result.get('status') == 'resource_unavailable':
        store.update(job['id'], expected={job['state']}, state='queued', execution=None, resource_blocked=True,
                     last_result=result, error=result.get('reason') or result.get('guard', {}).get('reason', 'GPU resource unavailable'))
    else:
        state = 'completed' if result.get('status') == 'passed' else 'failed'
        if result.get('status') == 'interrupted':
            state = 'interrupted'
        store.update(job['id'], expected={job['state']}, state=state, execution=None, last_result=result,
                     error=result.get('error'))


def work(root, *, once=False):
    protect_supervisor()
    store = Store(root)
    lock = (store.root / 'worker/lease').open('a+b')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0
    running = True
    def stop(*_):
        nonlocal running
        running = False
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    write_json(store.root / 'worker/identity.json', {'pid': os.getpid(), 'start_ticks': proc_identity(os.getpid())[1], 'host_id': host_identity()})
    needs_probe = True
    children = {}
    try:
        while running:
            busy = False
            for job in store.queue():
                execution = job.get('execution')
                if execution:
                    busy = True
                    admission = store.root / execution['attempt'] / 'launch-admitted.json'
                    if alive(execution) and not admission.exists() and job['state'] != 'cancelling':
                        # Finish a durable handoff if the previous supervisor died between
                        # recording ownership and releasing its child's startup gate.
                        verify_plan(store.root / job['plan_dir'])
                        write_json(admission, execution)
                        store.event(job['id'], 'adopted', execution)
                    try:
                        harvest(store, job)
                    except (OSError, ValueError, TypeError) as exc:
                        store.event(job['id'], 'telemetry_error', {'message': str(exc)},
                                    dedupe='telemetry-error:' + job['id'] + ':' + digest(str(exc)))
                    if (job['state'] == 'cancelling' or time.time() - execution['started'] > execution['deadline_s']) and not execution.get('termination'):
                        execution = {**execution, 'termination': 'cancel' if job['state'] == 'cancelling' else 'timeout'}
                        store.update(job['id'], execution=execution)
                        terminate(execution)
                    process = children.get(job['id'])
                    if process is not None:
                        process.poll()
                    if not alive(execution):
                        if process is not None:
                            process.wait(timeout=5)
                            children.pop(job['id'], None)
                        try:
                            settle(store, store.get(job['id']))
                        except ValueError:
                            pass  # concurrent cancellation is reconciled on the next iteration
                    break
            if not busy:
                request = store.root / 'worker/probe-request.json'
                if request.exists() or needs_probe:
                    probe_environments(store)
                    request.unlink(missing_ok=True)
                    needs_probe = False
                for job in store.queue():
                    if job['state'] == 'prechecking' or (job['state'] == 'queued' and not job.get('resource_blocked')):
                        operation = 'inspect' if job['state'] == 'prechecking' else 'run'
                        try:
                            process = launch(store, job, operation)
                            if process is not None:
                                children[job['id']] = process
                        except Exception as exc:
                            current = store.get(job['id'])
                            if current['state'] not in ('cancelled', 'cancelling'):
                                store.update(job['id'], state='invalid' if operation == 'inspect' else 'failed',
                                             execution=None, error=f'{type(exc).__name__}: {exc}')
                        break
            if once:
                break
            time.sleep(.2)
    finally:
        # A graceful Worker stop interrupts only its own active task; API restart alone leaves it running.
        for job in store.jobs():
            if job.get('execution') and alive(job['execution']):
                terminate(job['execution'])
                harvest(store, job)
                if alive(job['execution']):
                    store.update(job['id'], state='cancelling',
                                 execution={**job['execution'], 'termination': 'cancel'},
                                 error='owned process has not exited; identity and resource ownership retained')
                else:
                    store.update(job['id'], state='interrupted', last_execution=job['execution'], execution=None,
                                 error='Worker stopped; resume uses the native measurement cache')
        for process in children.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        lock.close()
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    worker = sub.add_parser('work'); worker.add_argument('--workspace', required=True); worker.add_argument('--once', action='store_true')
    child = sub.add_parser('execute'); child.add_argument('--plan', required=True); child.add_argument('--attempt', required=True)
    child.add_argument('--plan-sha', required=True)
    child.add_argument('--namespace-inner', action='store_true')
    child.add_argument('--operation', choices=('inspect', 'run', 'native'), required=True)
    probe = sub.add_parser('probe'); probe.add_argument('--plugins', default='[]')
    args = parser.parse_args()
    if args.mode == 'work':
        return work(args.workspace, once=args.once)
    if args.mode == 'probe':
        print(json.dumps({'status': 'available', 'runtime': environment_identity(),
                          'assets': describe(catalogue(json.loads(args.plugins)))}))
        return 0
    attempt = Path(args.attempt)
    attempt.mkdir(parents=True, exist_ok=True)
    try:
        if args.operation != 'native' and not args.namespace_inner:
            deadline = time.monotonic() + 10
            while not (attempt / 'launch-admitted.json').is_file() and time.monotonic() < deadline:
                time.sleep(.025)
            admission = read_json(attempt / 'launch-admitted.json', {})
            own = proc_identity(os.getpid())
            if admission.get('pid') != os.getpid() or own is None or admission.get('start_ticks') != own[1]:
                raise RuntimeError('launch was not durably admitted by the queue Worker')
        result = execute(args.plan, attempt, args.operation, args.plan_sha, namespace_inner=args.namespace_inner)
    except BaseException as exc:
        result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'exit_code': 2}
        traceback.print_exc()
    write_json(attempt / 'result.json', result)
    return result.get('exit_code', 0 if result.get('status') == 'passed' else 2)


if __name__ == '__main__':
    raise SystemExit(main())
