"""Workbench contracts: no GPU/API calls; real CPU Runner and owned subprocesses."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import zipfile

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient

from kernel_tuning_lab.console.isolation import clean_environment, landlock_abi
from kernel_tuning_lab.console.plans import plan_spec, templates
from kernel_tuning_lab.console.projects import save_project, trust, verify_project
from kernel_tuning_lab.console.server import create_app
from kernel_tuning_lab.console.store import Store, read_json, write_json
from kernel_tuning_lab.console.worker import alive, host_identity, settle
from kernel_tuning_lab.console.auth import PASSWORD_KEY, initialize_password

TEST_PASSWORD = 'owned-test-password-42'


def password_app(root, **kwargs):
    initialize_password(Store(root), TEST_PASSWORD)
    return create_app(root, **kwargs)


def cpu_spec():
    return json.loads(json.dumps(templates()[0]['spec']))


@pytest.fixture
def client(tmp_path):
    app = password_app(tmp_path / 'workspace', auto_worker=False)
    with TestClient(app, base_url='http://127.0.0.1:18000', headers={'Origin': 'http://127.0.0.1:18000'}) as client:
        response = client.post('/api/session', json={'password': TEST_PASSWORD})
        assert response.status_code == 200
        client.headers['X-KTL-CSRF'] = response.json()['csrf']
        client.store = app.state.store
        yield client


def start_worker(client):
    root = client.store.root
    log = (root / 'test-worker.log').open('ab')
    process = subprocess.Popen([sys.executable, '-m', 'kernel_tuning_lab', 'worker', '--workspace', str(root)],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, env={**os.environ, 'OPENAI_API_KEY': 'synthetic-must-not-forward'})
    process.log_handle = log
    return process


def stop_worker(process):
    if process.poll() is None:
        process.terminate()
    process.wait(timeout=15)
    process.log_handle.close()


def wait_job(client, key, states, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/' + key).json()
        if job['state'] in states:
            return job
        time.sleep(.04)
    raise AssertionError(f'task did not reach {states}: {job}')


def create_prechecked(client, spec=None, project=None):
    response = client.post('/api/jobs', json={'spec': spec or cpu_spec(), 'environment': 'core', 'project': project})
    assert response.status_code == 200, response.text
    job = response.json()
    response = client.post('/api/jobs/' + job['id'] + '/precheck', json={})
    assert response.status_code == 200, response.text
    result = wait_job(client, job['id'], {'ready', 'invalid'})
    assert result['state'] == 'ready', result
    return result


def submit(client, job, key=None):
    return client.post('/api/jobs/' + job['id'] + '/submit', json={
        'plan_sha': job['plan_sha'], 'idempotency_key': key or job['id'] + ':' + job['plan_sha']})


def test_native_plan_separates_reveals_acceptance_and_hard_wall():
    plan, _ = plan_spec(cpu_spec(), 'run', {})
    assert plan['search_trials'] == 2 and plan['max_observations'] == 8
    assert plan['max_validation_evaluations'] == 1 and plan['max_physical_measurements'] == 9
    with pytest.raises(ValueError, match='measurement budget'):
        plan_spec(cpu_spec(), 'run', {'max_observations': 8})
    with pytest.raises(ValueError, match='independent correctness'):
        plan_spec({**cpu_spec(), 'validation': {'trust_claims': True}}, 'run', {})
    with pytest.raises(ValueError, match='unknown experiment fields'):
        plan_spec({**cpu_spec(), 'shell': 'echo forbidden'}, 'run', {})
    matrix = {'schema_version': 1, 'kind': 'research', 'name': 'bounded matrix',
        'algorithms': ['random@1.0.0'], 'dtypes': ['float32'], 'seeds': [1, 2, 3],
        'budgets': [{'max_evaluations': 2, 'max_wall_time_s': 2}], 'defaults': {'validation': {'trust_claims': False}},
        'cases': [{'name': 'gemm', 'operator': 'gemm@1.0.0', 'kernels': ['python/gemm@1.0.0'], 'shapes': [{'M': 1, 'N': 1, 'K': 1}]}]}
    with pytest.raises(ValueError, match='max_jobs'):
        plan_spec(matrix, 'research', {'max_jobs': 2})


def make_zip(name, data=b'hello', *, mode=None, duplicate=False):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        member = zipfile.ZipInfo(name)
        if mode is not None:
            member.create_system = 3; member.external_attr = mode << 16
        archive.writestr(member, data)
        if duplicate:
            archive.writestr(member, b'duplicate')
    return output.getvalue()


@pytest.mark.parametrize('name,mode', [('../escape.py',None),('/absolute.py',None),('C:/drive.py',None),
    ('dir\\escape.py',None),('link.py',stat.S_IFLNK|0o777),('pipe.py',stat.S_IFIFO|0o600)])
def test_zip_paths_and_special_members_cannot_escape(tmp_path, name, mode):
    store = Store(tmp_path / 'vault')
    with pytest.raises(ValueError):
        save_project(store, [('input.zip', make_zip(name, mode=mode))])
    assert not list((store.root / 'projects').iterdir())


def test_duplicate_bomb_and_static_upload_do_not_execute_or_overwrite(tmp_path):
    store = Store(tmp_path / 'vault')
    marker = tmp_path / 'must-not-exist'
    code = f'from pathlib import Path\nPath({str(marker)!r}).write_text("executed")\n'
    project = save_project(store, [('assets.py', code.encode())])
    assert not marker.exists() and project['trusted'] is None
    with pytest.raises(ValueError, match='trusted'):
        verify_project(store, project['id'], project['version'])
    with pytest.raises(ValueError, match='duplicate'):
        save_project(store, [('input.zip', make_zip('assets.py', duplicate=True))])
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('huge.txt', b'a' * (2 * 1024**2))
    with pytest.raises(ValueError, match='ratio'):
        save_project(store, [('bomb.zip', out.getvalue())])
    trust(store, project['id'], project['version'], True, 'owner-reviewed test source')
    file = store.root / 'projects' / project['id'] / project['version'] / 'files/assets.py'
    file.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        verify_project(store, project['id'], project['version'])
    assert not marker.exists()


def test_auth_origin_csrf_invalid_inputs_and_file_access(client, tmp_path):
    with TestClient(password_app(tmp_path / 'anonymous', auto_worker=False), base_url='http://127.0.0.1:18000') as anonymous:
        assert anonymous.get('/api/jobs').status_code == 401
        assert anonymous.post('/api/session', json={'token': 'wrong'}).status_code == 403
    assert client.get('/api/jobs', headers={'Host': 'attacker.example'}).status_code == 403
    assert client.post('/api/jobs', json={'spec': cpu_spec()}, headers={'Origin': 'https://attacker.example'}).status_code == 403
    assert client.post('/api/jobs', json={'spec': cpu_spec()}, headers={'X-KTL-CSRF': 'wrong'}).status_code == 403
    assert client.post('/api/jobs', json=[]).status_code == 400
    assert client.post('/api/jobs', content=b'x' * (1024**2 + 1)).status_code == 413
    assert client.post('/api/shell', json={'command': 'echo no'}).status_code in (404, 405)
    assert client.post('/api/projects', files={'files': ('bad.zip', b'not a ZIP')}).status_code == 400
    job = client.post('/api/jobs', json={'spec': cpu_spec()}).json()
    target = client.store.directory(job['id']) / 'output'; target.mkdir()
    secret = tmp_path / 'private.txt'; secret.write_text('private-outside-the-task')
    (target / 'report.md').symlink_to(secret)
    response = client.post('/api/jobs/' + job['id'] + '/exports', json={'files': ['output/report.md']})
    assert response.status_code == 403 and secret.read_text() not in response.text
    response = client.post('/api/jobs/' + job['id'] + '/exports', json={'files': ['../../private.txt']})
    assert response.status_code == 403
    assert client.get('/api/exports/..').status_code == 404


@pytest.mark.skipif(not landlock_abi(), reason='Linux Landlock is required for uploaded code')
def test_real_runner_idempotency_replay_download_and_isolated_upload(client, tmp_path):
    secret = tmp_path / 'unrelated-private-file'; secret.write_text('outside-private-scope')
    source = f'''import os
from kernel_tuning_lab.sdk import SearchAlgorithmDefinition
from kernel_tuning_lab.search import make_searcher
def register_assets(catalog):
    assert 'OPENAI_API_KEY' not in os.environ and 'SSH_AUTH_SOCK' not in os.environ
    try:
        open({str(secret)!r}).read()
    except PermissionError:
        pass
    else:
        raise RuntimeError('private files must be outside the Worker read scope')
    catalog.register(SearchAlgorithmDefinition('test/reviewed-random', '1.0.0', make))
def make(seed=0):
    return make_searcher('random', seed=seed)
'''
    upload = client.post('/api/projects', files={'files': ('assets.py', source.encode())}).json()
    project = {'id': upload['id'], 'version': upload['version']}
    spec = {**cpu_spec(), 'plugins': ['assets.py'], 'algorithms': ['test/reviewed-random@1.0.0']}
    untrusted = client.post('/api/jobs', json={'spec': spec, 'project': project}).json()
    assert client.post('/api/jobs/' + untrusted['id'] + '/precheck', json={}).status_code == 400
    assert client.post(f'/api/projects/{upload["id"]}/{upload["version"]}/trust',
                       json={'acknowledged': True, 'source': 'owned integration fixture'}).status_code == 200
    process = start_worker(client)
    try:
        job = create_prechecked(client, spec, project)
        first = submit(client, job, 'idempotent-fixture-request')
        second = submit(client, job, 'idempotent-fixture-request')
        assert first.status_code == second.status_code == 200
        done = wait_job(client, job['id'], {'completed', 'failed'})
        assert done['state'] == 'completed', done
        listed = next(j for j in client.get('/api/jobs').json()['jobs'] if j['id'] == job['id'])
        assert listed['planned'] == done['planned'] and listed['last_result']['status'] == 'passed'
        assert 'precheck' not in listed and 'plan' not in listed
        assert client.get('/api/jobs/' + job['id']).json()['precheck'] == done['precheck']
        assert done['last_result']['isolation']['mode'] in ('landlock', 'bubblewrap')
        events = client.get('/api/jobs/' + job['id'] + '/event-history').json()['events']
        observations = [e for e in events if e['kind'] == 'observation']
        assert 1 <= len(observations) <= 4
        assert len(observations) == done['last_result']['observations']
        assert all(e['data']['observation']['correctness_verified'] for e in observations)
        from kernel_tuning_lab.storage import ExperimentStore
        with ExperimentStore(client.store.directory(job['id']) / 'output/measurements.sqlite', read_only=True) as native:
            for event in observations:
                row = event['data']['observation']
                assert row['config'] == dict(native.get_record(row['record_id']).config.params)
                assert row['dataset'] and row['samples_count'] > 0
        cursor = observations[1]['seq']
        replay = client.get(f'/api/jobs/{job["id"]}/event-history?after={cursor}').json()['events']
        assert replay and all(e['seq'] > cursor for e in replay)
        assert len({e['seq'] for e in replay}) == len(replay)
        results = client.get(f'/api/jobs/{job["id"]}/results').json()
        assert len(results['details'][0]['records']) >= len(observations)
        cohorts = {r['id']: r['cohort'] for d in results['details'] for r in d['records']}
        assert all(e['data']['observation']['cohort'] == cohorts[e['data']['observation']['record_id']]
                   for e in observations)
        db = client.store.directory(job['id']) / 'output/measurements.sqlite'
        before = hashlib.sha256(db.read_bytes()).hexdigest()
        export = client.post(f'/api/jobs/{job["id"]}/exports', json={'files': ['output/report.md', 'output/measurements.sqlite']})
        assert export.status_code == 200, export.text
        package = client.get(export.json()['download']).content
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            manifest = json.loads(archive.read('EXPORT_MANIFEST.json'))
            for row in manifest['files']:
                assert hashlib.sha256(archive.read(row['path'])).hexdigest() == row['sha256']
        assert hashlib.sha256(db.read_bytes()).hexdigest() == before
        assert client.get(f'/api/jobs/{job["id"]}/table?format=csv').status_code == 200
        other = create_prechecked(client, spec, project)
        assert submit(client, other, 'idempotent-fixture-request').status_code == 400
        assert sum(j['state'] == 'completed' for j in client.store.jobs()) == 1
    finally:
        stop_worker(process)


def slow_project(client):
    source = '''import time
from kernel_tuning_lab.sdk import SearchAlgorithmDefinition
from kernel_tuning_lab.search import make_searcher
class Slow:
    def __init__(self, seed=0): self.inner = make_searcher('random',seed=seed)
    def suggest(self,*args,**kwargs):
        time.sleep(.7)
        return self.inner.suggest(*args,**kwargs)
    def observe(self,*args,**kwargs):
        callback = getattr(self.inner,'observe',None)
        return callback(*args,**kwargs) if callable(callback) else None
def register_assets(c): c.register(SearchAlgorithmDefinition('test/slow-random','1.0.0',Slow))
'''
    value = client.post('/api/projects', files={'files': ('slow.py', source.encode())}).json()
    client.post(f'/api/projects/{value["id"]}/{value["version"]}/trust', json={'acknowledged': True, 'source': 'owned cancellation fixture'})
    spec = {**cpu_spec(), 'plugins': ['slow.py'], 'algorithms': ['test/slow-random@1.0.0'],
            'budget': {'max_evaluations': 8, 'max_wall_time_s': 20}}
    return spec, {'id': value['id'], 'version': value['version']}


@pytest.mark.skipif(not landlock_abi(), reason='Linux Landlock')
def test_worker_crash_adopts_own_task_without_restarting_observations(client):
    spec, project = slow_project(client)
    process = start_worker(client)
    try:
        job = create_prechecked(client, spec, project)
        assert submit(client, job).status_code == 200
        running = wait_job(client, job['id'], {'running'})
        original = running['execution']
        process.kill(); process.wait(timeout=10); process.log_handle.close()
        process = start_worker(client)
        done = wait_job(client, job['id'], {'completed', 'failed', 'interrupted'})
        assert done['state'] == 'completed', done
        assert done['last_execution']['pid'] == original['pid']
        assert done['last_execution']['start_ticks'] == original['start_ticks']
        execution_events = [e for e in client.store.events(job['id'], limit=1000) if e['kind'] == 'execution' and e['data']['operation'] == 'run']
        assert len(execution_events) == 1
        assert client.get('/api/bootstrap').status_code == 200
    finally:
        stop_worker(process)


@pytest.mark.skipif(not landlock_abi(), reason='Linux Landlock')
def test_cancel_only_owned_tree_then_native_cache_resume(client):
    spec, project = slow_project(client)
    process = start_worker(client)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        job = create_prechecked(client, spec, project)
        submit(client, job)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            observed = [e for e in client.store.events(job['id'], limit=1000) if e['kind'] == 'observation']
            if observed:
                break
            time.sleep(.025)
        assert observed
        assert client.post(f'/api/jobs/{job["id"]}/cancel', json={}).status_code == 200
        stopped = wait_job(client, job['id'], {'cancelled'})
        assert unrelated.poll() is None
        assert client.post(f'/api/jobs/{job["id"]}/resume', json={}).status_code == 200
        done = wait_job(client, job['id'], {'completed', 'failed'})
        assert done['state'] == 'completed', done
        assert done['last_result']['cache_hits'] >= 1
        assert all(e['data']['observation']['record_id'] for e in observed)
        assert unrelated.poll() is None
    finally:
        stop_worker(process)
        unrelated.terminate(); unrelated.wait(timeout=5)


def test_resource_busy_is_durable_and_requires_explicit_retry(client):
    job = client.store.create({'spec': cpu_spec(), 'environment': 'core'})
    attempt = client.store.directory(job['id']) / 'attempts/fixture'; attempt.mkdir(parents=True)
    write_json(attempt / 'result.json', {'status': 'resource_unavailable', 'exit_code': 75, 'reason': 'fixture busy GPU'})
    client.store.update(job['id'], state='running', execution={'operation': 'run', 'attempt': str(attempt.relative_to(client.store.root)),
                        'pid': os.getpid(), 'start_ticks': 0, 'host_id': 'not-this-host'})
    settle(client.store, client.store.get(job['id']))
    result = client.store.get(job['id'])
    assert result['state'] == 'queued' and result['resource_blocked'] and result['execution'] is None
    assert not alive({'pid': os.getpid(), 'start_ticks': 0, 'host_id': 'not-this-host'})


def test_changed_frozen_plan_is_rejected_before_execution(client):
    from kernel_tuning_lab.console.store import digest
    process = start_worker(client)
    try:
        job = create_prechecked(client)
        file = client.store.root / job['plan_dir'] / 'plan.json'
        plan = read_json(file)
        plan['gpu'] = 'GPU-unapproved-change'
        plan['sha256'] = digest({k: v for k, v in plan.items() if k != 'sha256'})
        write_json(file, plan)
        assert submit(client, job).status_code == 200
        failed = wait_job(client, job['id'], {'failed'})
        assert 'approved task identity' in failed['error']
        assert not (client.store.directory(job['id']) / 'output/measurements.sqlite').exists()
    finally:
        stop_worker(process)


def test_inherited_allocation_cannot_be_overridden_by_a_web_gpu_selection(tmp_path):
    from kernel_tuning_lab.console.plans import freeze_plan
    store = Store(tmp_path / 'workspace')
    config = store.config()
    config['environments'].append({**config['environments'][0], 'id': 'allocated', 'device': 'cuda'})
    write_json(store.root / 'server.json', config)
    for resource_context in ({'CUDA_VISIBLE_DEVICES': ''}, {'CUDA_VISIBLE_DEVICES': '0'},
                             {'SLURM_JOB_ID': 'fixture'}, {'PBS_JOBID': 'fixture'}):
        write_json(store.root / 'worker/resource-context.json', resource_context)
        job = store.create({'spec': cpu_spec(), 'environment': 'allocated', 'gpu': 'GPU-fixture'})
        with pytest.raises(ValueError, match='inherited a GPU allocation'):
            freeze_plan(store, job)
        assert not (store.directory(job['id']) / 'attempts').exists()


def test_gpu_device_binding_uses_driver_uuid_and_minor_instead_of_display_order(tmp_path):
    from kernel_tuning_lab.console.isolation import nvidia_device_minor
    for pci, uuid, minor in [('0000:01:00.0', 'GPU-selected', 7), ('0000:02:00.0', 'GPU-other', 2)]:
        directory = tmp_path / pci
        directory.mkdir()
        (directory / 'information').write_text(f'GPU UUID: {uuid}\nDevice Minor: {minor}\n')
    assert nvidia_device_minor('GPU-selected', tmp_path) == 7
    assert nvidia_device_minor('GPU-other', tmp_path) == 2
    with pytest.raises(RuntimeError, match='selected UUID'):
        nvidia_device_minor('GPU-missing', tmp_path)


@pytest.mark.skipif(not landlock_abi(), reason='Linux Landlock')
def test_trusted_plugin_failure_is_retained_and_next_queue_job_can_run(client):
    source = b"raise RuntimeError('owned fixture plugin registration failure')\n"
    project = client.post('/api/projects', files={'files': ('invalid.py', source)}).json()
    assert project['trusted'] is None  # Static upload did not execute the Python.
    identity = {'id': project['id'], 'version': project['version']}
    client.post(f'/api/projects/{project["id"]}/{project["version"]}/trust',
                json={'acknowledged': True, 'source': 'owned negative acceptance fixture'})
    process = start_worker(client)
    try:
        rejected = client.post('/api/jobs', json={'environment': 'core', 'project': identity,
            'spec': {**cpu_spec(), 'plugins': ['invalid.py']}}).json()
        client.post(f'/api/jobs/{rejected["id"]}/precheck', json={})
        result = wait_job(client, rejected['id'], {'invalid'})
        assert 'owned fixture plugin registration failure' in result['error']
        assert client.get(f'/api/jobs/{rejected["id"]}/logs').status_code == 200
        job = create_prechecked(client)
        assert submit(client, job).status_code == 200
        assert wait_job(client, job['id'], {'completed', 'failed'})['state'] == 'completed'
        assert client.store.get(rejected['id'])['state'] == 'invalid'
        assert process.poll() is None
    finally:
        stop_worker(process)


def test_workspace_snapshot_uses_existing_restore_and_rotates_access(tmp_path):
    from kernel_tuning_lab.console.archive import snapshot
    from kernel_tuning_lab.portable import restore_vault, verify_vault
    root = tmp_path / 'source'
    app = password_app(root, auto_worker=False)
    with TestClient(app, base_url='http://127.0.0.1:18000', headers={'Origin': 'http://127.0.0.1:18000'}) as client:
        password_hash = app.state.store.meta(PASSWORD_KEY)
        response = client.post('/api/session', json={'password': TEST_PASSWORD})
        client.headers['X-KTL-CSRF'] = response.json()['csrf']
        client.store = app.state.store
        process = start_worker(client)
        try:
            job = create_prechecked(client)
            submit(client, job)
            completed = wait_job(client, job['id'], {'completed'})
            original_results = client.get(f'/api/jobs/{job["id"]}/results').json()
            stopped = subprocess.run([sys.executable, '-m', 'kernel_tuning_lab', 'worker', '--workspace', str(root), '--stop'],
                                     capture_output=True, text=True, timeout=30)
            assert stopped.returncode == 0 and json.loads(stopped.stdout)['status'] == 'stopped'
            import fcntl
            with (root / 'worker/lease').open('a+b') as lease:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            stop_worker(process)
        with pytest.raises(ValueError, match='stop this workspace'):
            snapshot(root, tmp_path / 'not-quiescent')
        old_cookies = client.cookies
    vault = tmp_path / 'vault'
    result = snapshot(root, vault)
    assert result['status'] == verify_vault(vault)['status'] == 'passed'
    output = tmp_path / 'restored'
    restored = restore_vault(vault, 'workspace', output)
    assert restored['experiments_executed'] == 0 and restored['SQLite_checked'] >= 2
    assert output.stat().st_mode & 0o077 == 0
    restored_app = create_app(output, auto_worker=False)
    assert restored_app.state.store.meta(PASSWORD_KEY) == password_hash
    with TestClient(restored_app, base_url='http://127.0.0.1:18000', headers={'Origin': 'http://127.0.0.1:18000'}, cookies=old_cookies) as client:
        assert client.get('/api/jobs').status_code == 401
        assert client.post('/api/session', json={'password': TEST_PASSWORD}).status_code == 200
        assert client.get(f'/api/jobs/{job["id"]}').json()['state'] == 'completed'
        restored_results = client.get(f'/api/jobs/{job["id"]}/results').json()
        assert [r['id'] for r in restored_results['details'][0]['records']] == [r['id'] for r in original_results['details'][0]['records']]
    public = tmp_path / 'public'; public.mkdir(); public.chmod(0o755)
    with pytest.raises(ValueError, match='owner-only'):
        restore_vault(vault, 'workspace', public)


def test_password_is_persistent_and_legacy_tokens_cannot_log_in(tmp_path):
    root = tmp_path / 'workspace'
    store = Store(root)
    legacy = root / 'access-token'
    legacy.write_text('retired-test-token')
    assert initialize_password(store, TEST_PASSWORD)
    encoded = store.meta(PASSWORD_KEY)
    assert TEST_PASSWORD not in encoded
    assert not initialize_password(store, TEST_PASSWORD)
    with pytest.raises(ValueError, match='not changed'):
        initialize_password(store, 'another-test-password')
    headers = {'Origin': 'http://127.0.0.1:18000'}
    app = create_app(root, auto_worker=False)
    with TestClient(app, base_url=headers['Origin'], headers=headers) as client:
        assert client.post('/api/session', json={'token': 'retired-test-token'}).status_code == 401
        assert client.post('/api/session', json={'password': 'incorrect'}).status_code == 401
        assert client.post('/api/session', json={'password': TEST_PASSWORD}).status_code == 200
    assert not legacy.exists() and list(root.glob('access-token.retired-*'))
    restarted = create_app(root, auto_worker=False)
    assert restarted.state.store.meta(PASSWORD_KEY) == encoded
    with TestClient(restarted, base_url=headers['Origin'], headers=headers) as client:
        assert client.post('/api/session', json={'password': TEST_PASSWORD}).status_code == 200
