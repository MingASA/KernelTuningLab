"""Loopback FastAPI service; request handlers never execute experiment code."""
from __future__ import annotations

import asyncio
import csv
from contextlib import asynccontextmanager
import hashlib
import hmac
import io
from importlib.resources import files
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import time
from urllib.parse import urlsplit
import uuid
import zipfile

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .isolation import clean_environment, landlock_abi, protect_supervisor, restriction_capabilities
from .plans import plan_spec, templates
from .projects import project, save_project, trust, verify_project
from .store import ACTIVE, Store, beneath, digest, identifier, read_json, write_json
from .telemetry import native_details
from .worker import alive, command, host_identity


async def request_object(request):
    value = await request.json()
    if not isinstance(value, dict):
        raise ValueError('request body must be a JSON object')
    return value


def file_inventory(store, key):
    root = store.directory(key)
    store.get(key)
    rows = []
    for top in ('output', 'attempts', 'plans'):
        parent = root / top
        if not parent.exists():
            continue
        for directory, dirs, names in os.walk(parent, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink()
                             and d not in ('home', 'tmp', '__pycache__'))
            for name in sorted(names):
                file = Path(directory) / name
                if file.is_symlink() or not file.is_file() or name.endswith(('-wal', '-shm')) or name.startswith('.'):
                    continue
                relative = file.relative_to(root).as_posix()
                rows.append({'path': relative, 'bytes': file.stat().st_size,
                             'kind': ('compiled' if name.endswith(('.cubin', '.ptx', '.ttir', '.ttgir', '.llir', '.so')) else
                                      'source' if name.endswith(('.py', '.cu', '.cpp', '.h', '.hpp', '.js', '.css')) else
                                      'database' if name.endswith(('.sqlite', '.db')) else
                                      'report' if name.endswith(('.md', '.html', '.svg', '.png')) else
                                      'log' if name.endswith(('.log', '.jsonl')) else 'configuration')})
    return rows


def export_files(store, key, selected):
    if not isinstance(selected, list) or not selected or len(selected) > 4096 or any(not isinstance(x, str) for x in selected):
        raise ValueError('select at least one result file')
    allowed = {r['path']: r for r in file_inventory(store, key)}
    if len(set(selected)) != len(selected) or any(x not in allowed for x in selected):
        raise PermissionError('file is not an admitted task artifact')
    export_id = uuid.uuid4().hex
    target = store.root / 'exports' / export_id
    target.mkdir(mode=0o700)
    manifest = {'task': key, 'files': [], 'created_at': time.time(), 'complete_raw_results_retained_on_server': True}
    with zipfile.ZipFile(target / 'results.zip', 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for index, relative in enumerate(selected):
            original = beneath(store.directory(key), relative)
            source = original
            if original.suffix in ('.sqlite', '.db'):
                source = target / (str(index) + '.sqlite')
                with sqlite3.connect(original.resolve().as_uri() + '?mode=ro', uri=True, timeout=5) as incoming:
                    with sqlite3.connect(source) as outgoing:
                        incoming.backup(outgoing, pages=256, sleep=.05)
                        if outgoing.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                            raise ValueError('exported database integrity check failed')
            # Copy each selected file once; the SHA describes exactly the bytes in the ZIP.
            sha = hashlib.sha256(); size = 0
            with source.open('rb') as reader, archive.open(relative, 'w', force_zip64=True) as writer:
                for block in iter(lambda: reader.read(1024**2), b''):
                    sha.update(block); size += len(block); writer.write(block)
            manifest['files'].append({'path': relative, 'bytes': size, 'sha256': sha.hexdigest(),
                                      'snapshot': 'SQLite backup' if source != original else 'selected file'})
            if source != original:
                source.unlink()
        archive.writestr('EXPORT_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    archive_sha = hashlib.sha256()
    with (target / 'results.zip').open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            archive_sha.update(block)
    result = {'id': export_id, 'sha256': archive_sha.hexdigest(), 'bytes': (target / 'results.zip').stat().st_size,
              'manifest': manifest, 'download': '/api/exports/' + export_id}
    write_json(target / 'manifest.json', result)
    return result


def create_app(workspace, *, auto_worker=True):
    protect_supervisor()
    store = Store(workspace)
    token_file = store.root / 'access-token'
    binding = digest([host_identity(), str(store.root)])
    if store.meta('authentication_binding') not in (None, binding):
        if token_file.exists():
            token_file.rename(store.root / ('access-token.previous-' + str(time.time_ns())))
        with store.connect() as db:
            db.execute('UPDATE metadata SET value=? WHERE key=?', (secrets.token_hex(32), 'session_secret'))
    with store.connect() as db:
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('authentication_binding', binding))
    if not token_file.exists():
        descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(secrets.token_urlsafe(32) + '\n')
    token = token_file.read_text().strip()
    secret = bytes.fromhex(store.meta('session_secret'))
    worker_process = None
    failures = []

    async def supervise():
        nonlocal worker_process
        while True:
            if worker_process is not None:
                worker_process.poll()
            if not alive(read_json(store.root / 'worker/identity.json', {})):
                now = time.time()
                failures[:] = [t for t in failures if now - t < 300]
                if len(failures) < 3:
                    directory = store.root / 'worker/supervisor'
                    directory.mkdir(exist_ok=True)
                    env = {'python': sys.executable, 'device': 'cpu', 'env': {}}
                    with (store.root / 'worker/supervisor.log').open('ab') as log:
                        worker_process = subprocess.Popen(command(sys.executable, Path(__file__).parents[2],
                            'work', '--workspace', store.root), cwd=directory,
                            env=clean_environment(directory, env, cpu_only=True), stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
                    failures.append(now)
            await asyncio.sleep(2)

    @asynccontextmanager
    async def lifespan(app):
        from ..run_lifecycle import OutputLease
        with OutputLease(store.root, filename='.service.lock'):
            watcher = asyncio.create_task(supervise()) if auto_worker else None
            try:
                yield
            finally:
                if watcher:
                    watcher.cancel()
                    try:
                        await watcher
                    except asyncio.CancelledError:
                        pass
            # API shutdown intentionally leaves the separately supervised Worker and its queue alive.

    app = FastAPI(title='KernelTuningLab Workbench', docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.store = store
    app.state.token_file = token_file
    app.state.stopping = False

    def session(request):
        cookie = request.cookies.get('ktlab_session', '')
        try:
            payload, signature = cookie.rsplit('.', 1)
            expected = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected) or time.time() - int(payload.split(':')[0]) > 12 * 3600:
                return None
            return hmac.new(secret, ('csrf:' + payload).encode(), hashlib.sha256).hexdigest()
        except (ValueError, TypeError):
            return None

    @app.middleware('http')
    async def boundary(request, call_next):
        host = request.headers.get('host', '')
        try:
            parsed = urlsplit('http://' + host)
            valid_host = parsed.hostname in ('127.0.0.1', 'localhost', '::1') and parsed.username is None and parsed.path == ''
        except ValueError:
            valid_host = False
        origin = request.headers.get('origin')
        if not valid_host or (origin and origin != 'http://' + host) or request.headers.get('sec-fetch-site') == 'cross-site':
            return JSONResponse({'error': 'request origin is not admitted'}, status_code=403)
        if request.method not in ('GET', 'HEAD', 'POST', 'PUT'):
            return JSONResponse({'error': 'operation not supported'}, status_code=405)
        csrf = session(request)
        public = request.url.path in ('/', '/index.html', '/app.js', '/app.css', '/THIRD_PARTY_LICENSES.txt', '/api/session')
        if request.url.path.startswith('/api/') and not public and not csrf:
            return JSONResponse({'error': 'session required'}, status_code=401)
        if request.method in ('POST', 'PUT'):
            if origin != 'http://' + host:
                return JSONResponse({'error': 'a same-origin request is required'}, status_code=403)
            if request.url.path != '/api/session' and not hmac.compare_digest(request.headers.get('x-ktl-csrf', ''), csrf or 'missing'):
                return JSONResponse({'error': 'request protection failed'}, status_code=403)
            maximum = store.config()['upload_bytes'] + 1024**2 if request.url.path == '/api/projects' else 1024**2
            try:
                if int(request.headers.get('content-length', 0)) > maximum:
                    return JSONResponse({'error': 'request exceeds the server limit'}, status_code=413)
            except ValueError:
                return JSONResponse({'error': 'invalid request length'}, status_code=400)
            original_receive = request._receive
            received = 0
            async def limited_receive():
                nonlocal received
                message = await original_receive()
                received += len(message.get('body', b''))
                if received > maximum:
                    from starlette.exceptions import HTTPException
                    raise HTTPException(413, 'request exceeds the server limit')
                return message
            request._receive = limited_receive
        response = await call_next(request)
        response.headers.update({'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Referrer-Policy': 'no-referrer', 'X-Frame-Options': 'DENY',
            'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"})
        return response

    @app.exception_handler(ValueError)
    async def bad_value(_request, exc):
        return JSONResponse({'error': str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(_request, exc):
        return JSONResponse({'error': str(exc)}, status_code=404)

    @app.exception_handler(PermissionError)
    async def forbidden(_request, exc):
        return JSONResponse({'error': str(exc)}, status_code=403)

    @app.get('/api/session')
    def current_session(request: Request):
        csrf = session(request)
        return {'authenticated': bool(csrf), 'csrf': csrf}

    @app.post('/api/session')
    async def login(request: Request):
        body = await request_object(request)
        if not isinstance(body, dict) or not isinstance(body.get('token'), str) or not hmac.compare_digest(body['token'], token):
            return JSONResponse({'error': 'access token is invalid'}, status_code=401)
        payload = str(int(time.time())) + ':' + secrets.token_hex(16)
        cookie = payload + '.' + hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
        response = JSONResponse({'authenticated': True, 'csrf': hmac.new(secret, ('csrf:' + payload).encode(), hashlib.sha256).hexdigest()})
        response.set_cookie('ktlab_session', cookie, httponly=True, samesite='strict', max_age=12 * 3600)
        return response

    @app.get('/api/bootstrap')
    def bootstrap():
        from .. import __version__
        return {'version': __version__, 'workspace_id': store.meta('workspace_id'),
                'workspace_name': store.root.name, 'templates': templates(), 'landlock_ABI': landlock_abi(),
                'execution_restriction': restriction_capabilities(),
                'worker': {'alive': alive(read_json(store.root / 'worker/identity.json', {})), 'automatic': auto_worker},
                'limits': {k: v for k, v in store.config().items() if k.startswith(('max_', 'upload_'))}}

    @app.get('/api/environments')
    def environments():
        return {'environments': [{**env, 'probe': read_json(store.root / 'worker' / ('environment-' + identifier(env['id']) + '.json'),
                    {'status': 'probing'})} for env in store.config()['environments']]}

    @app.post('/api/environments/probe')
    def reprobe_environments():
        for env in store.config()['environments']:
            (store.root / 'worker' / ('environment-' + identifier(env['id']) + '.json')).unlink(missing_ok=True)
        write_json(store.root / 'worker/probe-request.json', {'requested_at': time.time()})
        return {'status': 'queued'}

    @app.get('/api/devices')
    def devices():
        from ..gpu_admission import snapshot, idle
        try:
            return {'devices': [{'uuid': row['uuid'], 'name': row['name'], 'memory_mib': row['memory_mib'],
                                 'used_mib': row['used_mib'], 'util_pct': row['util_pct'], 'busy': not idle(row),
                                 'process_count': len(row['pids'])} for row in snapshot().values()]}
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            return {'devices': [], 'status': 'unavailable', 'reason': str(exc)}

    @app.get('/api/projects')
    def projects():
        rows = []
        for parent in sorted((store.root / 'projects').iterdir()):
            if not parent.is_dir() or parent.is_symlink():
                continue
            for version in sorted(parent.iterdir()):
                if version.is_dir() and len(version.name) == 64:
                    rows.append(project(store, parent.name, version.name))
        return {'projects': rows}

    @app.post('/api/projects')
    async def upload(request: Request):
        async with request.form(max_files=32, max_fields=4) as form:
            incoming = []
            total = 0
            for value in form.getlist('files'):
                if not hasattr(value, 'read') or not value.filename:
                    raise ValueError('select source files or a ZIP project')
                content = await value.read(store.config()['upload_bytes'] + 1)
                total += len(content)
                if total > store.config()['upload_bytes']:
                    raise ValueError('upload exceeds the server byte limit')
                incoming.append((value.filename, content))
            try:
                return await run_in_threadpool(save_project, store, incoming,
                    project_id=form.get('project_id'), label=form.get('name'))
            except (zipfile.BadZipFile, NotImplementedError, EOFError) as exc:
                raise ValueError('invalid or unsupported ZIP project') from exc

    @app.get('/api/projects/{key}/{version}/file')
    def project_file(key: str, version: str, path: str):
        value = project(store, key, version)
        if path not in {row['path'] for row in value['files']}:
            raise KeyError('project file not found')
        file = beneath(store.root / 'projects' / key / version / 'files', path)
        return {'path': path, 'content': file.read_text()[:65536]}

    @app.post('/api/projects/{key}/{version}/trust')
    async def confirm_trust(key: str, version: str, request: Request):
        body = await request_object(request)
        return trust(store, key, version, body.get('acknowledged'), body.get('source'))

    @app.get('/api/jobs')
    def jobs():
        return {'jobs': store.jobs()}

    @app.post('/api/jobs')
    async def create_job(request: Request):
        body = await request_object(request)
        if not isinstance(body, dict) or not isinstance(body.get('spec'), dict):
            raise ValueError('a native experiment specification is required')
        store.environment(body.get('environment', 'core'))
        data = {key: body.get(key) for key in ('spec', 'project', 'gpu')}
        data.update(environment=body.get('environment', 'core'), action=body.get('action', 'run'))
        return store.create(data)

    @app.get('/api/jobs/{key}')
    def get_job(key: str):
        return store.get(key)

    @app.put('/api/jobs/{key}')
    async def edit_job(key: str, request: Request):
        body = await request_object(request)
        store.environment(body.get('environment', 'core'))
        return store.edit(key, body['revision'], {k: body.get(k) for k in ('spec', 'project', 'gpu', 'environment', 'action')})

    @app.post('/api/jobs/{key}/precheck')
    def precheck(key: str):
        job = store.get(key)
        plan, _ = plan_spec(job['spec'], job.get('action', 'run'), store.config())
        if job.get('project'):
            verify_project(store, job['project']['id'], job['project']['version'])
        if job['state'] not in ('draft', 'invalid', 'ready'):
            raise ValueError('this task cannot be prechecked in its current state')
        if job['state'] != 'draft':
            data = {k: job.get(k) for k in ('spec', 'project', 'gpu', 'environment', 'action')}
            job = store.edit(key, job['revision'], data)
        return store.update(key, expected={'draft'}, state='prechecking', planned=plan, error=None)

    @app.post('/api/jobs/{key}/submit')
    async def submit(key: str, request: Request):
        body = await request_object(request)
        return store.submit(key, body.get('plan_sha'), body.get('idempotency_key'))

    @app.post('/api/jobs/{key}/cancel')
    def cancel(key: str):
        job = store.get(key)
        if job['state'] not in ACTIVE:
            raise ValueError('this task is not active')
        return store.update(key, expected=ACTIVE, state='cancelling' if job.get('execution') else 'cancelled')

    @app.post('/api/jobs/{key}/resume')
    def resume(key: str):
        job = store.get(key)
        if job['state'] not in ('interrupted', 'cancelled', 'failed', 'queued') or not job.get('precheck'):
            raise ValueError('this task cannot be resumed')
        return store.update(key, expected={job['state']}, state='queued', resource_blocked=False, error=None)

    @app.post('/api/jobs/{key}/duplicate')
    def duplicate(key: str):
        old = store.get(key)
        data = {k: old.get(k) for k in ('spec', 'project', 'gpu', 'environment', 'action')}
        data['spec'] = {**data['spec'], 'name': old['name'] + ' · copy'}
        return store.create(data)

    @app.get('/api/jobs/{key}/events')
    async def events(key: str, request: Request, after: int = 0):
        store.get(key)
        try:
            cursor = int(request.headers.get('last-event-id', after))
            if cursor < 0:
                raise ValueError('invalid event cursor')
        except ValueError:
            raise ValueError('invalid event cursor')
        async def stream():
            nonlocal cursor
            heartbeat = time.monotonic()
            while not app.state.stopping and not await request.is_disconnected():
                rows = store.events(key, cursor)
                for row in rows:
                    cursor = row['seq']
                    yield 'id: ' + str(cursor) + '\ndata: ' + json.dumps(row, ensure_ascii=False) + '\n\n'
                if time.monotonic() - heartbeat > 10:
                    yield ': heartbeat\n\n'; heartbeat = time.monotonic()
                await asyncio.sleep(.25 if rows else .5)
        return StreamingResponse(stream(), media_type='text/event-stream', headers={'X-Accel-Buffering': 'no'})

    @app.get('/api/jobs/{key}/event-history')
    def event_history(key: str, after: int = 0):
        return {'events': store.events(key, after, 1000)}

    @app.get('/api/jobs/{key}/results')
    def results(key: str):
        store.get(key)
        return native_details(store.directory(key) / 'output')

    @app.get('/api/jobs/{key}/records/{dataset}/{record_id}')
    def record(key: str, dataset: str, record_id: str):
        from ..web import ViewerCatalog
        store.get(key)
        catalog = ViewerCatalog([store.directory(key) / 'output'])
        catalog.scan()
        return catalog.record(dataset, record_id)

    @app.get('/api/jobs/{key}/table')
    def table(key: str, format: str = 'json'):
        store.get(key)
        records = [r for d in native_details(store.directory(key) / 'output')['details'] for r in d['records']]
        if format == 'json':
            return Response(json.dumps(records, ensure_ascii=False, allow_nan=False), media_type='application/json',
                            headers={'Content-Disposition': 'attachment; filename="configurations.json"'})
        if format != 'csv':
            raise ValueError('choose json or csv')
        output = io.StringIO()
        fields = ('id', 'hardware', 'operator', 'implementation', 'dtype', 'shape', 'seed', 'config_id',
                  'params', 'status', 'verified', 'latency_ms', 'samples', 'compile_ms', 'wall_s', 'failure')
        writer = csv.writer(output); writer.writerow(fields)
        for row in records:
            values = [json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in fields]
            writer.writerow(["'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@')) else v for v in values])
        return Response(output.getvalue(), media_type='text/csv; charset=utf-8',
                        headers={'Content-Disposition': 'attachment; filename="configurations.csv"'})

    @app.get('/api/jobs/{key}/logs')
    def logs(key: str):
        job = store.get(key)
        candidates = [store.directory(key) / 'output/worker.log']
        if job.get('execution'):
            candidates.append(store.root / job['execution']['attempt'] / 'launcher.log')
        elif (store.directory(key) / 'attempts').exists():
            candidates.extend(sorted((store.directory(key) / 'attempts').glob('*/launcher.log'), key=lambda p: p.stat().st_mtime)[-1:])
        rows = []
        for file in candidates:
            if file.is_file() and not file.is_symlink():
                with file.open('rb') as reader:
                    reader.seek(max(0, file.stat().st_size - 65536))
                    rows.append({'file': file.name, 'content': reader.read().decode('utf-8', errors='replace')})
        return {'logs': rows}

    @app.get('/api/jobs/{key}/files')
    def task_files(key: str):
        return {'files': file_inventory(store, key)}

    @app.get('/api/jobs/{key}/download')
    def download_file(key: str, path: str):
        job = store.get(key)
        if job['state'] in ACTIVE or path.endswith(('.sqlite', '.db')):
            raise ValueError('use a selected export for a running artifact or a consistent SQLite snapshot')
        if path not in {row['path'] for row in file_inventory(store, key)}:
            raise PermissionError('file is not an admitted task artifact')
        return FileResponse(beneath(store.directory(key), path), filename=Path(path).name,
                            media_type='application/octet-stream')

    @app.post('/api/jobs/{key}/exports')
    async def export(key: str, request: Request):
        body = await request_object(request)
        return await run_in_threadpool(export_files, store, key, body.get('files'))

    @app.get('/api/exports/{export_id}')
    def download_export(export_id: str):
        directory = store.root / 'exports' / identifier(export_id)
        if not (directory / 'manifest.json').is_file():
            raise KeyError('export not found')
        return FileResponse(directory / 'results.zip', media_type='application/zip', filename='ktlab-' + export_id[:8] + '.zip')

    @app.get('/api/history')
    def history():
        from ..web import ViewerCatalog
        roots = store.config().get('history_roots', [])
        if not roots:
            return {'experiments': []}
        catalog = ViewerCatalog(roots)
        return {'experiments': catalog.scan()}

    @app.get('/api/history/{key}')
    def historical_result(key: str):
        from ..web import ViewerCatalog
        catalog = ViewerCatalog(store.config().get('history_roots', []))
        catalog.scan()
        return catalog.detail(key)

    @app.get('/{asset:path}')
    def static(asset: str):
        mapping = {'': ('index.html', 'text/html'), 'index.html': ('index.html', 'text/html'),
                   'app.js': ('app.js', 'text/javascript'), 'app.css': ('app.css', 'text/css'),
                   'THIRD_PARTY_LICENSES.txt': ('THIRD_PARTY_LICENSES.txt', 'text/plain')}
        if asset not in mapping:
            return JSONResponse({'error': 'not found'}, status_code=404)
        name, mime = mapping[asset]
        resource = files('kernel_tuning_lab').joinpath('console_assets', name)
        if not resource.is_file():
            return JSONResponse({'error': 'frontend assets are missing; install a built wheel'}, status_code=503)
        return Response(resource.read_bytes(), media_type=mime)

    return app


def serve(workspace, port=8765, *, config=None, auto_worker=True):
    import uvicorn
    store = Store(workspace)
    if config:
        value = read_json(Path(config).expanduser().resolve())
        if not isinstance(value, dict):
            raise ValueError('server configuration must be a JSON object')
        write_json(store.root / 'server.json', value)
    app = create_app(store.root, auto_worker=auto_worker)
    print(f'KernelTuningLab Workbench · http://127.0.0.1:{port}', flush=True)
    print(f'Access token: {app.state.token_file} (owner-only file)', flush=True)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port,
                            access_log=False, server_header=False, timeout_graceful_shutdown=5))
    original_exit = server.handle_exit
    def close_streams(signum, frame):
        app.state.stopping = True
        original_exit(signum, frame)
    server.handle_exit = close_streams
    server.run()
