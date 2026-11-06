"""Content-addressed uploads. Inspection parses text and AST; it never imports code."""
from __future__ import annotations

import ast
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid
import zipfile

from .store import beneath, digest, identifier, read_json, write_json

EXTENSIONS = {'.py', '.json', '.md', '.txt', '.toml', '.yaml', '.yml', '.ini', '.cfg',
              '.csv', '.c', '.cpp', '.cu', '.h', '.hpp', '.jinja', '.gitignore'}


def upload_path(name):
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name or ':' in name:
        raise ValueError('upload requires a relative file name')
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ('', '.', '..') for p in name.split('/')) or len(name) > 240:
        raise ValueError('upload path cannot escape the project')
    if any(p in ('.git', '__pycache__', 'node_modules') for p in path.parts):
        raise ValueError('repository metadata and installed dependencies are not project uploads')
    if path.suffix.lower() not in EXTENSIONS and path.name not in ('LICENSE', 'NOTICE', '.gitignore'):
        raise ValueError('upload source, specification or dependency text files; binaries are not installed')
    return path.as_posix()


def inspect_file(name, data):
    result = {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
              'assets': [], 'imports': [], 'errors': []}
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        result['errors'].append('source files must be UTF-8 text')
        return result
    result.update(lines=len(text.splitlines()), preview=text[:1200])
    if name.endswith('.py'):
        try:
            if len(data) > 2 * 1024**2:
                raise ValueError('one Python source file exceeds 2 MiB')
            tree = ast.parse(text, filename=name)
            for index, node in enumerate(ast.walk(tree)):
                if index > 100000:
                    raise ValueError('Python syntax tree is too large')
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    result['imports'].extend([a.name for a in node.names] if isinstance(node, ast.Import)
                                             else [node.module or 'relative import'])
                if isinstance(node, ast.Call):
                    target = getattr(node.func, 'id', getattr(node.func, 'attr', ''))
                    if target in ('OperatorDefinition', 'KernelDefinition', 'SearchAlgorithmDefinition'):
                        values = {k.arg: k.value.value for k in node.keywords if isinstance(k.value, ast.Constant)}
                        for key, value in zip(('id', 'version'), node.args):
                            if isinstance(value, ast.Constant):
                                values[key] = value.value
                        result['assets'].append({'constructor': target, **values, 'discovery': 'static'})
            result['registration_hook'] = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in ('register_assets', 'register_plugin') for n in tree.body)
        except (SyntaxError, ValueError, RecursionError) as exc:
            result['errors'].append(str(exc))
    elif name.endswith('.json'):
        try:
            from ..asset_experiments import AssetExperiment, _json_pairs, _json_constant
            import json
            value = json.loads(text, object_pairs_hook=_json_pairs, parse_constant=_json_constant)
            if isinstance(value, dict) and value.get('kind') == 'research':
                from ..research import expand_research
                result['experiment'] = value
                result['planned'] = expand_research(value)['planned']
            elif isinstance(value, dict) and value.get('schema_version') == 2:
                result['experiment'] = AssetExperiment.from_dict(value).to_dict()
        except (ValueError, TypeError, KeyError) as exc:
            result['errors'].append(str(exc))
    if Path(name).name.startswith('requirements') or name.endswith('pyproject.toml'):
        result['dependency_text'] = text[:16000]
    return result


def save_project(store, files, *, project_id=None, label=None):
    config = store.config()
    members = {}
    total = 0
    raw_bytes = sum(len(data) for _, data in files)
    if not files or raw_bytes > config['upload_bytes']:
        raise ValueError('upload exceeds the server byte limit')

    def add(name, data):
        nonlocal total
        name = upload_path(name)
        folded = name.casefold()
        if any(key.casefold() == folded or key.casefold().startswith(folded + '/')
               or folded.startswith(key.casefold() + '/') for key in members):
            raise ValueError('duplicate or conflicting upload paths')
        total += len(data)
        if len(members) >= config['upload_files'] or total > config['upload_expanded_bytes']:
            raise ValueError('expanded project exceeds the server limit')
        members[name] = data

    for filename, data in files:
        if filename.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if len(archive.infolist()) > config['upload_files']:
                    raise ValueError('ZIP has too many members')
                declared = 0
                directories = set()
                for info in archive.infolist():
                    mode = info.external_attr >> 16
                    if info.flag_bits & 1 or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise ValueError('encrypted ZIPs, links and special files are not admitted')
                    if info.is_dir():
                        raw = info.filename.rstrip('/')
                        if not raw or PurePosixPath(raw).is_absolute() or '..' in PurePosixPath(raw).parts or '\\' in raw or ':' in raw:
                            raise ValueError('invalid ZIP directory')
                        if raw.casefold() in directories:
                            raise ValueError('duplicate ZIP directory')
                        directories.add(raw.casefold())
                        continue
                    declared += info.file_size
                    if declared > config['upload_expanded_bytes'] or info.file_size > config['upload_expanded_bytes']:
                        raise ValueError('ZIP expansion exceeds the server limit')
                    if info.file_size > 1024**2 and info.file_size > max(1, info.compress_size) * 200:
                        raise ValueError('ZIP compression ratio exceeds the server limit')
                    with archive.open(info) as reader:
                        content = reader.read(min(info.file_size, config['upload_expanded_bytes']) + 1)
                    if len(content) != info.file_size:
                        raise ValueError('ZIP member length mismatch')
                    add(info.filename, content)
        else:
            add(filename, data)
    if not members:
        raise ValueError('project contains no source or specification files')
    rows = [inspect_file(name, data) for name, data in sorted(members.items())]
    manifest = {'files': rows, 'bytes': total, 'inspection': 'static; no imports or installation'}
    version = digest(manifest)
    project_id = identifier(project_id) if project_id else uuid.uuid4().hex
    parent = store.root / 'projects' / project_id
    if project_id and parent.exists() and not (parent / 'project.json').is_file():
        raise ValueError('invalid project')
    parent.mkdir(mode=0o700, exist_ok=True)
    if not (parent / 'project.json').exists():
        write_json(parent / 'project.json', {'id': project_id, 'name': str(label or files[0][0])[:160]})
    target = parent / version
    if not target.exists():
        staging = parent / ('.upload-' + uuid.uuid4().hex)
        staging.mkdir(mode=0o700)
        try:
            for name, data in members.items():
                file = beneath(staging / 'files', name)
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(data)
            write_json(staging / 'manifest.json', {**manifest, 'version': version})
            staging.rename(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return project(store, project_id, version)


def project(store, key, version):
    parent = store.root / 'projects' / identifier(key)
    if not re.fullmatch(r'[0-9a-f]{64}', version):
        raise ValueError('invalid project version')
    manifest = read_json(beneath(parent, version + '/manifest.json'))
    if manifest is None:
        raise KeyError('project version not found')
    if manifest.get('version') != version or digest({k: v for k, v in manifest.items() if k != 'version'}) != version:
        raise ValueError('project manifest identity changed after inspection')
    return {**read_json(parent / 'project.json'), **manifest,
            'trusted': read_json(parent / version / 'trust.json', None)}


def trust(store, key, version, acknowledged, source):
    value = project(store, key, version)
    if acknowledged is not True or not isinstance(source, str) or not source.strip():
        raise ValueError('confirm trusted code and record its source')
    if any(row['errors'] for row in value['files']):
        raise ValueError('fix source parsing errors before trusting this version')
    if value['trusted']:
        return value  # A confirmed provenance record is immutable for this exact version.
    record = {'version': version, 'source': source[:1000], 'confirmed_by': 'authenticated workspace owner',
              'scope': 'execute this exact source version in the registered Worker environment'}
    import time
    record['confirmed_at'] = time.time()
    write_json(store.root / 'projects' / key / version / 'trust.json', record)
    return project(store, key, version)


def verify_project(store, key, version, *, require_trust=True):
    value = project(store, key, version)
    if require_trust and any(row['path'].endswith('.py') for row in value['files']) and not value['trusted']:
        raise ValueError('confirm this exact project version as trusted code first')
    base = store.root / 'projects' / key / version / 'files'
    for row in value['files']:
        file = beneath(base, row['path'])
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('uploaded source changed after inspection')
    return value
