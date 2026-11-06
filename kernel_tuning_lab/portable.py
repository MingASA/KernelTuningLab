"""CPU-safe environment diagnosis and verified content-archive restoration."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import subprocess
import tarfile


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb', buffering=4*1024**2) as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def profiles():
    return json.loads((Path(__file__).parent/'adapters/data/runtime_profiles.json').read_text())


def doctor(*, devices=False):
    from . import __version__
    versions = {}
    for name in ('kernel-tuning-lab', 'torch', 'triton', 'numpy', 'smac', 'kernel_tuner', 'transformers', 'gemlite', 'flute-kernel'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {'version': __version__, 'python': platform.python_version(), 'platform': platform.platform(),
              'packages': versions, 'profiles': profiles(), 'CUDA_initialized_by_doctor': False}
    if devices:
        command = ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']
        try:
            probe = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
            result['devices'] = {'exit_code': probe.returncode, 'output': probe.stdout.strip(), 'error': probe.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            result['devices'] = {'error': str(error)}
    return result


def _relative(value):
    path = Path(value)
    if path.is_absolute() or not path.parts or '..' in path.parts or '\\' in value:
        raise ValueError('archive paths must be relative and cannot traverse parents')
    return path


def _destination(root, relative):
    path = root/_relative(relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('restore path escapes the destination')
    return path


def _vault(root):
    root = Path(root).resolve()
    index = json.loads((root/'catalogs/OBJECT_INDEX.json').read_text())
    if index.get('format') != 'ktlab-content-archive-v1':
        raise ValueError('unsupported content archive format')
    packs = {row['source']: row for row in index['packs']}
    return root, index, packs


def verify_vault(root):
    root, index, packs = _vault(root)
    checked = set()
    for source, row in packs.items():
        archive = root/'archives'/Path(row['file']).name
        if sha256(archive) != row['sha256']:
            raise ValueError('archive SHA256 mismatch: '+archive.name)
        with archive.open('rb', buffering=4*1024**2) as raw, tarfile.open(fileobj=raw, mode='r|*') as stream:
            for member in stream:
                if member.name == 'CATALOG.json' and member.isfile():
                    continue
                if not member.isfile() or not re.fullmatch(r'objects/[0-9a-f]{64}', member.name):
                    raise ValueError('unexpected archive member')
                digest = member.name[8:]
                if digest not in index['object_owner']:
                    raise ValueError('archive contains an unlisted content object')
                content = hashlib.sha256()
                reader = stream.extractfile(member)
                for block in iter(lambda: reader.read(4*1024**2), b''):
                    content.update(block)
                if content.hexdigest() != digest:
                    raise ValueError('member SHA256 mismatch: '+digest)
                checked.add(digest)
    if checked != set(index['object_owner']):
        raise ValueError('content archive has missing objects')
    return {'status': 'passed', 'archives': len(packs), 'objects': len(checked)}


def restore_vault(root, source, output, *, prefix='', check_sqlite=True):
    root, index, packs = _vault(root)
    declared = index.get('source_catalogs', {})
    if source in declared:
        catalog_file = _destination(root, declared[source])
    else:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', source):raise ValueError('invalid source name')
        matches = list((root/'catalogs').glob(source+'-assets-*.json'))
        if len(matches) != 1:raise ValueError('source catalog must be identified explicitly in OBJECT_INDEX.json')
        catalog_file = matches[0]
    catalog = json.loads(catalog_file.read_text())
    prefix = str(_relative(prefix)) if prefix else ''
    rows = [row for row in catalog['files'] if not prefix or Path(row['path']).is_relative_to(prefix)]
    if not rows:
        raise ValueError('no archived files match this source/prefix')
    output = Path(output).resolve()
    private_workspace = index.get('private_workspace') is True
    output.mkdir(mode=0o700 if private_workspace else 0o777, parents=True, exist_ok=True)
    if private_workspace and os.name == 'posix' and output.stat().st_mode & 0o077:
        raise ValueError('private workspace restore requires an owner-only output directory (mode0700)')
    wanted = defaultdict(list)
    for row in rows:
        _destination(output, row['path'])
        if row['sha256'] not in index['object_owner']:
            raise ValueError('file has only metadata; content was not archived: '+row['path'])
        wanted[row['sha256']].append(row)
    by_pack = defaultdict(set)
    for digest in wanted:
        by_pack[index['object_owner'][digest]].add(digest)
    restored = []
    for owner, digests in by_pack.items():
        archive = root/'archives'/Path(packs[owner]['file']).name
        if sha256(archive) != packs[owner]['sha256']:
            raise ValueError('archive SHA256 mismatch: '+archive.name)
        with archive.open('rb', buffering=4*1024**2) as raw, tarfile.open(fileobj=raw, mode='r|*') as stream:
            for member in stream:
                digest = member.name[8:] if member.name.startswith('objects/') else ''
                if digest not in digests:
                    continue
                if not member.isfile() or member.name != 'objects/'+digest:
                    raise ValueError('content member must be a regular file')
                group = wanted[digest];first = _destination(output, group[0]['path'])
                first.parent.mkdir(parents=True, exist_ok=True)
                if first.exists():
                    if sha256(first) != digest:
                        raise ValueError('restore refuses to overwrite different content: '+str(first))
                else:
                    temporary = first.with_name(first.name+'.restore-partial')
                    with temporary.open('xb') as sink:
                        shutil.copyfileobj(stream.extractfile(member), sink, 4*1024**2)
                    if temporary.stat().st_size != group[0]['bytes'] or sha256(temporary) != digest:
                        raise ValueError('restored content SHA256 mismatch; partial preserved')
                    temporary.rename(first)
                restored.append(first)
                for row in group[1:]:
                    path = _destination(output, row['path']);path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        if sha256(path) != digest:raise ValueError('restore target content differs')
                    else:
                        # Separate copies keep each experiment independently writable.
                        shutil.copyfile(first, path)
                    restored.append(path)
                digests.remove(digest)
                if not digests:break
        if digests:
            raise ValueError('archive is missing required content objects')
    checked = 0
    if check_sqlite:
        for file in restored:
            if file.suffix not in ('.sqlite', '.sqlite3', '.db'):continue
            with sqlite3.connect(file.as_uri()+'?mode=ro', uri=True) as database:
                if database.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                    raise ValueError('restored SQLite integrity check failed: '+str(file))
            checked += 1
    receipt = {'status': 'passed', 'source': source, 'prefix': prefix, 'files': len(restored),
               'bytes': sum(row['bytes'] for row in rows), 'SQLite_checked': checked, 'output': str(output),
               'experiments_executed': 0}
    (output/'RESTORE_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt
