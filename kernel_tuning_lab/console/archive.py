"""Workspace snapshots use the existing content archive and restore format."""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tarfile
import tempfile

from ..portable import sha256
from .store import Store, write_json
from contextlib import ExitStack


def snapshot(workspace, output):
    if not (Path(workspace).expanduser() / 'control.sqlite').is_file():
        raise ValueError('select an existing workbench workspace')
    store = Store(workspace)
    output = Path(output).expanduser().resolve()
    if output.exists() or output.is_relative_to(store.root):
        raise ValueError('choose a new archive directory outside the live workspace')
    with ExitStack() as stack:
        for name in ('.service.lock', 'worker/lease'):
            lease = stack.enter_context((store.root / name).open('a+b'))
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('stop this workspace API service and Worker before archiving') from exc
        if store.queue():
            raise ValueError('finish or cancel active tasks before archiving')
        output.mkdir(mode=0o700, parents=True)
        (output / 'archives').mkdir(); (output / 'catalogs').mkdir()
        rows, objects = [], {}
        # SQLite backup includes any remaining WAL without changing its source.
        with tempfile.TemporaryDirectory(prefix='ktlab-workspace-') as temporary:
            temp = Path(temporary)
            for directory, dirs, names in os.walk(store.root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink()
                                 and not (Path(directory) / d / 'pyvenv.cfg').is_file()
                                 and d not in ('node_modules', '.git'))
                for name in sorted(names):
                    file = Path(directory) / name
                    if file.is_symlink():
                        raise ValueError('workspace contains a symlink; preserve it separately instead of following it')
                    if not file.is_file() or name.endswith(('-wal', '-shm')) or file.relative_to(store.root).as_posix() == 'worker/lease':
                        continue
                    source = file
                    if file.suffix in ('.sqlite', '.db', '.sqlite3'):
                        source = temp / (str(len(rows)) + '.sqlite')
                        with sqlite3.connect(file.resolve().as_uri() + '?mode=ro', uri=True) as incoming:
                            with sqlite3.connect(source) as target:
                                incoming.backup(target)
                                if target.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                                    raise ValueError('SQLite snapshot integrity failed')
                    value = sha256(source)
                    rows.append({'path': file.relative_to(store.root).as_posix(), 'sha256': value, 'bytes': source.stat().st_size})
                    objects.setdefault(value, source)
            catalog = {'root': str(store.root), 'files': rows, 'source': 'workspace',
                       'scope': 'private workspace including queue, uploaded source, frozen plans and native experiment journals'}
            archive = output / 'archives/workspace-assets.tar.gz'
            with tarfile.open(archive, 'x:gz', compresslevel=3) as stream:
                raw = json.dumps(catalog).encode()
                info = tarfile.TarInfo('CATALOG.json'); info.size = len(raw)
                stream.addfile(info, io.BytesIO(raw))
                for value, source in sorted(objects.items()):
                    stream.add(source, arcname='objects/' + value, recursive=False)
        write_json(output / 'catalogs/workspace-assets.json', catalog)
        archive_sha = sha256(archive)
        write_json(output / 'catalogs/OBJECT_INDEX.json', {'format': 'ktlab-content-archive-v1',
            'private_workspace': True,
            'packs': [{'source': 'workspace', 'file': archive.name, 'sha256': archive_sha}],
            'object_owner': {value: 'workspace' for value in objects},
            'source_catalogs': {'workspace': 'catalogs/workspace-assets.json'}})
        result = {'status': 'passed', 'files': len(rows), 'objects': len(objects), 'bytes': sum(r['bytes'] for r in rows),
                  'archive_bytes': archive.stat().st_size, 'archive_sha256': archive_sha,
                  'private_credentials': 'includes session state; keep private; serving a relocated restore rotates access',
                  'restore': 'ktlab archive restore --source workspace --vault ARCHIVE --output NEW_WORKSPACE'}
        write_json(output / 'SNAPSHOT_RECEIPT.json', result)
        return result
