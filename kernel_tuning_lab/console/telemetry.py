"""Incremental native trace tailing, with durable replay identifiers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .store import read_json


def harvest(store, job):
    output = store.directory(job['id']) / 'output'
    cursors = dict(job.get('trace_cursors', {}))
    if not output.is_dir():
        return
    # Walk only this task's native output; never scan unrelated private experiments.
    for file in output.rglob('observations.jsonl'):
        if file.is_symlink() or 'attempts' in file.relative_to(output).parts:
            continue
        relative = file.relative_to(output).as_posix()
        metadata = file.stat()
        previous = cursors.get(relative, {})
        with file.open('rb') as stream:
            prefix = hashlib.sha256(stream.readline()).hexdigest()
            reset = (previous.get('inode') != metadata.st_ino or previous.get('offset', 0) > metadata.st_size
                     or (previous.get('prefix') and previous['prefix'] != prefix))
            generation = previous.get('generation', 0) + int(reset)
            offset = 0 if reset else previous.get('offset', 0)
            stream.seek(offset)
            trial = read_json(file.parent / 'trial.json', {})
            while True:
                start = stream.tell()
                raw = stream.readline(1024**2)
                if not raw or not raw.endswith(b'\n'):
                    break
                try:
                    value = json.loads(raw)
                except ValueError:
                    break
                token = f'{job["id"]}:{relative}:{generation}:{start}:{hashlib.sha256(raw).hexdigest()}'
                store.event(job['id'], 'observation', {'source': relative, 'generation': generation,
                    'attempt': job.get('execution', {}).get('attempt'), 'trial': trial, 'observation': value}, dedupe=token)
                offset = stream.tell()
        cursors[relative] = {'inode': metadata.st_ino, 'offset': offset, 'prefix': prefix, 'generation': generation}
    phase = read_json(output / 'phase.json', {})
    if phase and phase != job.get('last_phase'):
        store.event(job['id'], 'phase', phase)
    if cursors != job.get('trace_cursors') or phase != job.get('last_phase'):
        store.update(job['id'], trace_cursors=cursors, last_phase=phase)


def native_details(directory):
    from ..web import ViewerCatalog
    directory = Path(directory)
    if not directory.is_dir():
        return {'experiments': [], 'details': []}
    catalog = ViewerCatalog([directory])
    experiments = catalog.scan()
    details = []
    for item in experiments:
        # Archived attempts remain downloadable, but default charts use current native output.
        if '/attempts/' not in catalog.entries[item['id']][1]:
            details.append(catalog.detail(item['id']))
    return {'experiments': experiments, 'details': details}
