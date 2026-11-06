"""Incremental native trace tailing, with durable replay identifiers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from .store import read_json


def observation_configuration(trace, output, observation):
    """Resolve one native record by primary key, without scanning measurement history."""
    from ..storage import ExperimentStore
    for parent in trace.parents:
        database = parent / 'measurements.sqlite'
        if database.is_file() and not database.is_symlink() and database.resolve().is_relative_to(output.resolve()):
            try:
                with ExperimentStore(database, read_only=True) as native:
                    record = native.get_record(observation.get('record_id', ''))
                if record is not None and record.config.id == observation.get('config_id'):
                    return {'config': dict(record.config.params), 'dtype': record.operator.dtype,
                            'implementation': record.operator.implementation,
                            'samples_count': len(record.runtime_samples_ms),
                            'dataset': hashlib.sha256(str(parent.resolve()).encode()).hexdigest()[:24]}
            except (OSError, ValueError, sqlite3.DatabaseError):
                pass  # Trace remains usable if the optional record is not yet available.
            return {}
        if parent == output:
            break
    return {}


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
                    'attempt': job.get('execution', {}).get('attempt'), 'trial': trial,
                    'observation': {**value, **observation_configuration(file, output, value)}}, dedupe=token)
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
