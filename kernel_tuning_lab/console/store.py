"""Durable queue and event index; measurements stay in the original Runner journals."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import sys
import time
import uuid

from ..models import canonical_json

TERMINAL = {'completed', 'failed', 'interrupted', 'cancelled'}
ACTIVE = {'prechecking', 'queued', 'running', 'cancelling'}


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', value):
        raise ValueError('invalid identifier')
    return value


def beneath(root, relative):
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in str(relative) or '\x00' in str(relative):
        raise PermissionError('path is outside the workspace')
    target = root / path
    if not target.resolve().is_relative_to(root):
        raise PermissionError('path is outside the workspace')
    if any(parent.is_symlink() for parent in (target, *target.parents) if parent != root and parent.is_relative_to(root)):
        raise PermissionError('symbolic links are not admitted')
    return target


class Store:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in ('projects', 'jobs', 'exports', 'worker'):
            (self.root / name).mkdir(mode=0o700, exist_ok=True)
        self.path = self.root / 'control.sqlite'
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, state TEXT NOT NULL, revision INTEGER NOT NULL,
                    name TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                    submit_key TEXT UNIQUE);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL,
                    kind TEXT NOT NULL, payload TEXT NOT NULL, at REAL NOT NULL, dedupe TEXT UNIQUE);
                CREATE INDEX IF NOT EXISTS event_job ON events(job,seq);
                CREATE INDEX IF NOT EXISTS job_state ON jobs(state,created);
            ''')
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('workspace_id', uuid.uuid4().hex))
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('session_secret', secrets.token_hex(32)))
        config = self.root / 'server.json'
        if not config.exists():
            write_json(config, {'schema_version': 1, 'environments': [{
                'id': 'core', 'label': 'Core · CPU', 'python': sys.executable,
                'profile': 'core', 'device': 'cpu', 'plugins': [], 'env': {}, 'read_paths': []}],
                'history_roots': [], 'max_jobs': 128, 'max_observations': 100000,
                'max_wall_time_s': 86400, 'upload_bytes': 32 * 1024**2,
                'upload_expanded_bytes': 128 * 1024**2, 'upload_files': 512,
                'require_filesystem_restriction_for_uploads': True})

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def config(self):
        config = read_json(self.root / 'server.json')
        if config.get('schema_version') != 1:
            raise ValueError('unsupported server configuration')
        return config

    def environment(self, key):
        for env in self.config()['environments']:
            if env['id'] == key:
                if env.get('device') not in ('cpu', 'cuda') or not Path(env['python']).is_absolute():
                    raise ValueError('environment requires an absolute Python path and cpu/cuda device')
                return env
        raise ValueError('environment is not registered on this server')

    def directory(self, key):
        return beneath(self.root / 'jobs', identifier(key))

    def meta(self, key):
        with self.connect() as db:
            row = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return row[0] if row else None

    def event(self, key, kind, value, *, dedupe=None):
        with self.connect() as db:
            cursor = db.execute('INSERT OR IGNORE INTO events(job,kind,payload,at,dedupe) VALUES (?,?,?,?,?)',
                                (key, kind, canonical_json(value), time.time(), dedupe))
            return cursor.lastrowid

    @staticmethod
    def _event_on(db, key, kind, value):
        db.execute('INSERT INTO events(job,kind,payload,at) VALUES (?,?,?,?)',
                   (key, kind, canonical_json(value), time.time()))

    def events(self, key, after=0, limit=200):
        if type(after) is not int or not 0 <= after < 2**63:
            raise ValueError('invalid event cursor')
        self.get(key)
        with self.connect() as db:
            rows = db.execute('SELECT * FROM events WHERE job=? AND seq>? ORDER BY seq LIMIT ?',
                              (key, after, min(limit, 1000))).fetchall()
        return [{'seq': r['seq'], 'kind': r['kind'], 'at': r['at'], 'data': json.loads(r['payload'])} for r in rows]

    @staticmethod
    def decode(row):
        return {key: row[key] for key in ('id', 'state', 'revision', 'name', 'created', 'updated')} | json.loads(row['data'])

    def jobs(self):
        with self.connect() as db:
            return [self.decode(row) for row in db.execute('SELECT * FROM jobs ORDER BY created DESC')]

    def queue(self):
        with self.connect() as db:
            return [self.decode(row) for row in db.execute(
                "SELECT * FROM jobs WHERE state IN ('prechecking','queued','running','cancelling') ORDER BY created")]

    def get(self, key):
        identifier(key)
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (key,)).fetchone()
        if row is None:
            raise KeyError('task not found')
        return self.decode(row)

    def create(self, data):
        key = uuid.uuid4().hex
        name = str(data.get('spec', {}).get('name') or '新实验')[:160]
        now = time.time()
        self.directory(key).mkdir(mode=0o700)
        with self.connect() as db:
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,NULL)',
                       (key, 'draft', 1, name, canonical_json(data), now, now))
            self._event_on(db, key, 'state', {'state': 'draft'})
        return self.get(key)

    def update(self, key, *, expected=None, state=None, emit=None, **changes):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE id=?', (identifier(key),)).fetchone()
            if row is None:
                raise KeyError('task not found')
            if expected is not None and row['state'] not in expected:
                raise ValueError('task changed; refresh before this operation')
            data = json.loads(row['data'])
            data.update(changes)
            db.execute('UPDATE jobs SET state=?,data=?,updated=? WHERE id=?',
                       (state or row['state'], canonical_json(data), time.time(), key))
            if state and state != row['state']:
                self._event_on(db, key, 'state', {'state': state})
            if emit:
                self._event_on(db, key, emit[0], emit[1])
        return self.get(key)

    def edit(self, key, revision, data):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE id=?', (identifier(key),)).fetchone()
            if row is None:
                raise KeyError('task not found')
            if row['state'] in ACTIVE or row['state'] == 'completed' or row['revision'] != revision:
                raise ValueError('this revision cannot be edited; duplicate it to create a new experiment')
            db.execute('UPDATE jobs SET state=?,revision=revision+1,name=?,data=?,updated=? WHERE id=?',
                       ('draft', str(data['spec'].get('name', row['name']))[:160], canonical_json(data), time.time(), key))
            self._event_on(db, key, 'state', {'state': 'draft', 'revision': revision + 1})
        return self.get(key)

    def submit(self, key, plan_sha, submit_key):
        if not isinstance(submit_key, str) or not 8 <= len(submit_key) <= 160:
            raise ValueError('an idempotency key of 8–160 characters is required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE id=?', (identifier(key),)).fetchone()
            if row is None:
                raise KeyError('task not found')
            data = json.loads(row['data'])
            if row['submit_key'] == submit_key and data.get('plan_sha') == plan_sha:
                return self.decode(row)
            if row['state'] != 'ready' or data.get('plan_sha') != plan_sha:
                raise ValueError('precheck this exact revision before submitting')
            try:
                db.execute('UPDATE jobs SET state=?,submit_key=?,updated=? WHERE id=?',
                           ('queued', submit_key, time.time(), key))
                self._event_on(db, key, 'state', {'state': 'queued'})
            except sqlite3.IntegrityError as exc:
                raise ValueError('idempotency key already belongs to another task') from exc
        return self.get(key)
