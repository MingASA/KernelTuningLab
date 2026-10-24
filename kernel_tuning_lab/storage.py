"""SQLite measurement journal with normalized dimensions and strict JSONL interchange."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterable, Iterator

from .models import MeasurementRecord, RecordStatus, canonical_json


class ExperimentStore:
    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(f"experiment database does not exist: {self.path}")
            self.connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
            try:
                versions = self.connection.execute("SELECT version FROM schema_info").fetchall()
            except sqlite3.DatabaseError as exc:
                self.connection.close()
                raise ValueError(f"not a Kernel Tuning Lab journal: {self.path}") from exc
            if versions != [(1,)]:
                self.connection.close()
                raise ValueError(f"unsupported database schema: {versions}")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS schema_info (version INTEGER PRIMARY KEY);
            INSERT OR IGNORE INTO schema_info VALUES (1);
            CREATE TABLE IF NOT EXISTS hardware (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS operators (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS shapes (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS configs (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS measurements (
                id TEXT PRIMARY KEY,
                cache_key TEXT NOT NULL,
                hardware_id TEXT NOT NULL REFERENCES hardware(id),
                operator_id TEXT NOT NULL REFERENCES operators(id),
                shape_id TEXT NOT NULL REFERENCES shapes(id),
                config_id TEXT NOT NULL REFERENCES configs(id),
                status TEXT NOT NULL,
                split TEXT NOT NULL,
                provenance TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            CREATE INDEX IF NOT EXISTS measurement_cache ON measurements(cache_key);
            CREATE INDEX IF NOT EXISTS measurement_case ON measurements(hardware_id, operator_id, shape_id);
            CREATE INDEX IF NOT EXISTS measurement_split ON measurements(split, provenance);
        """)
        versions = self.connection.execute("SELECT version FROM schema_info").fetchall()
        if versions != [(1,)]:
            raise ValueError(f"unsupported database schema: {versions}")
        self.connection.commit()

    def __enter__(self) -> ExperimentStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def add(self, record: MeasurementRecord) -> bool:
        if self.read_only:
            raise PermissionError("cannot write to a read-only experiment journal")
        with self.connection:
            return self._insert_record(record)

    def _insert_record(self, record: MeasurementRecord) -> bool:
        data = record.to_dict()
        hardware_id, operator_id, shape_id, config_id = record.hardware.id, record.operator.id, record.shape.id, record.config.id
        for table, key, spec in (("hardware", hardware_id, data["hardware"]),
                                 ("operators", operator_id, data["operator"]),
                                 ("shapes", shape_id, data["shape"]),
                                 ("configs", config_id, data["config"])):
            self.connection.execute(f"INSERT OR IGNORE INTO {table}(id,payload) VALUES (?,?)", (key, canonical_json(spec)))
        conflict = self.connection.execute(
            "SELECT split FROM measurements WHERE hardware_id=? AND operator_id=? AND shape_id=? AND experiment_id=? AND split<>? LIMIT 1",
            (hardware_id, operator_id, shape_id, record.experiment_id, record.split),
        ).fetchone()
        if conflict:
            raise ValueError(f"case already belongs to split {conflict[0]!r} in this experiment")
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO measurements(id,cache_key,hardware_id,operator_id,shape_id,config_id,status,split,provenance,experiment_id,payload) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (record.id, record.cache_key, hardware_id, operator_id, shape_id, config_id,
             record.status.value, record.split, record.provenance, record.experiment_id, canonical_json(data)),
        )
        return cursor.rowcount == 1

    def add_many(self, records: Iterable[MeasurementRecord]) -> int:
        if self.read_only:
            raise PermissionError("cannot write to a read-only experiment journal")
        with self.connection:
            return sum(self._insert_record(record) for record in records)

    def records(self, **filters: str) -> list[MeasurementRecord]:
        return list(self.iter_records(**filters))

    def iter_records(self, **filters: str) -> Iterator[MeasurementRecord]:
        allowed = {"hardware_id", "operator_id", "shape_id", "config_id", "split", "provenance", "experiment_id", "status"}
        if filters.keys() - allowed:
            raise ValueError(f"unsupported filters: {sorted(filters.keys() - allowed)}")
        clauses, values = [], []
        for key, value in filters.items():
            clauses.append(f"{key}=?")
            values.append(value.value if isinstance(value, RecordStatus) else value)
        query = "SELECT payload FROM measurements"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY rowid"
        for row in self.connection.execute(query, values):
            yield MeasurementRecord.from_dict(json.loads(row[0]))

    def get_cached(self, cache_key: str) -> MeasurementRecord | None:
        row = self.connection.execute("SELECT payload FROM measurements WHERE cache_key=? ORDER BY rowid DESC LIMIT 1", (cache_key,)).fetchone()
        return MeasurementRecord.from_dict(json.loads(row[0])) if row else None

    def export_jsonl(self, path: str | Path) -> int:
        target = Path(path)
        if target.resolve() == self.path.resolve():
            raise ValueError("JSONL export must not overwrite its SQLite journal")
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                             prefix=f".{target.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                for record in self.iter_records():
                    stream.write(canonical_json(record.to_dict()) + "\n")
                    count += 1
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return count

    def import_jsonl(self, path: str | Path) -> dict[str, int]:
        if self.read_only:
            raise PermissionError("cannot write to a read-only experiment journal")
        read, added = 0, 0
        # Stream under one transaction: malformed tails and split conflicts roll back every insert.
        with self.connection, Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = MeasurementRecord.from_dict(json.loads(line))
                    added += self._insert_record(record)
                    read += 1
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        return {"read": read, "added": added, "duplicates": read - added}

    def summary(self) -> dict[str, object]:
        counts = self.connection.execute("SELECT provenance,status,COUNT(*) FROM measurements GROUP BY provenance,status ORDER BY provenance,status").fetchall()
        return {"path": str(self.path), "total": sum(row[2] for row in counts),
                "groups": [{"provenance": p, "status": s, "count": n} for p, s, n in counts]}
