"""Loopback-only, read-only viewer for the existing experiment journals."""
from __future__ import annotations

import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import math
import os
import sqlite3
from pathlib import Path
from urllib.parse import urlsplit

from .measurement_context import source_context_id
from .models import MeasurementRecord, stable_id
from .storage import ExperimentStore

_MARKERS = {"measurements.sqlite", "measurements.jsonl", "records.jsonl", "trials.json",
            "trials.csv", "run_summary.json", "l3.json", "resource-guard.json"}
_SKIP = {".git", "__pycache__", "node_modules", "site-packages", "wheelhouse", "archives",
         ".cache", "cache", "objects", ".tox"}


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _finite(value):
    """Old reports may contain non-finite diagnostics; JSON never emits NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


class ViewerCatalog:
    def __init__(self, roots):
        self.roots = tuple(dict.fromkeys(Path(p).expanduser().resolve(strict=True) for p in roots))
        if not self.roots or any(not p.is_dir() for p in self.roots):
            raise ValueError("select at least one existing experiment directory with --root")
        self.entries = {}

    def _admitted(self, path):
        resolved = path.resolve(strict=True)
        if not any(resolved.is_relative_to(root) for root in self.roots):
            raise PermissionError("file outside selected experiment roots")
        return resolved

    def _file(self, path, name):
        candidate = path / name
        return self._admitted(candidate) if candidate.exists() else candidate

    def scan(self):
        entries = {}
        for root in self.roots:
            for directory, dirs, names in os.walk(root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if d not in _SKIP and not d.startswith(".venv")
                                 and not (Path(directory) / d / "pyvenv.cfg").exists()
                                 and not (Path(directory) / d).is_symlink())
                if not _MARKERS.intersection(names):
                    continue
                path = self._admitted(Path(directory))
                key = hashlib.sha256(str(path).encode()).hexdigest()[:24]
                label = str(Path(root.name) / path.relative_to(root))
                entries[key] = (path, label)
        self.entries = entries
        rows = []
        for key, (path, label) in sorted(entries.items(), key=lambda item: item[1][1]):
            errors = []
            try:
                summary = _json(self._file(path, "run_summary.json"), {})
                guard = _json(self._file(path, "resource-guard.json"), {})
                metadata = _json(self._file(path, "run_metadata.json"), {})
                spec = _json(self._file(path, "experiment.json"), {})
                if not isinstance(summary, dict) or not isinstance(spec, dict):
                    raise ValueError("summary/spec must be JSON objects")
            except (ValueError, OSError, TypeError) as exc:
                summary, guard, metadata, spec = {}, {}, {}, {}
                errors.append(str(exc))
            rows.append({"id": key, "label": label, "status": summary.get("status") or
                         ("not_executed" if guard and not guard.get("eligible", False) else "available"),
                         "experiment": summary.get("experiment") or spec.get("name") or path.name,
                         "hardware": spec.get("hardware"), "evaluations": summary.get("evaluations"),
                         "observations": summary.get("observations"), "cache_hits": summary.get("cache_hits"),
                         "GPU_validated": summary.get("physical_gpu_validated"),
                         "files": sorted(n for n in _MARKERS if (path / n).is_file()), "errors": errors})
        return rows

    def _path(self, key):
        if key not in self.entries:
            raise KeyError("unknown experiment; refresh the experiment list")
        return self._admitted(self.entries[key][0])

    def _records(self, path, errors):
        db = self._file(path, "measurements.sqlite")
        if db.is_file():
            try:
                with ExperimentStore(db, read_only=True) as store:
                    return store.records()
            except (ValueError, OSError, sqlite3.DatabaseError) as exc:
                errors.append("SQLite: " + str(exc))
        records = []
        for name in ("measurements.jsonl", "records.jsonl"):
            journal = self._file(path, name)
            if not journal.is_file():
                continue
            with journal.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        records.append(MeasurementRecord.from_dict(json.loads(line)))
                    except (ValueError, TypeError, KeyError) as exc:
                        errors.append(f"{name}:{line_number}: {exc}")
            break
        return records

    def detail(self, key):
        path = self._path(key)
        errors = []

        def read(name, default):
            try:
                return _json(self._file(path, name), default)
            except (ValueError, OSError) as exc:
                errors.append(name + ": " + str(exc))
                return default

        records = self._records(path, errors)
        by_id = {record.id: record for record in records}
        rows, cohorts = [], {}
        for record in records:
            # Keep dtype, seed, implementation, hardware and timing/source identity distinct.
            cohort = stable_id([record.hardware.id, record.operator.id, record.shape.id,
                                record.operator.dtype, record.seed, source_context_id(record)])
            info = {"hardware": record.hardware.name, "operator": record.operator.name,
                    "implementation": record.operator.implementation, "dtype": record.operator.dtype,
                    "shape": record.shape.dims, "seed": record.seed, "provenance": record.provenance,
                    "mode": record.measurement_mode, "context_id": source_context_id(record)}
            cohorts[cohort] = info
            rows.append({"id": record.id, "cohort": cohort, **info, "config_id": record.config.id,
                         "params": record.config.params, "status": record.status.value,
                         "verified": record.correctness_verified, "latency_ms": record.median_ms,
                         "samples": len(record.runtime_samples_ms), "compile_ms": record.compile_time_ms,
                         "wall_s": record.wall_time_s, "failure": record.failure_reason,
                         "split": record.split, "experiment_id": record.experiment_id})
        trials = read("trials.json", [])
        if not isinstance(trials, list):
            errors.append("trials.json must be a list")
            trials = []
        for trial in trials:
            references = [by_id[r] for r in trial.get("record_ids", []) if r in by_id]
            trial["cohorts"] = sorted({stable_id([r.hardware.id, r.operator.id, r.shape.id,
                                                 r.operator.dtype, r.seed, source_context_id(r)])
                                       for r in references})
            # Never reconstruct wall clocks or correctness from requested configurations.
            trial["observation_trace"] = trial.get("observation_trace", [])
        csv_rows = []
        if not trials and (path / "trials.csv").is_file():
            with self._file(path, "trials.csv").open(encoding="utf-8", newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
        l3 = read("l3.json", {})
        if isinstance(l3, list):
            l3 = {"results": l3}
        return _finite({"id": key, "label": self.entries[key][1], "records": rows, "cohorts": cohorts,
                        "trials": trials, "csv_trials": csv_rows, "summary": read("run_summary.json", {}),
                        "acceptance": read("acceptance.json", {}), "guard": read("resource-guard.json", {}),
                        "l3": l3, "errors": errors, "report_available": (path / "report.md").is_file()})

    def record(self, key, record_id):
        path = self._path(key)
        errors = []
        for record in self._records(path, errors):
            if record.id == record_id:
                return _finite({"id": record.id, **record.to_dict()})
        raise KeyError("record not found")

    def report(self, key):
        return self._file(self._path(key), "report.md").read_bytes()


def create_server(roots, *, port=8765):
    if not 0 <= port <= 65535:
        raise ValueError("invalid localhost port")
    catalog = ViewerCatalog(roots)
    catalog.scan()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _send(self, status, body, content_type="application/json; charset=utf-8", attachment=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                             "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            if attachment:
                self.send_header("Content-Disposition", 'attachment; filename="report.md"')
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status, obj):
            self._send(status, json.dumps(obj, ensure_ascii=False, allow_nan=False).encode())

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            port = self.server.server_port
            allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in allowed_hosts or (
                self.headers.get("Origin") and self.headers["Origin"] not in
                {"http://" + host for host in allowed_hosts}
            ):
                return self._json(403, {"error": "localhost origin required"})
            route = urlsplit(self.path).path
            try:
                if route == "/api/experiments":
                    return self._json(200, {"experiments": catalog.scan()})
                parts = route.strip("/").split("/")
                if parts[:2] == ["api", "experiments"] and len(parts) >= 3:
                    if len(parts) == 3:
                        return self._json(200, catalog.detail(parts[2]))
                    if len(parts) == 4 and parts[3] == "report":
                        return self._send(200, catalog.report(parts[2]), "text/markdown; charset=utf-8", True)
                    if len(parts) == 5 and parts[3] == "records":
                        return self._json(200, catalog.record(parts[2], parts[4]))
                static = {"/": ("index.html", "text/html; charset=utf-8"),
                          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/style.css": ("style.css", "text/css; charset=utf-8")}
                if route in static:
                    name, mime = static[route]
                    return self._send(200, files("kernel_tuning_lab").joinpath("web_assets", name).read_bytes(), mime)
                self._json(404, {"error": "not found"})
            except KeyError as exc:
                self._json(404, {"error": str(exc)})
            except PermissionError:
                self._json(403, {"error": "file outside selected experiment roots"})
            except (OSError, ValueError, TypeError) as exc:
                self._json(422, {"error": str(exc)})

        def do_POST(self):
            self._json(405, {"error": "read-only viewer"})

        do_PUT = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.catalog = catalog
    return server


def serve(roots, port=8765):
    server = create_server(roots, port=port)
    print(f"KernelTuningLab · http://localhost:{server.server_port} · {len(server.catalog.entries)} experiments", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return {"status": "stopped"}
