"""Exclusive output ownership and recoverable experiment-attempt metadata.

Locks are owned by the OS, so a stale PID file never blocks a later run. The
measurement database remains cumulative; the small files describing each run
are retained separately instead of being silently overwritten on resume.
"""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import threading
import uuid


class OutputDirectoryBusyError(RuntimeError):
    pass


class WorkerInterrupted(Exception):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"supervisor received signal {signum}")


@contextmanager
def supervised_worker(process):
    """Forward cancellation and always reap/terminate the owned worker group."""
    previous = {}

    def interrupt(signum, frame):
        raise WorkerInterrupted(signum)

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupt)
    try:
        yield process
    finally:
        # Restore caller signal behavior even when interruption arrives during
        # startup or before the first phase marker has been written.
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        running = process.poll() is None
        try:
            if os.name == "posix" and getattr(process, "pid", None) is not None:
                # A crashed/reaped worker can leave compiler children in its
                # process group. Clean those even after the leader has exited.
                os.killpg(process.pid, signal.SIGKILL)
            elif running:
                process.kill()
        except ProcessLookupError:
            pass
        if running:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # An unresponsive driver may keep a killed process in kernel
                # space. Its inherited lease still prevents concurrent writes.
                pass


class OutputLease:
    def __init__(self, directory: str | Path, *, filename: str = ".ktlab.lock"):
        self.directory = Path(directory).resolve()
        self.path = self.directory / filename
        self.stream = None
        self.metadata = {"run_id": uuid.uuid4().hex, "pid": os.getpid(),
                         "hostname": socket.gethostname(),
                         "started_at": datetime.now(timezone.utc).isoformat()}

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        self.stream = os.fdopen(descriptor, "r+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.path.stat().st_size == 0:
                    self.stream.write(b"\n")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.seek(0)
            owner = self.stream.read(4096).decode("utf-8", errors="replace").strip()
            self.stream.close()
            self.stream = None
            raise OutputDirectoryBusyError(
                f"output directory is in use: {self.directory}; owner: {owner or 'unknown'}; "
                "choose another output directory or wait for its current run to finish") from exc
        self.stream.seek(0)
        self.stream.write(("\n" + json.dumps(self.metadata, ensure_ascii=False) + "\n").encode("utf-8"))
        self.stream.truncate()
        self.stream.flush()
        return self

    @property
    def subprocess_options(self) -> dict:
        # The inherited descriptor keeps the same flock alive if the supervisor
        # itself is killed while its worker is still writing this directory.
        return {"pass_fds": (self.stream.fileno(),)} if os.name == "posix" else {}

    def __exit__(self, *args):
        # Closing, rather than explicitly unlocking, preserves an inherited
        # worker's ownership until that worker has also exited.
        if self.stream is not None:
            self.stream.close()
            self.stream = None


ATTEMPT_FILES = (
    "run_metadata.json", "run_summary.json", "preflight.json", "acceptance.json", "assets.lock.json",
    "phase.json", "interrupted_attempt.json", "validation_report.json", "validation_records.jsonl",
    "validation_failures.jsonl", "interrupted_record_error.json", "counterexamples.json",
    "source_experiment.json", "experiment.json", "experiment_origin.json", "trials.json",
    "report.md", "worker.log", "trials",
)


def archive_previous_attempt(directory: Path) -> Path | None:
    paths = [directory / name for name in ATTEMPT_FILES if (directory / name).exists()]
    if not paths:
        return None
    previous = {}
    try:
        previous = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    # Only IDs produced by this module become directory names, never arbitrary
    # strings supplied by a plugin or by a manually edited metadata file.
    token = previous.get("run_id") if isinstance(previous, dict) else None
    if not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        token = "legacy-" + uuid.uuid4().hex
    archive = directory / "attempts" / token
    if archive.exists():
        archive = archive.with_name(archive.name + "-" + uuid.uuid4().hex[:8])
    archive.mkdir(parents=True)
    for path in paths:
        path.replace(archive / path.name)
    (archive / "archive.json").write_text(json.dumps({
        "measurement_journal": "../../measurements.sqlite", "journal_scope": "cumulative",
        "notice": "Archived record IDs refer to the retained cumulative journal. This directory preserves attempt metadata, not a duplicate database.",
    }, indent=2) + "\n", encoding="utf-8")
    return archive
