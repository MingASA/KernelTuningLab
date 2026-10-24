"""Discover, verify, and import full public Benchmark Hub cache files.

Network access occurs only when these functions are explicitly called. The
source is pinned to a full Git commit; predictions and local GPU claims are
never manufactured from published timing files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
import urllib.request

from ..models import MeasurementRecord
from .kernel_tuner import KernelTunerImportError, bundled_sample_path, load_kernel_tuner_json


HUB_REPOSITORY = "AutoTuningAssociation/benchmark_hub"
HUB_COMMIT = "ca104e907d22c7a78067c234fb933da731d18a90"


@dataclass(frozen=True)
class BenchmarkHubEntry:
    kernel: str
    gpu: str
    path: str
    url: str
    size_bytes: int
    git_blob_sha: str
    source_commit: str = HUB_COMMIT
    format: str = "kernel_tuner_cache"
    license: str = "MIT"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise ValueError("a full immutable 40-character Git commit SHA is required")
    return value.lower()


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "kernel-tuning-lab/0.1", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _catalogue_file(cache_dir: str | Path, commit: str) -> Path:
    return Path(cache_dir) / f"benchmark_hub_catalogue_{commit}.json"


def discover_benchmark_hub(
    *, commit: str = HUB_COMMIT, include_t4: bool = False,
    cache_dir: str | Path | None = None, refresh: bool = False,
) -> tuple[BenchmarkHubEntry, ...]:
    """List every active ``cachefiles/`` blob at a pinned commit.

    The saved catalogue always retains both original caches and T4 files;
    ``include_t4`` only filters the returned view. Archived and hyperparameter
    directories are separate research collections and are outside this active
    catalogue. Existing catalogue files allow subsequent offline discovery.
    """
    commit = _commit(commit)
    cache_path = _catalogue_file(cache_dir, commit) if cache_dir is not None else None
    if cache_path is not None and cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("source_commit") != commit or payload.get("repository") != HUB_REPOSITORY:
            raise ValueError("cached catalogue identity does not match the requested source")
        entries = tuple(BenchmarkHubEntry(**row) for row in payload["entries"])
    else:
        tree_url = f"https://api.github.com/repos/{HUB_REPOSITORY}/git/trees/{commit}?recursive=1"
        tree = json.loads(_get(tree_url))
        if tree.get("truncated"):
            raise ValueError("upstream Git tree is truncated; refusing an incomplete catalogue")
        entries_list = []
        for item in tree.get("tree", ()):
            path = PurePosixPath(item["path"])
            if item.get("type") != "blob" or len(path.parts) != 3 or path.parts[0] != "cachefiles" or not path.name.endswith(".json.gz"):
                continue
            stem = path.name.removesuffix(".json.gz")
            is_t4 = stem.endswith("_T4")
            gpu = stem.removesuffix("_T4")
            entries_list.append(BenchmarkHubEntry(
                kernel=path.parts[1], gpu=gpu, path=str(path),
                url=f"https://raw.githubusercontent.com/{HUB_REPOSITORY}/{commit}/{path}",
                size_bytes=int(item["size"]), git_blob_sha=item["sha"], source_commit=commit,
                format="t4" if is_t4 else "kernel_tuner_cache", license="MIT" if commit == HUB_COMMIT else "unspecified",
            ))
        entries = tuple(sorted(entries_list, key=lambda row: row.path))
        if not entries:
            raise ValueError("no active Benchmark Hub JSON.gz cache files found")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1, "repository": HUB_REPOSITORY, "source_commit": commit,
                "tree_url": tree_url, "tree_sha": tree.get("sha"), "truncated": False,
                "scope": "active cachefiles/; includes original and T4 files; excludes archived/hyperparameter collections",
                "entries": [entry.to_dict() for entry in entries],
            }
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return tuple(entry for entry in entries if include_t4 or entry.format == "kernel_tuner_cache")


def _verify(entry: BenchmarkHubEntry, raw: bytes) -> str:
    if len(raw) != entry.size_bytes:
        raise ValueError(f"size mismatch for {entry.path}: expected {entry.size_bytes}, got {len(raw)}")
    git_blob_sha = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if git_blob_sha != entry.git_blob_sha:
        raise ValueError(f"Git blob checksum mismatch for {entry.path}")
    return hashlib.sha256(raw).hexdigest()


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".source.json")


def download_benchmark_hub(entry: BenchmarkHubEntry, output_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Download one complete file; verify its tree size and Git blob checksum.

    Already downloaded files are rechecked before reuse. A sidecar records
    SHA-256, commit, URL, format, license, and the unknown correctness boundary.
    The original gzip bytes are retained without reserialization or sampling.
    """
    commit = _commit(entry.source_commit)
    relative = PurePosixPath(entry.path)
    if len(relative.parts) != 3 or relative.parts[0] != "cachefiles" or ".." in relative.parts or relative.is_absolute():
        raise ValueError("entry path must be a relative active cachefiles/kernel/file path")
    expected_url = f"https://raw.githubusercontent.com/{HUB_REPOSITORY}/{commit}/{relative}"
    if entry.url != expected_url:
        raise ValueError("entry URL must match its pinned Benchmark Hub path and commit")
    target = Path(output_dir).joinpath(*relative.parts)
    raw = target.read_bytes() if target.exists() and not overwrite else _get(entry.url)
    sha256 = _verify(entry, raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or overwrite:
        temporary = target.with_name(target.name + ".part")
        temporary.write_bytes(raw)
        temporary.replace(target)
    license_name = "BENCHMARK_HUB_LICENSE.txt" if commit == HUB_COMMIT else f"BENCHMARK_HUB_LICENSE_{commit}.txt"
    license_path = Path(output_dir) / license_name
    if not license_path.exists():
        if commit == HUB_COMMIT:
            license_text = bundled_sample_path().with_name("BENCHMARK_HUB_LICENSE.txt").read_text(encoding="utf-8")
        else:
            license_text = _get(f"https://raw.githubusercontent.com/{HUB_REPOSITORY}/{commit}/LICENSE").decode("utf-8")
        license_path.write_text(license_text, encoding="utf-8")
    manifest = {
        "schema_version": 1, "repository": HUB_REPOSITORY, "commit": commit,
        "source_url": entry.url, "source_path": entry.path, "kernel": entry.kernel, "gpu_label": entry.gpu,
        "source_compressed_sha256": sha256, "git_blob_sha": entry.git_blob_sha,
        "compressed_size_bytes": len(raw), "source_format": entry.format,
        "source_license": entry.license, "license_file": license_path.name,
        "license_url": f"https://github.com/{HUB_REPOSITORY}/blob/{commit}/LICENSE",
        "selection": "complete original upstream file; no sampling", "download_complete": True,
        "correctness": "unknown", "local_gpu_rerun": False,
        "notice": "Published upstream timings; no independent output correctness verification or local GPU measurement.",
    }
    _manifest_path(target).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def import_benchmark_hub(
    entry: BenchmarkHubEntry, output_dir: str | Path, *, split: str = "train",
    experiment_id: str = "benchmark_hub_import",
) -> list[MeasurementRecord]:
    """Download/reuse a full original cache and import unverified observations.

    T4 files are discoverable and downloadable, but require a separate schema
    adapter and are rejected here. Correctness unknown remains unsuccessful;
    upstream invalid/compilation/runtime failure distinctions are preserved.
    """
    if entry.format != "kernel_tuner_cache":
        raise KernelTunerImportError("T4 import is not implemented; select the original kernel_tuner_cache entry")
    path = download_benchmark_hub(entry, output_dir)
    records = load_kernel_tuner_json(path, source_url=entry.url, split=split, experiment_id=experiment_id)
    manifest = json.loads(_manifest_path(path).read_text(encoding="utf-8"))
    manifest.update({"source_record_count": len(records), "sample_record_count": len(records)})
    _manifest_path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [replace(record, metadata={
        **record.metadata, "source_manifest": manifest, "source_commit": entry.source_commit,
        "source_license": entry.license, "source_git_blob_sha": entry.git_blob_sha,
        "source_dataset_kernel": entry.kernel, "source_gpu_label": entry.gpu,
        "source_complete_file": True,
    }) for record in records]
