"""Offline verification of the full public Benchmark Hub import boundary."""

from dataclasses import replace
import gzip
import hashlib
import json

import pytest

from kernel_tuning_lab.adapters import benchmark_hub as hub
from kernel_tuning_lab.adapters import KernelTunerImportError
from kernel_tuning_lab.models import RecordStatus


def _fixture():
    # Synthetic cache confined to tests; never exported as a real measurement.
    payload = {"device_name": "TEST synthetic device", "kernel_name": "fixture_kernel", "problem_size": [8, 65],
               "tune_params_keys": ["vector_width"], "cache": {
                   "1": {"vector_width": 1, "time": 0.02, "times": [0.019, 0.021]},
                   "2": {"vector_width": 2, "time": "CompilationFailedConfig"},
                   "4": {"vector_width": 4, "time": "RuntimeFailedConfig"},
               }}
    raw = gzip.compress(json.dumps(payload).encode(), mtime=0)
    sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    path = "cachefiles/convolution_milo/A100.json.gz"
    entry = hub.BenchmarkHubEntry("convolution_milo", "A100", path,
                                  f"https://raw.githubusercontent.com/{hub.HUB_REPOSITORY}/{hub.HUB_COMMIT}/{path}", len(raw), sha)
    return entry, raw


def test_catalogue_discovery_retains_t4_but_filters_default_view(tmp_path, monkeypatch):
    entry, raw = _fixture()
    tree = {"sha": "test-tree", "truncated": False, "tree": [
        {"type": "blob", "path": entry.path, "size": len(raw), "sha": entry.git_blob_sha},
        {"type": "blob", "path": entry.path.replace(".json.gz", "_T4.json.gz"), "size": 99, "sha": "t4-sha"},
        {"type": "blob", "path": "currently_unused/cachefiles/legacy.json.gz", "size": 9, "sha": "archive"},
    ]}
    calls = []
    def get(url):
        calls.append(url)
        return json.dumps(tree).encode()
    monkeypatch.setattr(hub, "_get", get)
    assert hub.discover_benchmark_hub(cache_dir=tmp_path) == (entry,)
    assert len(calls) == 1
    all_entries = hub.discover_benchmark_hub(cache_dir=tmp_path, include_t4=True)
    assert len(all_entries) == 2
    assert {row.format for row in all_entries} == {"kernel_tuner_cache", "t4"}
    assert len(calls) == 1  # Cached catalogue remains usable without a network.
    assert all(row.source_commit == hub.HUB_COMMIT for row in all_entries)


def test_download_checks_size_blob_hash_and_retains_original_bytes(tmp_path, monkeypatch):
    entry, raw = _fixture()
    monkeypatch.setattr(hub, "_get", lambda url: raw)
    path = hub.download_benchmark_hub(entry, tmp_path)
    assert path.read_bytes() == raw
    manifest = json.loads(path.with_name(path.name + ".source.json").read_text())
    assert manifest["source_compressed_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["git_blob_sha"] == entry.git_blob_sha
    assert manifest["correctness"] == "unknown"
    assert manifest["local_gpu_rerun"] is False
    assert (tmp_path / manifest["license_file"]).exists()
    monkeypatch.setattr(hub, "_get", lambda url: pytest.fail("cached reuse must not download"))
    assert hub.download_benchmark_hub(entry, tmp_path) == path
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        hub.download_benchmark_hub(entry, tmp_path)


def test_complete_import_preserves_unknown_and_distinct_failures(tmp_path, monkeypatch):
    entry, raw = _fixture()
    monkeypatch.setattr(hub, "_get", lambda url: raw)
    rows = hub.import_benchmark_hub(entry, tmp_path)
    assert len(rows) == 3
    assert [row.status for row in rows] == [RecordStatus.CORRECTNESS_UNKNOWN, RecordStatus.COMPILE_ERROR, RecordStatus.RUNTIME_ERROR]
    assert not any(row.successful for row in rows)
    assert all(row.provenance == "imported" for row in rows)
    assert all(row.metadata["source_complete_file"] for row in rows)
    assert all(row.metadata["source_commit"] == hub.HUB_COMMIT for row in rows)
    assert all(row.metadata["source_manifest"]["source_record_count"] == 3 for row in rows)
    again = hub.import_benchmark_hub(entry, tmp_path)
    assert [row.id for row in rows] == [row.id for row in again]


def test_rejects_mutable_revision_truncated_tree_wrong_size_and_t4(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="immutable"):
        hub.discover_benchmark_hub(commit="main")
    monkeypatch.setattr(hub, "_get", lambda url: json.dumps({"truncated": True}).encode())
    with pytest.raises(ValueError, match="truncated"):
        hub.discover_benchmark_hub()
    entry, raw = _fixture()
    monkeypatch.setattr(hub, "_get", lambda url: raw)
    with pytest.raises(ValueError, match="size mismatch"):
        hub.download_benchmark_hub(replace(entry, size_bytes=len(raw) + 1), tmp_path)
    with pytest.raises(KernelTunerImportError, match="T4 import is not implemented"):
        hub.import_benchmark_hub(replace(entry, format="t4"), tmp_path)
