"""Reproduce the bundled public-data excerpt from its pinned primary source.

Run ``python -m kernel_tuning_lab.adapters.fetch_sample``. Only this explicit
command uses the network; ordinary imports and the mock demo remain offline.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

from .kernel_tuner import bundled_sample_path


UPSTREAM_COMMIT = "ca104e907d22c7a78067c234fb933da731d18a90"
UPSTREAM_BASE = f"https://raw.githubusercontent.com/AutoTuningAssociation/benchmark_hub/{UPSTREAM_COMMIT}/"
UPSTREAM_PATH = "cachefiles/convolution_milo/A100.json.gz"


def fetch_benchmark_hub_sample(output_path: str | Path | None = None, *, max_records: int = 24) -> Path:
    """Download a pinned real cache and retain evenly spaced original entries."""
    if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
        raise ValueError("max_records must be a positive integer")
    source_url = UPSTREAM_BASE + UPSTREAM_PATH
    with urllib.request.urlopen(source_url, timeout=60) as response:
        compressed = response.read()
    decoded = gzip.decompress(compressed)
    payload = json.loads(decoded)
    entries = list(payload["cache"].items())
    count = min(max_records, len(entries))
    indices = [0] if count == 1 else [i * (len(entries) - 1) // (count - 1) for i in range(count)]
    payload["cache"] = {entries[i][0]: entries[i][1] for i in indices}
    payload["_kernel_tuning_lab_source"] = {
        "repository": "AutoTuningAssociation/benchmark_hub", "commit": UPSTREAM_COMMIT,
        "source_url": source_url, "source_license": "MIT",
        "source_compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "source_uncompressed_sha256": hashlib.sha256(decoded).hexdigest(),
        "source_record_count": len(entries), "sample_record_count": count,
        "selection": "evenly spaced original insertion-order indices; floor(i*(N-1)/(n-1))",
        "source_indices": indices, "correctness": "unknown",
        "notice": "Real upstream timings, excerpt only. No local GPU rerun or independent correctness verification.",
        "source_input_url": UPSTREAM_BASE + "kernels/convolution_milo.json",
    }
    output = Path(output_path) if output_path is not None else bundled_sample_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with urllib.request.urlopen(UPSTREAM_BASE + "LICENSE", timeout=60) as response:
        license_text = response.read().decode("utf-8")
    output.with_name("BENCHMARK_HUB_LICENSE.txt").write_text(license_text, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(fetch_benchmark_hub_sample())
