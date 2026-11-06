"""Viewer acceptance: real CPU observations, immutable data and local access boundaries."""
import hashlib
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from kernel_tuning_lab.cli import main
from kernel_tuning_lab.web import ViewerCatalog, create_server


@pytest.fixture
def experiment(tmp_path):
    spec = Path(__file__).resolve().parents[1] / "examples/quickstart/cpu.json"
    output = tmp_path / "cpu"
    assert main(["run", "--spec", str(spec), "--output", str(output)]) == 0
    return output


def test_viewer_preserves_real_observations_costs_cache_and_bytes(experiment):
    database = experiment / "measurements.sqlite"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    viewer = ViewerCatalog([experiment])
    key = viewer.scan()[0]["id"]
    detail = viewer.detail(key)
    assert detail["summary"]["observations"] == 4
    assert detail["summary"]["evaluations"] == 2
    assert detail["summary"]["cache_hits"] == 2
    assert len(detail["records"]) == 3  # two physical candidates + independent acceptance
    assert sum(t["observations"] for t in detail["trials"]) == 4
    assert sum(p["cache_hit"] for t in detail["trials"] for p in t["observation_trace"]) == 2
    assert all(t["observation_trace"][-1]["elapsed_wall_time_s"] > 0 for t in detail["trials"])
    raw = json.loads((experiment / "trials.json").read_text())
    assert [t["algorithm_costs_ms"] for t in detail["trials"]] == [t["algorithm_costs_ms"] for t in raw]
    assert all(r["verified"] for r in detail["records"])
    assert viewer.record(key, detail["records"][0]["id"])["provenance"] == "measured"
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_http_cannot_execute_escape_roots_or_accept_foreign_origins(experiment, tmp_path):
    secret = tmp_path / "outside.txt"
    secret.write_text("must stay outside viewer")
    (experiment / "report.md").unlink()
    (experiment / "report.md").symlink_to(secret)
    server = create_server([experiment], port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def denied(route, status, *, method="GET", headers=None):
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + route, method=method, headers=headers or {}), timeout=5)
        assert error.value.code == status
        assert b"must stay outside viewer" not in error.value.read()

    try:
        with urlopen(base + "/api/experiments", timeout=5) as response:
            key = json.load(response)["experiments"][0]["id"]
        denied("/api/experiments/../../outside.txt", 404)
        denied("/api/experiments/" + key + "/report", 403)
        denied("/api/experiments?root=/", 405, method="POST")
        denied("/", 403, headers={"Host": "attacker.example"})
        denied("/api/experiments", 403, headers={"Origin": "https://attacker.example"})
        with urlopen(base + "/", timeout=5) as response:
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
            assert b"KernelTuningLab" in response.read()
        assert secret.read_text() == "must stay outside viewer"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_missing_trace_and_l3_only_outputs_remain_explicit(tmp_path):
    run = tmp_path / "old-run"
    run.mkdir()
    (run / "trials.json").write_text(json.dumps([{"trial_id": "old", "wall_time_s": 10}]))
    l3 = tmp_path / "l3"
    l3.mkdir()
    payload = {"results": [{"k": 1, "configs": [{"tile": 4}], "relative_loss": 0.2}]}
    (l3 / "l3.json").write_text(json.dumps(payload))
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "resource-guard.json").write_text('{"eligible":false}')
    viewer = ViewerCatalog([tmp_path])
    rows = viewer.scan()
    by_name = {row["label"].split("/")[-1]: row for row in rows}
    assert viewer.detail(by_name["old-run"]["id"])["trials"][0]["observation_trace"] == []
    assert viewer.detail(by_name["l3"]["id"])["l3"] == payload
    assert by_name["guard"]["status"] == "not_executed"
