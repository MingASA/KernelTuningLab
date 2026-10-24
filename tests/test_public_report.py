"""Public-data trust boundaries and report regression checks (CPU, offline)."""

from collections import Counter
import json
import xml.etree.ElementTree as ET

import pytest

from kernel_tuning_lab.adapters import KernelTunerImportError, bundled_sample_path, load_kernel_tuner_json
from kernel_tuning_lab.analysis import generate_report
from kernel_tuning_lab.models import HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec, RecordStatus, ShapeSpec
from kernel_tuning_lab.storage import ExperimentStore


def test_real_excerpt_preserves_unknown_correctness_and_source_failure():
    records = load_kernel_tuner_json(bundled_sample_path())
    assert len(records) == 24
    assert Counter(record.status for record in records) == {
        RecordStatus.CORRECTNESS_UNKNOWN: 23, RecordStatus.RUNTIME_ERROR: 1,
    }
    assert not any(record.successful for record in records)
    assert all(record.median_ms is None for record in records)
    assert all(record.provenance == "imported" for record in records)
    unknown = [record for record in records if record.status == RecordStatus.CORRECTNESS_UNKNOWN]
    assert all(len(record.runtime_samples_ms) == 32 for record in unknown)
    assert all(record.metadata["correctness_verified"] is False for record in records)
    assert all(record.metadata["source_manifest"]["source_record_count"] == 4362 for record in records)


@pytest.mark.parametrize("evidence", [None, "", "  "])
def test_verified_import_requires_nonempty_evidence(evidence):
    with pytest.raises(KernelTunerImportError, match="correctness_evidence"):
        load_kernel_tuner_json(bundled_sample_path(), correctness="verified", correctness_evidence=evidence)


def test_import_roundtrip_deduplicates_and_retains_unknown_status(tmp_path):
    first = load_kernel_tuner_json(bundled_sample_path())
    second = load_kernel_tuner_json(bundled_sample_path())
    assert [record.id for record in first] == [record.id for record in second]
    export = tmp_path / "measurements.jsonl"
    with ExperimentStore(tmp_path / "source.sqlite") as store:
        assert store.add_many(first) == 24
        assert store.add_many(second) == 0
        assert store.export_jsonl(export) == 24
    with ExperimentStore(tmp_path / "target.sqlite") as store:
        assert store.import_jsonl(export) == {"read": 24, "added": 24, "duplicates": 0}
        assert store.import_jsonl(export) == {"read": 24, "added": 0, "duplicates": 24}
        restored = store.records()
    assert [record.id for record in restored] == [record.id for record in first]
    assert [record.status for record in restored] == [record.status for record in first]
    assert [record.runtime_samples_ms for record in restored] == [record.runtime_samples_ms for record in first]
    assert not any(record.successful for record in restored)


def test_empty_report_explicitly_marks_insufficient_data(tmp_path, monkeypatch):
    import kernel_tuning_lab.analysis.report as implementation
    monkeypatch.setattr(implementation, "_PLOTTING", False)
    report = generate_report(iter(()), tmp_path)
    assert report == tmp_path / "report.md"
    assert "Insufficient data" in report.read_text()
    assert len(list(tmp_path.glob("*.svg"))) == 7
    for path in tmp_path.glob("*.svg"):
        ET.parse(path)
    assert json.loads((tmp_path / "measurement_inventory.json").read_text()) == []


def _mock_records():
    hardware = HardwareSpec("MOCK-unit-fixture", architecture="mock")
    operator = OperatorSpec("softmax_fixture", kind="softmax", implementation="mock_softmax")
    shape = ShapeSpec({"ROWS": 8, "COLS": 65})
    return [MeasurementRecord(hardware, operator, shape, KernelConfig({"block": block}), RecordStatus.SUCCESS,
                              runtime_samples_ms=(latency, latency * 1.01), provenance="mock")
            for block, latency in [(64, 0.02), (128, 0.01)]]


def test_mock_report_and_optional_png_are_labelled(tmp_path):
    report = generate_report(_mock_records(), tmp_path)
    text = report.read_text()
    assert "MOCK / SYNTHETIC DATA" in text
    assert "must not be cited as real GPU performance" in text
    assert "softmax_fixture" in text
    svg = (tmp_path / "config_heatmap.svg").read_text()
    assert "MOCK / SYNTHETIC DATA" in svg
    try:
        import matplotlib  # noqa: F401 -- optional dependency is explicitly supported
    except ImportError:
        assert not list(tmp_path.glob("*.png"))
    else:
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) == 7
        assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in pngs)


def test_public_unknown_is_counted_separately_from_failure(tmp_path, monkeypatch):
    import kernel_tuning_lab.analysis.report as implementation
    monkeypatch.setattr(implementation, "_PLOTTING", False)
    report = generate_report(load_kernel_tuner_json(bundled_sample_path()), tmp_path)
    text = report.read_text()
    assert "correctness-verified successful records: **0**" in text
    assert "| imported | correctness_unknown | 23 |" in text
    assert "| imported | runtime_error | 1 |" in text
    assert "| imported | 1 | 4.17% | 100.00% | 23 |" in text
    assert "4362" in text
    assert "no correctness-verified successful measurements" in text


def test_replay_and_l3_curves_keep_hardware_and_operator_separate(tmp_path, monkeypatch):
    import kernel_tuning_lab.analysis.report as implementation
    monkeypatch.setattr(implementation, "_PLOTTING", False)
    replay = [dict(algorithm="random", hardware_id=hardware, hardware_name=hardware,
                   operator_id=operator, operator_name="softmax" if operator == "softmax" else "gemm", budget=8, evaluations=8,
                   regret=regret, provenance="mock", seed=0, split="test")
              for hardware, operator, regret in [("GPU_A", "operator_gemm_v1", 0.1), ("GPU_B", "operator_gemm_v1", 0.5), ("GPU_A", "softmax", 0.9), ("GPU_A", "operator_gemm_v2", 0.3)]]
    l3 = [dict(method="greedy", hardware_id=row["hardware_id"], hardware_name=row["hardware_name"],
               operator_id=row["operator_id"], operator_name=row["operator_name"], k=1,
               relative_loss=row["regret"], provenance="mock", expected_latency_ms=1.0,
               coverage=1.0, selected_indices=[0], metadata={"matrix_complete": True}) for row in replay]
    generate_report((), tmp_path, replay, l3)
    for filename in ("regret_budget.svg", "k_loss.svg"):
        root = ET.parse(tmp_path / filename).getroot()
        points = root.findall(".//{http://www.w3.org/2000/svg}polyline")
        # Identical display names and identical short-ID prefixes cannot merge
        # distinct operator implementations into one line.
        assert len(points) == 4
        titles = " ".join(node.text or "" for node in root.findall(".//{http://www.w3.org/2000/svg}title"))
        assert "GPU_A" in titles and "GPU_B" in titles
        assert "gemm" in titles and "softmax" in titles
    for prefix in ("regret_budget", "k_loss"):
        cohort_figures = list(tmp_path.glob(prefix + "_*.svg"))
        assert len(cohort_figures) == 4
        for path in cohort_figures:
            root = ET.parse(path).getroot()
            assert len(root.findall(".//{http://www.w3.org/2000/svg}polyline")) == 1


def test_main_report_uses_readable_method_comparisons_per_cohort(tmp_path, monkeypatch):
    import kernel_tuning_lab.analysis.report as implementation
    monkeypatch.setattr(implementation, "_PLOTTING", False)
    replay, l3 = [], []
    for hardware in ("MOCK-GPU-A", "MOCK-GPU-B"):
        for operator in ("gemm", "softmax"):
            common = dict(hardware_id=hardware, hardware_name=hardware, operator_id=operator,
                          operator_name=operator, provenance="mock")
            for method_index, algorithm in enumerate(("random", "grid", "heuristic", "local", "surrogate", "transfer")):
                for budget in (8, 16):
                    replay.append(dict(common, algorithm=algorithm, budget=budget, evaluations=budget,
                                       regret=0.01 * (method_index + 1), seed=0))
            for method_index, method in enumerate(("oracle", "greedy", "greedy_local_swap", "bucket")):
                for k in (1, 2):
                    l3.append(dict(common, method=method, k=k, relative_loss=0.01 * method_index,
                                   expected_latency_ms=1, coverage=1, selected_indices=[0]))
    path = generate_report((), tmp_path, replay, l3)
    text = path.read_text()
    assert "![Regret versus budget](regret_budget.svg)" not in text
    assert "![K versus loss](k_loss.svg)" not in text
    assert "![Search regret:" in text and "![L3 K-loss:" in text
    assert "<summary>All search replay scores</summary>" in text
    for prefix, methods in (("regret_budget", 6), ("k_loss", 4)):
        paths = list(tmp_path.glob(prefix + "_*.svg"))
        assert len(paths) == 4
        for figure in paths:
            root = ET.parse(figure).getroot()
            polylines = root.findall(".//{http://www.w3.org/2000/svg}polyline")
            assert len(polylines) == methods
            assert len({line.get("stroke") for line in polylines}) == methods
            assert float(root.attrib["height"]) <= 600
            title = root.find("{http://www.w3.org/2000/svg}title").text
            assert "MOCK-GPU" in title and ("gemm" in title or "softmax" in title)


def test_public_adapter_accepts_explicit_non_gemm_identity(tmp_path):
    # Synthetic data is confined to this unit test; no GPU result is asserted.
    fixture = {"device_name": "test fixture device", "kernel_name": "elementwise", "problem_size": [8, 65],
               "tune_params_keys": ["vector_width"], "cache": {"4": {"vector_width": 4, "time": 0.02}}}
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture))
    operator = OperatorSpec("layernorm", kind="layernorm", implementation="kernel_tuner", dtype="float32")
    shape = ShapeSpec({"ROWS": 8, "COLS": 65})
    records = load_kernel_tuner_json(path, operator=operator, shape=shape)
    assert records[0].operator == operator
    assert records[0].shape == shape
    assert records[0].config.params == {"vector_width": 4}
    assert records[0].status == RecordStatus.CORRECTNESS_UNKNOWN
    assert records[0].metadata["runtime_statistic"] == "upstream_aggregate_only"
