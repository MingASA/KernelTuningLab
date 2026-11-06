"""One-command research, real CPU execution, failure isolation and provenance."""

import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import time
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def run_cli(*args, timeout=60):
    return subprocess.run([sys.executable, "-m", "kernel_tuning_lab", *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def cpu_spec():
    return {"schema_version": 1, "kind": "research", "name": "cpu-research",
        "plugins": [str(ROOT / "examples/research/stride_search.py")],
        "algorithms": ["example/stride@1.0.0", "random@1.0.0"],
        "dtypes": ["float32"], "seeds": [7],
        "budgets": [{"max_evaluations": 2, "max_wall_time_s": 10}],
        "defaults": {"validation": {"max_configs": 4, "max_cases": 0, "max_wall_time_s": 10,
                                    "trust_claims": False}, "warmup": 0, "repetitions": 3},
        "cases": [{"name": "gemm", "operator": "gemm@1.0.0", "kernels": ["python/gemm@1.0.0"],
                   "shapes": [{"M": 2, "N": 3, "K": 5}, {"M": 3, "N": 5, "K": 2}]},
                  {"name": "softmax", "operator": "softmax@1.0.0", "kernels": ["python/softmax@1.0.0"],
                   "shapes": [{"M": 2, "N": 7}]}]}


def test_asset_trial_records_step_costs_on_cache_resume(tmp_path):
    from kernel_tuning_lab.asset_runner import execute_asset_experiment
    spec = write(tmp_path / "asset.json", {"schema_version": 2, "name": "trace-cost",
        "operator": "gemm", "kernels": ["python/gemm"], "algorithms": ["random"], "dtype": "float32",
        "shapes": [{"M": 2, "N": 3, "K": 5}], "seed": 7, "warmup": 0, "repetitions": 3,
        "budget": {"max_evaluations": 2, "max_wall_time_s": 10},
        "validation": {"max_configs": 2, "max_cases": 0, "max_wall_time_s": 10, "trust_claims": False}})
    output = tmp_path / "run"
    first = execute_asset_experiment(spec, output)
    assert first["status"] == "passed"
    trial = json.loads((output / "trials.json").read_text())[0]
    assert trial["trace_schema_version"] == 1
    assert len(trial["observation_trace"]) == trial["observations"] == 2
    assert all(not row["cache_hit"] for row in trial["observation_trace"])
    second = execute_asset_experiment(spec, output)
    assert second["evaluations"] == 0 and second["cache_hits"] == 2
    cached = json.loads((output / "trials.json").read_text())[0]
    assert cached["record_ids"] == trial["record_ids"]
    assert cached["physical_compile_time_ms"] == cached["physical_measurement_wall_time_s"] == 0
    assert cached["source_measurement_wall_time_s"] > 0


def test_research_cli_executes_multiple_cases_and_resumes_without_extra_measurements(tmp_path):
    spec = write(tmp_path / "research.json", cpu_spec())
    output = tmp_path / "study"
    result = run_cli("research", "--spec", spec, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Research: 4 jobs' in result.stderr and '[1/4]' in result.stderr
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "passed"
    assert report["trial_count"] == 6
    assert not report["physical_gpu_validated"]
    assert {t["operator"] for t in report["trials"]} == {"gemm", "softmax"}
    assert report["comparison_groups"]
    assert report["algorithm_cost_trace_complete"]
    assert report["algorithm_costs_ms"]["proposal_ms"] > 0
    for key, total in report["algorithm_costs_ms"].items():
        assert total == pytest.approx(sum(t["algorithm_costs_ms"][key] for t in report["trials"]))
    assert "algorithm_diagnostics" in (output / "trials.csv").read_text()
    assert list((output / "figures").glob("*.svg"))
    assert (output / "observations.csv").is_file() and (output / "failures.jsonl").is_file()
    rerun = run_cli("research", "--spec", spec, "--output", output)
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    resumed = json.loads((output / "summary.json").read_text())
    assert resumed["research_evaluations"] == 0
    assert resumed["cache_hits"] == resumed["observations"] == report["observations"]
    assert list((output / "attempts").glob("*/summary.json"))
    assert list((output / "runs").glob("*/attempts/*/trials.json"))
    for old in (output / "attempts").glob("*/report.md"):
        for href in re.findall(r"\[attempt\]\(([^)]+)\)", old.read_text()):
            assert (old.parent / href).is_file(), href


def test_research_dry_run_counts_all_dimensions_without_loading_plugins(tmp_path):
    spec = cpu_spec()
    spec["plugins"] = ["absent_plugin.py"]
    spec["seeds"] = [0, 1]
    spec["dtypes"] = ["float32", "float64"]
    spec["budgets"] *= 1
    spec["budgets"].append({"max_evaluations": 4, "max_wall_time_s": 10})
    path = write(tmp_path / "spec.json", spec)
    result = run_cli("research", "--spec", path, "--output", tmp_path / "out", "--write-only")
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((tmp_path / "out/research_manifest.json").read_text())
    assert len(manifest["jobs"]) == 32
    assert manifest["planned"]["trials"] == 48
    assert manifest["planned"]["max_observations"] == 144
    assert manifest["planned"]["max_validation_evaluations"] == 128
    assert not list((tmp_path / "out").rglob("measurements.sqlite"))


@pytest.mark.parametrize("update", [{"seeds": [True]}, {"seeds": [1, 1]},
                                     {"budgets": [{"max_evaluations": 4}]}, {"surprise": 1},
                                     {"algorithms": ["random", {"asset": "random", "options": {}}]}])
def test_research_rejects_ambiguous_or_unbounded_configuration(tmp_path, update):
    spec = {**cpu_spec(), **update}
    result = run_cli("research", "--spec", write(tmp_path / "spec.json", spec), "--output", tmp_path / "out")
    assert result.returncode == 2


def test_failed_algorithm_does_not_block_baseline(tmp_path):
    plugin = tmp_path / "broken.py"
    plugin.write_text('''from kernel_tuning_lab.sdk import SearchAlgorithmDefinition, KernelConfig
class Bad:
    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        return [] if budget.exhausted(0, 0) else [KernelConfig({"not_a_parameter": 1})]
def create(seed=0): return Bad()
ASSETS = [SearchAlgorithmDefinition("broken", "1", create)]
''')
    spec = cpu_spec()
    spec.update(plugins=[str(plugin)], algorithms=["broken@1", "random@1.0.0"])
    result = run_cli("research", "--spec", write(tmp_path / "spec.json", spec), "--output", tmp_path / "out")
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads((tmp_path / "out/summary.json").read_text())
    assert report["job_statuses"] == {"failed": 2, "passed": 2}
    assert any(t["algorithm"] == "random@1.0.0" and t["observations"] == 2 for t in report["trials"])
    assert all(any(a["algorithm"] == "broken@1" and a["n"] == 0 and a["requested_seeds"] == [7]
                   for a in group["aggregates"]) for group in report["comparison_groups"])
    assert any(json.loads(line)["kind"] == "job" for line in (tmp_path / "out/failures.jsonl").read_text().splitlines())


def test_unsupported_dtype_is_explicit_and_other_jobs_still_run(tmp_path):
    spec = cpu_spec()
    spec["dtypes"] = ["float32", "float16"]
    result = run_cli("research", "--spec", write(tmp_path / "spec.json", spec), "--output", tmp_path / "out")
    assert result.returncode == 3, result.stdout + result.stderr
    report = json.loads((tmp_path / "out/summary.json").read_text())
    assert report["job_statuses"] == {"passed": 4, "not_applicable": 4}
    assert {t["status"] for t in report["trials"] if t["dtype"] == "float16"} == {"not_applicable"}


def test_report_only_needs_no_plugin_and_keeps_timing_protocols_separate(tmp_path):
    spec = cpu_spec()
    spec["cases"] = [spec["cases"][0]]
    spec["cases"][0]["shapes"] = [spec["cases"][0]["shapes"][0]]
    spec["cases"].append({**spec["cases"][0], "name": "different-protocol", "repetitions": 5})
    path = write(tmp_path / "spec.json", spec)
    out = tmp_path / "out"
    assert run_cli("research", "--spec", path, "--output", out).returncode == 0
    before = json.loads((out / "summary.json").read_text())
    assert len(before["comparison_groups"]) == 2
    # Destroy the source spec/plugin references: reporting is archive-only.
    path.unlink()
    manifest = json.loads((out / "research_manifest.json").read_text())
    for job in manifest["jobs"]:
        job["experiment"]["plugins"] = ["missing.py"]
    write(out / "research_manifest.json", manifest)
    reported = run_cli("research", "--report-only", "--output", out)
    assert reported.returncode == 0, reported.stdout + reported.stderr
    after = json.loads((out / "summary.json").read_text())
    assert before["observations"] == after["observations"]
    assert before["run_id"] == after["run_id"]


def test_report_corruption_cannot_retain_success_flag(tmp_path):
    spec = cpu_spec()
    out = tmp_path / "out"
    assert run_cli("research", "--spec", write(tmp_path / "spec.json", spec), "--output", out).returncode == 0
    store = next((out / "runs").glob("*/measurements.sqlite"))
    store.rename(store.with_suffix(".retained"))
    result = run_cli("research", "--report-only", "--output", out)
    assert result.returncode == 2
    summary = json.loads((out / "summary.json").read_text())
    assert summary["phase"] == "report" and summary["physical_gpu_validated"] is False


def test_comparisons_separate_algorithm_options_and_full_budgets(tmp_path):
    import xml.etree.ElementTree as ET

    spec = cpu_spec()
    spec["cases"] = [spec["cases"][0]]
    spec["cases"][0]["shapes"] = [spec["cases"][0]["shapes"][0]]
    spec["algorithms"] = [{"asset": "example/stride@1.0.0", "options": {"probe_limit": n}} for n in (128, 256)]
    spec["budgets"] = [{"max_evaluations": 2, "max_wall_time_s": n} for n in (10, 11)]
    out = tmp_path / "out"
    result = run_cli("research", "--spec", write(tmp_path / "spec.json", spec), "--output", out)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / "summary.json").read_text())
    assert len(report["comparison_groups"]) == 1
    aggregates = report["comparison_groups"][0]["aggregates"]
    assert len(aggregates) == 4 and len({a["series_id"] for a in aggregates}) == 2
    for path in (out / "figures").glob("performance-*.svg"):
        labels = " ".join(ET.parse(path).getroot().itertext())
        assert "probe_limit" in labels and "128" in labels and "256" in labels
    text = (out / "report.md").read_text()
    assert "2 / 10" in text and "2 / 11" in text


def test_missing_cuda_is_deferred_and_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    spec = cpu_spec()
    spec.update(dtypes=["float16"], algorithms=["random@1.0.0"])
    spec["cases"] = [{"name": "cuda", "operator": "gemm", "kernels": ["triton/gemm"],
                       "shapes": [{"M": 3, "N": 5, "K": 7}]}]
    result = run_cli("research", "--spec", write(tmp_path / "spec.json", spec), "--output", tmp_path / "out")
    assert result.returncode == 3, result.stdout + result.stderr
    report = json.loads((tmp_path / "out/summary.json").read_text())
    assert report["job_statuses"] == {"deferred": 1}
    assert report["observations"] == 0 and report["physical_gpu_validated"] is False


def slow_spec(tmp_path, *, timeout=0.4, marker=None):
    flag = tmp_path / "hang"
    flag.touch()
    marker = marker or tmp_path / "worker-pid"
    plugin = tmp_path / "slow.py"
    plugin.write_text('''from pathlib import Path
import time, os
from kernel_tuning_lab.sdk import OperatorDefinition, KernelDefinition, SpaceDefinition, ParameterSpec
def inputs(ctx, seed):
    ctx.allocate_output("out", (1,1))
    return {"x": [[1.0]]}
def reference(ctx): return {"out": [[1.0]]}
def build(ctx, tile):
    def launch():
        if ctx.seed == 7 and tile == 2 and Path(FLAG).exists():
            Path(MARKER).write_text(str(os.getpid()))
            time.sleep(20)
        ctx.outputs["out"][0][0] = 1.0
    return launch
op = OperatorDefinition("slow", "1", ("M", "N"), ("float32",), inputs, reference, {"float32": (0., 0.)})
kernel = KernelDefinition("slow/cpu", "1", op.ref, build,
    SpaceDefinition({"tile": ParameterSpec((1,2))}), runtime="python", dtypes=("float32",))
ASSETS = [op, kernel]
'''.replace("FLAG", repr(str(flag))).replace("MARKER", repr(str(marker))))
    spec = cpu_spec()
    spec.update(plugins=[str(plugin)], algorithms=["grid@1.0.0"],
                budgets=[{"max_evaluations": 2, "max_wall_time_s": timeout}])
    spec["defaults"].update(repetitions=1, validation={"max_configs": 1, "max_cases": 0, "max_wall_time_s": 10})
    spec["cases"] = [{"name": "slow", "operator": "slow@1", "kernels": ["slow/cpu@1"], "shapes": [{"M": 1, "N": 1}]}]
    return write(tmp_path / "slow.json", spec), flag, marker


def test_timeout_keeps_partial_trace_cost_and_retries_only_incomplete_measurement(tmp_path):
    spec, flag, _ = slow_spec(tmp_path)
    out = tmp_path / "out"
    result = run_cli("research", "--spec", spec, "--output", out)
    assert result.returncode == 2, result.stdout + result.stderr
    partial = json.loads((out / "summary.json").read_text())
    assert partial["job_statuses"] == {"timeout": 1}
    assert partial["observations"] == 1 and partial["research_evaluations"] == 2
    assert partial["validation_evaluations"] == 1
    assert len(partial["incomplete_attempts"]) == 1 and not partial["cost_trace_complete"]
    assert partial["physical_compile_time_ms"] is None and partial["unknown_compile_cost_attempts"] == 1
    assert partial["physical_measurement_wall_time_s"] > 0.2
    flag.unlink()
    recovered = run_cli("research", "--spec", spec, "--output", out)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    final = json.loads((out / "summary.json").read_text())
    assert final["observations"] == 2 and final["cache_hits"] == final["research_evaluations"] == 1


def test_operator_dtype_capability_is_not_applicable_even_with_unrestricted_kernel(tmp_path):
    path, _, _ = slow_spec(tmp_path)
    plugin = tmp_path / "slow.py"
    plugin.write_text(plugin.read_text().replace('dtypes=("float32",)', 'dtypes=()'))
    spec = json.loads(path.read_text())
    spec["dtypes"] = ["float64"]
    write(path, spec)
    result = run_cli("research", "--spec", path, "--output", tmp_path / "out")
    assert result.returncode == 3, result.stdout + result.stderr
    report = json.loads((tmp_path / "out/summary.json").read_text())
    assert report["job_statuses"] == {"not_applicable": 1} and report["observations"] == 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX inherited output lease")
def test_orphan_worker_keeps_research_output_locked(tmp_path):
    spec, _, marker = slow_spec(tmp_path, timeout=30)
    out = tmp_path / "out"
    process = subprocess.Popen([sys.executable, "-m", "kernel_tuning_lab", "research", "--spec", str(spec),
                                "--output", str(out)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    worker = None
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.02)
        assert marker.exists()
        worker = int(marker.read_text())
        process.kill()
        process.wait(timeout=5)
        before = (out / "research_manifest.json").read_bytes()
        blocked = run_cli("research", "--spec", spec, "--output", out)
        assert blocked.returncode == 2 and "in use" in blocked.stdout
        assert (out / "research_manifest.json").read_bytes() == before
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if worker:
            try:
                os.killpg(worker, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_optional_optimizer_feedback_flushes_final_observation_and_keeps_measurements_on_feedback_error(tmp_path):
    from kernel_tuning_lab.asset_runner import execute_asset_experiment
    plugin=tmp_path/'hook.py'
    plugin.write_text('''from kernel_tuning_lab.sdk import SearchAlgorithmDefinition
class Hook:
 def __init__(self,seed=0,fail=False):self.count=0;self.bound=False;self.fail=fail
 def bind_trial(self,output_dir,identity):self.bound=True
 @property
 def diagnostics(self):return {'feedback_count':self.count,'bound':self.bound}
 def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
  if budget.exhausted(0,0):return []
  seen={r.config.id for r in history}|set(history.attempted_config_ids)
  return [c for c in space.iter_configs() if c.id not in seen and space.validate(c,hardware,operator,shape).valid][:min(batch_size,budget.max_evaluations)]
 def observe(self,record,cache_hit=False):
  assert self.bound
  self.count+=1
  if self.fail:raise RuntimeError('unit feedback failure')
ASSETS=(SearchAlgorithmDefinition('unit/hook','1.0.0',Hook),)
''')
    base={'schema_version':2,'name':'feedback-hook','operator':'gemm','kernels':['python/gemm'],
        'plugins':[str(plugin)],'algorithms':[{'asset':'unit/hook@1.0.0'}],'dtype':'float32',
        'shapes':[{'M':2,'N':3,'K':5}],'seed':7,'warmup':0,'repetitions':3,
        'budget':{'max_evaluations':2,'max_wall_time_s':10},
        'validation':{'max_configs':2,'max_cases':0,'max_wall_time_s':10,'trust_claims':False}}
    spec=write(tmp_path/'spec.json',base);output=tmp_path/'run'
    assert execute_asset_experiment(spec,output)['status']=='passed'
    first=json.loads((output/'trials.json').read_text())[0]
    assert first['algorithm_diagnostics']['optimizer']=={'feedback_count':2,'bound':True}
    assert set(first['algorithm_costs_ms'])=={'factory_ms','proposal_ms','feedback_ms','shutdown_ms'}
    assert execute_asset_experiment(spec,output)['status']=='passed'
    resumed=json.loads((output/'trials.json').read_text())[0]
    assert resumed['cache_hits']==2 and resumed['physical_evaluations']==0
    assert resumed['algorithm_diagnostics']['optimizer']['feedback_count']==2
    base['algorithms'][0]['options']={'fail':True};spec=write(tmp_path/'bad.json',base);bad=tmp_path/'failed'
    result=execute_asset_experiment(spec,bad)
    failed=json.loads((bad/'trials.json').read_text())[0]
    assert result['status']=='failed' and failed['stop_reason']=='algorithm_feedback_error'
    assert failed['observations']==1 and failed['failed_measurements']==0
    assert failed['recommendation_verified'] and failed['algorithm_contract_failures']==1


def test_execution_annotations_survive_real_cpu_measurement_and_storage(tmp_path):
    from kernel_tuning_lab.asset_runner import execute_asset_experiment
    plugin=tmp_path/'annotations.py'
    plugin.write_text('''from kernel_tuning_lab.sdk import KernelDefinition
from kernel_tuning_lab.catalog import get_catalog
BASE=get_catalog().resolve('kernel','python/gemm@1.0.0')
def build(ctx,**parameters):
 ctx.metadata['experiment_annotations']={'unit_evidence':{'scope':'real CPU execution, not GPU','layout_sha256':'fixed-fixture'}}
 return BASE.build(ctx,**parameters)
ASSETS=(KernelDefinition('unit/annotations','1.0.0',BASE.operator,build,BASE.space,runtime='python',dtypes=BASE.dtypes),)
''')
    spec=write(tmp_path/'spec.json',{'schema_version':2,'name':'annotations','operator':'gemm@1.0.0',
        'plugins':[str(plugin)],'kernels':['unit/annotations@1.0.0'],'algorithms':['grid@1.0.0'],'dtype':'float32',
        'shapes':[{'M':2,'N':3,'K':5}],'budget':{'max_evaluations':1,'max_wall_time_s':10},'warmup':0,'repetitions':3,
        'validation':{'max_configs':1,'max_cases':0,'max_wall_time_s':10,'trust_claims':False}})
    output=tmp_path/'out';result=execute_asset_experiment(spec,output);assert result['status']=='passed'
    from kernel_tuning_lab.storage import ExperimentStore
    with ExperimentStore(output/'measurements.sqlite',read_only=True) as store:
        rows=store.records()
    assert all(r.metadata['correctness']['execution']['annotations']['unit_evidence']['layout_sha256']=='fixed-fixture' for r in rows)
