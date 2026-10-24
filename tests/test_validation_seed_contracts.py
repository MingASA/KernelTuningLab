"""Deterministic checks for required validation input-seed coverage."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _plugin(path: Path, seeds):
    path.write_text(f'''from kernel_tuning_lab.sdk import KernelDefinition, OperatorDefinition, SearchAlgorithmDefinition, SearchCapabilities, ShapeSpec, SpaceDefinition, KernelConfig
def generate_inputs(ctx, seed):
    ctx.allocate_output("out", (1, 1), dtype="float32")
    return {{"x": [[1.0]]}}
def reference(ctx): return {{"out": [[1.0]]}}
operator = OperatorDefinition(id="seed_toy", version="1", shape_dims=("M", "N"), dtypes=("float32",),
    generate_inputs=generate_inputs, reference=reference, tolerances={{"float32": (0.0, 0.0)}},
    validation_shapes=(ShapeSpec({{"M": 1, "N": 1}}),), metadata={{"validation_seeds": {seeds!r}}})
def build(ctx):
    def launch(): ctx.outputs["out"][0][0] = 0.0 if ctx.seed == 0 else 1.0
    return launch
kernel = KernelDefinition(id="seed_toy/fixed", version="1", operator=operator.ref, build=build,
    space=SpaceDefinition({{}}), runtime="python", dtypes=("float32",), architectures=("cpu",))
class Searcher:
    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        if budget.exhausted(0, 0) or history: return []
        return [next(space.iter_configs())]
def make_searcher(seed=0): return Searcher()
algorithm = SearchAlgorithmDefinition("seed_first", "1", make_searcher, SearchCapabilities())
ASSETS = [operator, kernel, algorithm]
''', encoding="utf-8")


def _run(tmp_path, *, max_configs, seeds=(0,)):
    plugin = tmp_path / "seed_plugin.py"
    _plugin(plugin, list(seeds))
    spec = {
        "schema_version": 2, "name": "seed-validation", "operator": "seed_toy@1",
        "kernels": [{"asset": "seed_toy/fixed@1"}], "algorithms": [{"asset": "seed_first@1"}],
        "shapes": [{"M": 1, "N": 1}], "plugins": [str(plugin.resolve())], "dtype": "float32",
        "budget": {"max_evaluations": 1, "max_wall_time_s": 3.0},
        "validation": {"max_configs": max_configs, "max_cases": 1,
                       "max_wall_time_s": 5.0, "seed": 17, "trust_claims": False},
        "preflight_seconds": 5.0, "seed": 1, "warmup": 0, "repetitions": 1,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "output"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, "-m", "kernel_tuning_lab", "check", "--spec", str(spec_path), "--output", str(output)],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)
    return result, output


def test_fixed_kernel_validates_default_seed_and_required_zero_seed(tmp_path):
    result, output = _run(tmp_path, max_configs=2)
    report = json.loads((output / "validation_report.json").read_text())
    assert result.returncode != 0
    assert report["checks"]["attempted"] == 2
    assert report["checks"]["failed"] == 1
    context = report["contexts"][0]
    assert context["planned_input_seeds"] == [0, 17]
    pairs = context["attempted_config_seed_pairs"]
    assert [pair["seed"] for pair in pairs] == [17, 0]
    assert len({pair["config_id"] for pair in pairs}) == 1


def test_seed_zero_is_not_sampled_when_one_attempt_only_covers_default_seed(tmp_path):
    result, output = _run(tmp_path, max_configs=1)
    report = json.loads((output / "validation_report.json").read_text())
    assert result.returncode == 0
    assert report["checks"]["attempted"] == 1
    assert report["contexts"][0]["selected_seeds"] == [17]
    assert report["coverage"]["whole_space_verified"] is False


@pytest.mark.parametrize("seeds", [[True], ["0"]])
def test_invalid_declared_validation_seeds_fail_preflight(tmp_path, seeds):
    result, output = _run(tmp_path, max_configs=2, seeds=seeds)
    assert result.returncode != 0
    report = json.loads((output / "validation_report.json").read_text())
    assert report["status"] == "failed"
    assert "validation_seeds" in report["contexts"][0]["reason"]
