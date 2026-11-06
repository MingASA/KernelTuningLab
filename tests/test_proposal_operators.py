"""Cross-family contracts for proposal assets and bounded contextual spaces."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kernel_tuning_lab.assets import SpaceContext, SpaceDefinition, ParameterSpec, KernelDefinition
from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.catalog import get_catalog
from kernel_tuning_lab.models import HardwareSpec, OperatorSpec, ShapeSpec, KernelConfig
from kernel_tuning_lab.kernel_runtime import ExecutionContext


ROOT = Path(__file__).resolve().parents[1]


def _scope_plugin(path, factory_body='return ({"M": 1},)'):
    path.write_text(f'''from kernel_tuning_lab.sdk import (KernelDefinition, OperatorDefinition, ParameterSpec,
    ShapeSpec, SpaceDefinition)
def generate(ctx, seed):
    m = ctx.shape.get("M")
    ctx.allocate_output("out", (m, 1), dtype="float32")
    return {{}}
def reference(ctx):
    return {{"out": [[1.0] for _ in range(ctx.shape.get("M"))]}}
def factory(parameters, requested):
    {factory_body}
def build(ctx, tile=1):
    def launch():
        for i in range(ctx.shape.get("M")): ctx.outputs["out"][i][0] = 1.0
    return launch
def resolve(ctx): return {{"tile": (2,) if ctx.shape.get("M") > 1 else (1,)}}
op = OperatorDefinition("scope_toy", "1.0.0", ("M",), ("float32",), generate,
    reference, {{"float32": (0.0, 0.0)}}, validation_case_factory=factory)
kernel = KernelDefinition("python/scope_toy", "1.0.0", op.ref, build,
    SpaceDefinition({{"tile": ParameterSpec((1, 2))}}, resolver=resolve), runtime="python",
    dtypes=("float32",), architectures=("cpu",))
ASSETS = [op, kernel]
''', encoding="utf-8")


def _run_scope_check(tmp_path, factory_body='return ({"M": 1},)'):
    plugin = tmp_path / "scope_plugin.py"
    _scope_plugin(plugin, factory_body)
    spec = {
        "schema_version": 2, "name": "scope-check", "operator": "scope_toy@1.0.0",
        "kernels": [{"asset": "python/scope_toy@1.0.0", "space": {"tile": [2]}}],
        "algorithms": [{"asset": "random@1.0.0"}], "shapes": [{"M": 2}],
        "plugins": [str(plugin)], "dtype": "float32", "budget": {"max_evaluations": 0, "max_wall_time_s": 3},
        "validation": {"max_configs": 2, "max_cases": 2, "max_wall_time_s": 5, "trust_claims": False},
        "preflight_seconds": 5, "warmup": 0, "repetitions": 1,
    }
    spec_path = tmp_path / "scope.json"
    spec_path.write_text(json.dumps(spec))
    output = tmp_path / "out"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run([sys.executable, "-m", "kernel_tuning_lab", "check", "--spec", str(spec_path),
                             "--output", str(output)], cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=20)
    return result, output


def test_catalog_discovers_eight_proposal_operator_families_and_runtimes():
    catalog = get_catalog()
    expected = {
        "dequantize_w4", "dequantize_w3", "dequantize_w2",
        "w4a16_gemm", "w3a16_gemm", "w2a16_gemm",
        "attention", "moe_grouped_gemm",
    }
    operators = {item.id for item in catalog.list("operator")}
    assert expected <= operators
    for stem in expected:
        assert catalog.resolve("operator", f"{stem}@1.0.0")
        assert catalog.resolve("kernel", f"python/{stem}@1.0.0")
        assert catalog.resolve("kernel", f"triton/{stem}@1.0.0")


def test_space_context_is_immutable_and_resolver_only_prunes_domains():
    hw = HardwareSpec("test", architecture="sm_90", warp_size=32, max_threads_per_block=1024)
    op = OperatorSpec("toy", kind="toy", implementation="toy/python@1.0.0", dtype="float32")
    shape = ShapeSpec({"M": 4})
    context = SpaceContext(hw, op, shape, environment={"nested": {"x": [1, 2]}},
                           workspace_memory_limit_bytes=4096)
    with pytest.raises(TypeError):
        context.environment["x"] = 1
    with pytest.raises(TypeError):
        context.environment["nested"]["x"] = 3

    definition = SpaceDefinition({"flag": ParameterSpec((False, True))},
                                 resolver=lambda ctx: {"flag": (False,)})
    kernel = KernelDefinition("toy/python", "1.0.0", "toy@1.0.0", lambda ctx: lambda: None,
                              space=definition, runtime="python", dtypes=("float32",), architectures=("cpu",))
    space = build_effective_space(kernel, hw, op, shape, context=context)
    assert space.cardinality == 1
    assert tuple(space.resolution["context_domains"]["flag"]) == (False,)
    assert tuple(space.resolution["pruned_by_context"]["flag"]) == (True,)
    with pytest.raises(ValueError):
        build_effective_space(kernel, hw, op, shape, {"flag": [True]}, context=context)


def test_context_identity_mismatch_and_invalid_resolver_results_are_rejected():
    hw = HardwareSpec("test", architecture="cpu", warp_size=1, max_threads_per_block=1)
    op = OperatorSpec("toy", kind="toy", implementation="toy/python@1.0.0", dtype="float32")
    shape = ShapeSpec({"M": 1})
    other = ShapeSpec({"M": 2})
    kernel = KernelDefinition("toy/python", "1.0.0", "toy@1.0.0", lambda ctx: lambda: None,
        space=SpaceDefinition({"x": ParameterSpec((1, 2))}, resolver=lambda ctx: {"unknown": (1,)}),
        runtime="python", dtypes=("float32",), architectures=("cpu",))
    with pytest.raises(ValueError, match="identities must match"):
        build_effective_space(kernel, hw, op, shape, context=SpaceContext(hw, op, other))
    with pytest.raises(ValueError, match="undeclared parameters"):
        build_effective_space(kernel, hw, op, shape)


def test_context_resolver_internal_typeerror_is_not_retried_as_legacy_signature():
    hw = HardwareSpec("test", architecture="cpu", warp_size=1, max_threads_per_block=1)
    op = OperatorSpec("toy", kind="toy", implementation="toy/python@1.0.0", dtype="float32")
    shape = ShapeSpec({"M": 1})
    calls = []
    def resolver(ctx):
        calls.append(ctx)
        raise TypeError("resolver body failed")
    kernel = KernelDefinition("toy/python", "1.0.0", "toy@1.0.0", lambda ctx: lambda: None,
        space=SpaceDefinition({"x": ParameterSpec((1, 2))}, resolver=resolver),
        runtime="python", dtypes=("float32",), architectures=("cpu",))
    with pytest.raises(TypeError, match="resolver body failed"):
        build_effective_space(kernel, hw, op, shape)
    assert len(calls) == 1


def test_context_constraint_internal_typeerror_is_not_retried_and_legacy_apis_remain_covered():
    hw = HardwareSpec("test", architecture="cpu", warp_size=1, max_threads_per_block=1)
    op = OperatorSpec("toy", kind="toy", implementation="toy/python@1.0.0", dtype="float32")
    shape = ShapeSpec({"M": 1})
    calls = []
    def constraint(config, ctx):
        calls.append(ctx)
        raise TypeError("constraint body failed")
    kernel = KernelDefinition("toy/python", "1.0.0", "toy@1.0.0", lambda ctx: lambda: None,
        space=SpaceDefinition({"x": ParameterSpec((1, 2))}, constraints=(constraint,)),
        runtime="python", dtypes=("float32",), architectures=("cpu",))
    space = build_effective_space(kernel, hw, op, shape)
    result = space.validate(KernelConfig({"x": 1}), hw, op, shape)
    assert not result.valid
    assert any("constraint body failed" in reason for reason in result.reasons)
    assert len(calls) == 1


def test_compiled_resource_metadata_records_valid_values_and_rejects_shared_limit():
    hw = HardwareSpec("resource test", architecture="sm_80", warp_size=32,
                      max_threads_per_block=1024, smem_per_block=48 * 1024)
    op = OperatorSpec("toy", kind="toy", implementation="toy@1", dtype="float32")
    shape = ShapeSpec({"M": 1})
    context = ExecutionContext(hw, op, shape, "python")
    metadata = type("Metadata", (), {"shared": 1024, "num_warps": 4})()
    compiled = type("Compiled", (), {"metadata": metadata, "n_regs": 32})()
    context.record_compiled_kernel("fake", compiled)
    assert context.metadata["execution_resources"]["fake"] == {
        "shared_memory_bytes": 1024, "registers_per_thread": 32, "num_warps": 4,
        "compiler_cache_identity": None, "artifacts": {}}
    too_large = type("Compiled", (), {"metadata": type("Metadata", (), {"shared": 50_000, "num_warps": 4})(),
                                      "n_regs": 32})()
    with pytest.raises(RuntimeError, match="exceeds"):
        context.record_compiled_kernel("oversized", too_large)


def test_context_workspace_budget_prunes_quant_pipeline_and_moe_splitk():
    catalog = get_catalog()
    hw = HardwareSpec("fake sm80", architecture="sm_80", warp_size=32,
                      max_threads_per_block=1024, smem_per_block=100_000,
                      memory_bytes=80 * 1024**3)
    shape = ShapeSpec({"M": 128, "N": 128, "K": 512})
    op = OperatorSpec("w4a16_gemm", kind="w4a16_gemm", implementation="triton_pipeline/w4a16_gemm@1.0.0",
                      dtype="float16", parameters={"group_size": 32})
    pipeline = catalog.resolve("kernel", "triton_pipeline/w4a16_gemm@1.0.0")
    def domain_for(budget):
        ctx = SpaceContext(hw, op, shape, workspace_memory_limit_bytes=budget)
        return build_effective_space(pipeline, hw, op, shape, context=ctx)
    small = domain_for(150_000)
    large = domain_for(2_000_000)
    assert max(small.domains["SPLIT_K"]) == 1
    assert max(large.domains["SPLIT_K"]) > 1
    split_config = dict(small.config_at(0).params)
    split_config["SPLIT_K"] = 2
    assert not small.validate(KernelConfig(split_config), hw, op, shape).valid

    moe_shape = ShapeSpec({"M": 128, "N": 128, "K": 512, "Groups": 4})
    moe_op = OperatorSpec("moe_grouped_gemm", kind="moe_grouped_gemm",
                          implementation="triton/moe_grouped_gemm@1.0.0", dtype="float16")
    moe_kernel = catalog.resolve("kernel", "triton/moe_grouped_gemm@1.0.0")
    def moe_space(budget):
        ctx = SpaceContext(hw, moe_op, moe_shape, workspace_memory_limit_bytes=budget)
        return build_effective_space(moe_kernel, hw, moe_op, moe_shape, context=ctx)
    small_moe, large_moe = moe_space(20_000), moe_space(2_000_000)
    assert max(small_moe.domains["SPLIT_K"]) == 1
    assert max(large_moe.domains["SPLIT_K"]) > 1
    split_config = dict(small_moe.config_at(0).params)
    split_config["SPLIT_K"] = 2
    assert not small_moe.validate(KernelConfig(split_config), hw, moe_op, moe_shape).valid


def test_attention_large_batch_flattened_grid_stays_in_x_without_cpu_allocation():
    from kernel_tuning_lab.builtin_assets import attention

    hw = HardwareSpec("fake sm80", architecture="sm_80", warp_size=32,
                      max_threads_per_block=1024, smem_per_block=164 * 1024,
                      registers_per_sm=65536, memory_bytes=80 * 1024**3)
    op = OperatorSpec("attention", kind="attention", implementation="triton/attention@1.0.0",
                      dtype="float16", parameters={"causal": False})
    shape = ShapeSpec({"Batch": 65536, "Heads": 1, "KVHeads": 1,
                       "SeqQ": 1, "SeqKV": 1, "D": 1})
    assert attention._cuda_supports(hw, op, shape) is None
    config = KernelConfig({"BLOCK_M": 16, "BLOCK_N": 16, "num_warps": 4,
                           "num_stages": 1, "USE_EXP2": True, "PRE_LOAD_V": False})
    assert attention._triton_constraint(config, hw, op, shape) is None
    source = (__import__("pathlib").Path(__file__).parents[1] /
              "kernel_tuning_lab/backends/_triton_attention.py").read_text()
    assert "program = tl.program_id(0)" in source
    assert "batch_head = program // query_tiles" in source
    assert "grid = (triton.cdiv(sq, BLOCK_M) * b * hq,)" in __import__("inspect").getsource(attention._build_triton)
    # The kernel's flattened program decode assigns every (batch, head, query tile) once.
    batch, heads, query_tiles = 2, 3, 2
    decoded = []
    for program in range(batch * heads * query_tiles):
        tile = program % query_tiles
        batch_head = program // query_tiles
        decoded.append((batch_head // heads, batch_head % heads, tile))
    assert decoded == [(b, h, q) for b in range(batch) for h in range(heads) for q in range(query_tiles)]


def test_moe_large_weight_product_and_rank_three_tma_descriptor_are_static_safe():
    from kernel_tuning_lab.builtin_assets import moe
    from pathlib import Path

    shape = ShapeSpec({"M": 1, "N": 50_000, "K": 50_000, "Groups": 1})
    op = OperatorSpec("moe_grouped_gemm", kind="moe_grouped_gemm",
                      implementation="triton/moe_grouped_gemm@1.0.0", dtype="float16")
    ctx = type("Context", (), {"shape": shape, "operator": op, "runtime": "cuda",
                              "hardware": HardwareSpec("fake H100", architecture="sm_90",
                                  warp_size=32, max_threads_per_block=1024,
                                  memory_bytes=80 * 1024**3)})()
    assert moe._validate(ctx) is None
    source = (Path(__file__).parents[1] / "kernel_tuning_lab/backends/_triton_moe.py").read_text()
    assert "to(tl.int64)" in source
    asset_source = Path(moe.__file__).read_text()
    assert "TensorDescriptor(b, [groups, k, n]" in asset_source


def test_validation_factory_extra_uses_kernel_defaults_and_records_space_scopes(tmp_path):
    result, output = _run_scope_check(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    lock = json.loads((output / "assets.lock.json").read_text())
    assert lock["validation_cases"]["extra"] == [{"M": 1}]
    assert lock["requested_space_scope"] == "experiment_restrictions"
    assert lock["extra_space_scope"] == "kernel_defaults"
    spaces = lock["effective_spaces"]
    requested = next(item for item in spaces if item["shape"] == {"M": 2})
    extra = next(item for item in spaces if item["shape"] == {"M": 1})
    assert requested["space"]["resolution"]["space_scope"] == "experiment_restrictions"
    assert requested["space"]["domains"]["tile"] == [2]
    assert extra["space"]["resolution"]["space_scope"] == "kernel_defaults"
    assert extra["space"]["domains"]["tile"] == [1]


def test_validation_factory_rejects_duplicate_shapes_before_case_limit(tmp_path):
    result, output = _run_scope_check(tmp_path, 'return ({"M": 1}, {"M": 1})')
    assert result.returncode == 2, result.stdout + result.stderr
    summary = json.loads((output / "run_summary.json").read_text())
    assert summary["status"] == "failed"
    assert "duplicate case dimensions" in json.dumps(summary).lower()


def test_proposal_suite_write_only_generates_production_specs_without_measurements(tmp_path):
    output = tmp_path / "production"
    result = subprocess.run([sys.executable, "-m", "kernel_tuning_lab", "proposal-suite",
                             "--runtime", "cuda", "--scale", "production", "--write-only",
                             "--hardware", "5060ti16g", "--output", str(output)],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "suite_manifest.json").read_text())
    assert len(manifest["operators"]) == 8
    assert len(manifest["cases"]) > 8
    assert all(case["status"] == "generated" for case in manifest["cases"])
    assert not list(output.rglob("measurements.jsonl"))
    assert not list(output.rglob("measurements.sqlite"))


def test_proposal_suite_cli_propagates_cuda_deferred_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = subprocess.run([sys.executable, "-m", "kernel_tuning_lab", "proposal-suite",
                             "--runtime", "cuda", "--scale", "smoke", "--max-evaluations", "1",
                             "--operators", "attention@1.0.0", "--output", str(tmp_path / "cuda")],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 3, result.stdout + result.stderr
    summary = json.loads((tmp_path / "cuda" / "suite_summary.json").read_text())
    assert summary["status"] == "deferred"

