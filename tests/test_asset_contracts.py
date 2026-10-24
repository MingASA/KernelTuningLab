"""Public extension API contracts; no GPU SDK or private platform glue."""

from dataclasses import replace
import math
import time
from types import SimpleNamespace

import pytest

from kernel_tuning_lab.acceptance import ContractError, inspect_kernel_entrypoint, request_candidates
from kernel_tuning_lab.asset_space import SpaceView, build_effective_space, check_algorithm_compatibility
from kernel_tuning_lab.assets import (KernelDefinition, OperatorDefinition, ParameterSpec,
                                     SearchAlgorithmDefinition, SearchCapabilities, SpaceDefinition)
from kernel_tuning_lab.catalog import AssetCatalog
from kernel_tuning_lab.claims import claim_matches, declare_correctness
from kernel_tuning_lab.kernel_runtime import AssetKernelAdapter, ExecutionPlan, ExecutionStep
from kernel_tuning_lab.models import (HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec,
                                      RecordStatus, SearchBudget, ShapeSpec, WorkloadDistribution)
from kernel_tuning_lab.plugins import load_plugin
from kernel_tuning_lab.search import SearchHistory
from kernel_tuning_lab.space import ConfigSpace


def _operator(name="toy", *, source_files=()):
    def generate(ctx, seed):
        ctx.allocate_output("out", (ctx.shape.get("M"), ctx.shape.get("N")), dtype="float32")
        return {"x": [[1.0]]}

    def reference(ctx):
        return {"out": [[1.0]]}

    return OperatorDefinition(
        id=name, version="1", shape_dims=("M", "N"), dtypes=("float32",),
        generate_inputs=generate, reference=reference,
        tolerances={"float32": (0.0, 0.0)},
        validation_shapes=(ShapeSpec({"M": 1, "N": 1}),), source_files=source_files,
    )


def _build(ctx, tile=1):
    def launch():
        ctx.outputs["out"][0][0] = ctx.inputs["x"][0][0]
    return launch


def _kernel(operator=None, *, name="toy_kernel", version="1", space=None,
            parameter_map=None, build=_build, source_files=(), claims=()):
    operator = operator or _operator()
    return KernelDefinition(
        id=name, version=version, operator=operator.ref, build=build,
        space=space or SpaceDefinition({"tile": ParameterSpec((1, 2))}),
        runtime="python", dtypes=("float32",), architectures=("cpu",),
        parameter_map=parameter_map or {}, source_files=source_files, claims=claims,
    )


def _cpu_context(kernel, operator, *, shape=None, config=None):
    shape = shape or ShapeSpec({"M": 1, "N": 1})
    spec = OperatorSpec(name=operator.id, kind=operator.id, version=operator.version,
                        implementation=kernel.id, dtype="float32")
    hardware = HardwareSpec("CPU contract test", architecture="cpu", warp_size=1,
                            max_threads_per_block=1)
    config = config or KernelConfig({"tile": 1})
    adapter = AssetKernelAdapter(kernel, operator)
    return adapter, hardware, spec, shape, config


def test_catalog_is_idempotent_rejects_content_collision_and_requires_version_on_ambiguity():
    catalog = AssetCatalog()
    first = _operator("same")
    assert catalog.register(first) is first
    assert catalog.register(first) is first
    assert catalog.resolve("operators", first.ref) is first

    conflicting = replace(first, metadata={"revision": "changed"})
    with pytest.raises(ValueError, match="different content"):
        catalog.register(conflicting)
    assert catalog.resolve("operator", first.ref) is first

    newer = replace(first, version="2")
    catalog.register(newer)
    with pytest.raises(ValueError, match="ambiguous"):
        catalog.resolve("operator", "same")
    assert catalog.resolve("operator", "same@2") is newer


def test_plugin_assets_fingerprint_tracks_declared_source_and_conflicts_on_edit(tmp_path):
    plugin = tmp_path / "sample_plugin.py"

    def write_plugin(revision):
        plugin.write_text(f'''from kernel_tuning_lab.sdk import OperatorDefinition, ShapeSpec
def generate(ctx, seed): return {{"x": [[1.0]]}}
def reference(ctx): return {{"out": [[1.0]]}}
ASSETS = [OperatorDefinition(id="from_file", version="1", shape_dims=("M", "N"),
    dtypes=("float32",), generate_inputs=generate, reference=reference,
    tolerances={{"float32": (0.0, 0.0)}}, validation_shapes=(ShapeSpec({{"M": 1, "N": 1}}),),
    source_files=(__file__,), metadata={{"revision": "{revision}"}})]
''')

    catalog = AssetCatalog()
    write_plugin("a")
    loaded = load_plugin(str(plugin), catalog)
    original = catalog.resolve("operator", "from_file@1")
    original_fingerprint = original.fingerprint
    assert loaded.ASSETS[0] is original

    # A distinct catalog may load the same implementation as an idempotent asset.
    same_catalog = AssetCatalog()
    load_plugin(str(plugin), same_catalog)
    assert same_catalog.resolve("operator", "from_file@1").fingerprint == original_fingerprint

    time.sleep(0.002)
    write_plugin("b")
    with pytest.raises(ValueError, match="different content"):
        load_plugin(str(plugin), catalog)


def test_explicit_source_file_changes_asset_fingerprint(tmp_path):
    source = tmp_path / "helper.py"
    source.write_text("VALUE = 1\n")
    first = _operator(source_files=(str(source),))
    first_fingerprint = first.fingerprint
    source.write_text("VALUE = 22\n")
    second = _operator(source_files=(str(source),))
    assert first_fingerprint != second.fingerprint


def test_space_conditions_categories_and_restrictions_are_intersections_only():
    definition = SpaceDefinition({
        "mode": ParameterSpec(("fast", "safe"), kind="categorical"),
        "enabled": ParameterSpec((False, True), kind="boolean"),
        "tile": ParameterSpec((1, 2, 4), kind="integer", active_when={"enabled": True}, default=1),
    })
    kernel = _kernel(space=definition)
    hardware = HardwareSpec("CPU test", architecture="cpu", warp_size=1, max_threads_per_block=1)
    operator = OperatorSpec(name="toy", kind="toy", implementation=kernel.id, dtype="float32", version="1")
    shape = ShapeSpec({"M": 1, "N": 1})
    narrowed = build_effective_space(kernel, hardware, operator, shape,
                                     {"mode": ["safe"], "tile": [1, 4]})
    assert narrowed.domains == {"mode": ("safe",), "enabled": (False, True), "tile": (1, 4)}
    legal = narrowed.legal_configs(hardware, operator, shape)
    assert all(config.get("mode") == "safe" for config in legal)
    assert all(config.get("tile") == 1 for config in legal if config.get("enabled") is False)

    with pytest.raises(ValueError, match="widen"):
        build_effective_space(kernel, hardware, operator, shape, {"tile": [8]})
    with pytest.raises(ValueError, match="undeclared"):
        build_effective_space(kernel, hardware, operator, shape, {"unknown": [1]})
    with pytest.raises(ValueError, match="empty"):
        build_effective_space(kernel, hardware, operator, shape, {"tile": []})
    with pytest.raises(ValueError, match="duplicate"):
        build_effective_space(kernel, hardware, operator, shape, {"tile": [1, 1]})


def test_space_view_is_read_only_and_defensively_copies_configs():
    kernel = _kernel()
    hardware = HardwareSpec("CPU test", architecture="cpu", warp_size=1, max_threads_per_block=1)
    operator = OperatorSpec(name="toy", kind="toy", implementation=kernel.id, dtype="float32", version="1")
    shape = ShapeSpec({"M": 1, "N": 1})
    space = build_effective_space(kernel, hardware, operator, shape)
    view = SpaceView(space)
    with pytest.raises(TypeError):
        view.domains["tile"] = (99,)
    proposal = view.config_at(0)
    proposal.params["tile"] = 99
    assert space.config_at(0).get("tile") == 1
    with pytest.raises(AttributeError, match="read-only"):
        view.name = "changed"


def test_build_entrypoint_parameter_mapping_is_checked_before_execution():
    operator = _operator()
    def mapped_build(ctx, block_size=1):
        return lambda: None

    mapped = _kernel(operator, build=mapped_build, parameter_map={"tile": "block_size"})
    space = build_effective_space(
        mapped, HardwareSpec("CPU", architecture="cpu", warp_size=1, max_threads_per_block=1),
        OperatorSpec(name="toy", kind="toy", implementation=mapped.id, dtype="float32", version="1"),
        ShapeSpec({"M": 1, "N": 1}),
    )
    inspect_kernel_entrypoint(mapped, space)

    bad = _kernel(operator, build=lambda ctx: None, parameter_map={"tile": "unknown_kw"})
    bad_space = build_effective_space(
        bad, HardwareSpec("CPU", architecture="cpu", warp_size=1, max_threads_per_block=1),
        OperatorSpec(name="toy", kind="toy", implementation=bad.id, dtype="float32", version="1"),
        ShapeSpec({"M": 1, "N": 1}),
    )
    with pytest.raises(ContractError, match="signature/parameter mapping"):
        inspect_kernel_entrypoint(bad, bad_space)


class _Proposal:
    def __init__(self, proposals):
        self.proposals = proposals

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        return self.proposals


def _search_inputs():
    space = ConfigSpace({"tile": (1, 2)})
    hardware = HardwareSpec("CPU", architecture="cpu", warp_size=1, max_threads_per_block=1)
    operator = OperatorSpec(name="toy")
    shape = ShapeSpec({"M": 1, "N": 1})
    return space, hardware, operator, shape


@pytest.mark.parametrize("proposals,batch_size,history,budget,pattern", [
    ([KernelConfig({"tile": 1}), KernelConfig({"tile": 2})], 1, SearchHistory(), SearchBudget(2), "exceeded requested batch_size"),
    ([KernelConfig({"tile": 99})], 1, SearchHistory(), SearchBudget(2), "illegal configuration"),
    ([KernelConfig({"tile": 1})], 1, SearchHistory(attempted_config_ids=frozenset({KernelConfig({"tile": 1}).id})), SearchBudget(2), "repeated"),
    ([KernelConfig({"tile": 1}), KernelConfig({"tile": 1})], 2, SearchHistory(), SearchBudget(2), "repeated"),
    ([KernelConfig({"tile": 1})], 1, SearchHistory(), SearchBudget(0), "exhausted budget"),
])
def test_algorithm_proposals_are_bounded_legal_unique_and_budgeted(proposals, batch_size, history, budget, pattern):
    space, hardware, operator, shape = _search_inputs()
    with pytest.raises(ContractError, match=pattern):
        request_candidates(_Proposal(proposals), hardware, operator, shape, space, history, budget, batch_size)


def test_algorithm_declared_capabilities_reject_unsupported_parameter_kinds():
    kernel = _kernel(space=SpaceDefinition({"mode": ParameterSpec(("a", "b"), kind="categorical")}))
    hardware = HardwareSpec("CPU", architecture="cpu", warp_size=1, max_threads_per_block=1)
    operator = OperatorSpec(name="toy", kind="toy", implementation=kernel.id, dtype="float32", version="1")
    shape = ShapeSpec({"M": 1, "N": 1})
    space = build_effective_space(kernel, hardware, operator, shape)
    definition = SearchAlgorithmDefinition(
        "integers", "1", lambda seed=0: _Proposal([]),
        SearchCapabilities(parameter_kinds=("integer",), supports_constraints=True),
    )
    result = check_algorithm_compatibility(definition, space)
    assert not result.valid
    assert any("categorical" in reason for reason in result.reasons)

    conditional_space = SpaceDefinition({
        "enabled": ParameterSpec((False, True), kind="boolean"),
        "tile": ParameterSpec((1, 2), kind="integer", active_when={"enabled": True}, default=1),
    })
    conditional_kernel = _kernel(space=conditional_space)
    conditional_operator = OperatorSpec(name="toy", kind="toy", implementation=conditional_kernel.id,
                                        dtype="float32", version="1")
    conditional = build_effective_space(conditional_kernel, hardware, conditional_operator, shape)
    conditional_algorithm = SearchAlgorithmDefinition(
        "flat-only", "1", lambda seed=0: _Proposal([]),
        SearchCapabilities(supports_conditional=False),
    )
    conditional_result = check_algorithm_compatibility(conditional_algorithm, conditional)
    assert not conditional_result.valid
    assert any("conditional" in reason for reason in conditional_result.reasons)


def test_correctness_claim_is_content_bound_and_checks_range_and_version():
    operator = _operator()
    kernel = _kernel(operator)
    context = SimpleNamespace(
        runtime="python", hardware=HardwareSpec("CPU", architecture="cpu"),
        operator=OperatorSpec(name="toy", kind="toy", implementation=kernel.id, dtype="float32", version="1"),
        shape=ShapeSpec({"M": 1, "N": 1}), seed=7,
    )
    config = KernelConfig({"tile": 1})
    scope = {"runtimes": ["python"], "dtypes": ["float32"], "architectures": ["cpu"],
             "shapes": {"M": {"min": 1, "max": 4}, "N": [1]},
             "configs": {"tile": 1}, "parameters": {}, "seeds": {"min": 0, "max": 10}}
    claim = declare_correctness(kernel, operator, id="claim", source="submitter",
                                 evidence=["test evidence"], scope=scope)
    assert claim_matches(claim, kernel, operator, context, config).valid
    assert not claim_matches(claim, kernel, operator, context, KernelConfig({"tile": 2})).valid
    stale = replace(claim, kernel_fingerprint="stale-content-hash")
    mismatch = claim_matches(stale, kernel, operator, context, config)
    assert not mismatch.valid
    assert any("kernel_fingerprint" in reason for reason in mismatch.reasons)
    wrong_version = replace(claim, operator_ref="toy@2")
    version_mismatch = claim_matches(wrong_version, kernel, operator, context, config)
    assert not version_mismatch.valid
    assert any("operator_ref" in reason for reason in version_mismatch.reasons)


@pytest.mark.parametrize("corruption", ["wrong_shape", "nan"])
def test_runtime_rejects_wrong_output_shape_and_nonfinite_values(corruption):
    operator = _operator()

    def wrong_shape(ctx, tile=1):
        return lambda: {"out": [[1.0, 2.0]]}

    def nan_output(ctx, tile=1):
        return lambda: {"out": [[math.nan]]}

    kernel = _kernel(operator, build=wrong_shape if corruption == "wrong_shape" else nan_output)
    adapter, hardware, operator_spec, shape, config = _cpu_context(kernel, operator)
    prepared = adapter.prepare(hardware, operator_spec, shape, seed=0)
    compiled = adapter.compile(prepared, config)
    correct, details = adapter.check_correctness(compiled)
    assert not correct
    assert details["status"] == "failed"
    if corruption == "nan":
        assert details["outputs"]["out"]["nonfinite_output"] is True


def test_multistep_execution_reinitializes_state_for_every_correctness_invocation():
    operator = _operator()
    calls = {"initialize": 0, "setup": 0}

    def build(ctx, tile=1):
        accumulator = ctx.allocate_workspace("acc", (1, 1), dtype="float32", fill=0.0)

        def initialize():
            calls["initialize"] += 1
            accumulator[0][0] = 0.0

        def accumulate():
            accumulator[0][0] += ctx.inputs["x"][0][0]

        def copy_out():
            ctx.outputs["out"][0][0] = accumulator[0][0]

        def setup():
            calls["setup"] += 1

        return ExecutionPlan((ExecutionStep("accumulate", accumulate), ExecutionStep("write", copy_out)),
                             initialize=initialize, setup=setup)

    kernel = _kernel(operator, build=build)
    adapter, hardware, operator_spec, shape, config = _cpu_context(kernel, operator)
    compiled = adapter.compile(adapter.prepare(hardware, operator_spec, shape, 0), config)
    correct, details = adapter.check_correctness(compiled)
    assert correct
    assert calls == {"initialize": 2, "setup": 1}
    assert details["execution"]["steps"] == ["accumulate", "write"]
    assert details["execution"]["initialize_each_invocation"] is True


def test_claimed_correctness_remains_declared_and_never_platform_verified():
    from kernel_tuning_lab.operator_library import register_builtin_assets

    catalog = AssetCatalog()
    register_builtin_assets(catalog)
    operator = catalog.resolve("operator", "gemm@1.0.0")
    kernel = catalog.resolve("kernel", "python/gemm@1.0.0")
    hardware = HardwareSpec("CPU", architecture="cpu", warp_size=1, max_threads_per_block=1)
    operator_spec = OperatorSpec(name="gemm", kind="gemm", implementation=kernel.id,
                                 dtype="float32", version="1.0.0")
    shape = ShapeSpec({"M": 1, "N": 1, "K": 1})
    config = KernelConfig({"tile_size": 1})
    scope = {"runtimes": ["python"], "dtypes": ["float32"], "architectures": ["cpu"],
             "shapes": dict(shape.dims), "configs": dict(config.params), "parameters": {}}
    claim = declare_correctness(kernel, operator, id="claim", source="submitter", evidence=["external"], scope=scope)
    claimed_kernel = replace(kernel, claims=(claim,))
    adapter = AssetKernelAdapter(claimed_kernel, operator)
    prepared = adapter.prepare(hardware, operator_spec, shape, 0)
    compiled = adapter.compile(prepared, config)
    correct, details = adapter.check_correctness(compiled)
    assert correct
    assert details["status"] == "declared"
    assert details["correctness_origin"] == "submitter_claim"
    assert details["comparison_performed"] is False
    assert details["verified_on_cpu"] is False


def test_submitter_claim_measurements_are_excluded_from_replay_and_l3_oracles():
    from kernel_tuning_lab.l3.matrix import PerformanceMatrix
    from kernel_tuning_lab.replay import run_replay

    hardware = HardwareSpec("CPU", architecture="cpu", warp_size=1, max_threads_per_block=1)
    operator = OperatorSpec(name="toy", kind="toy", implementation="toy_kernel", dtype="float32")
    shape = ShapeSpec({"M": 1, "N": 1})
    config = KernelConfig({"tile": 1})
    declared = MeasurementRecord(
        hardware, operator, shape, config, RecordStatus.SUCCESS,
        runtime_samples_ms=(0.25,), provenance="measured", split="train",
        metadata={"correctness": {"status": "declared", "correctness_origin": "submitter_claim",
                                  "comparison_performed": False, "verified_on_cpu": False}},
    )
    workload = WorkloadDistribution((shape,), (1.0,), name="claim-oracle-test")
    matrix = PerformanceMatrix.from_records(workload, (config,), (declared,))
    assert matrix.latencies_ms == ((None,),)
    assert matrix.metadata["excluded_unverified_records"] == 1

    class FirstConfig:
        def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
            if budget.exhausted(0, 0) or any(record.config.id == config.id for record in history):
                return []
            return [config]

    replay = run_replay((declared,), FirstConfig(), ConfigSpace({"tile": (1,)}), SearchBudget(1))
    assert replay.oracle_best_latency_ms is None
    assert replay.best_latency_ms is None
    assert replay.trace[0].status == "missing"

