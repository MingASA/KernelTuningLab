"""CPU-only hardening checks for post-measurement asset integrity."""

import importlib.util
from pathlib import Path

import pytest

from kernel_tuning_lab.models import KernelConfig, RecordStatus
from kernel_tuning_lab.runner import BenchmarkRunner
from kernel_tuning_lab.space import ConfigSpace


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("asset_contract_fixture", ROOT / "tests" / "test_asset_contracts.py")
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def _evaluate(build, tmp_path, *, repetitions=1):
    operator = _fixture._operator()
    kernel = _fixture._kernel(operator, build=build)
    adapter, hardware, operator_spec, shape, config = _fixture._cpu_context(kernel, operator)
    hardware = adapter.probe_hardware()
    runner = BenchmarkRunner(adapter, warmup=0, repetitions=repetitions, experiment_id="post-measurement-hardening")
    record = runner.measure(hardware, operator_spec, shape, config, ConfigSpace({"tile": (1,)}), resume=False)
    return record, adapter, hardware, operator_spec, shape, config


def test_measurement_output_is_rechecked_after_timing_and_samples_are_discarded(tmp_path):
    def build(ctx, tile=1):
        def launch():
            count = ctx.metadata.get("launch_count", 0) + 1
            ctx.metadata["launch_count"] = count
            ctx.outputs["out"][0][0] = 1.0 if count <= 2 else 0.0
        return launch

    record, *_ = _evaluate(build, tmp_path)
    assert record.status == RecordStatus.CORRECTNESS_ERROR
    assert record.runtime_samples_ms == ()
    assert record.metadata["correctness_verified"] is False
    assert record.metadata["correctness"]["oracle_eligible"] is False
    assert record.metadata["correctness"]["post_measurement"]["status"] == "failed"
    assert record.metadata["correctness"]["post_measurement"]["additional_invocations"] == 0


def test_noop_timed_invocations_cannot_reuse_stale_correct_output(tmp_path):
    def build(ctx, tile=1):
        def launch():
            count = ctx.metadata.get("launch_count", 0) + 1
            ctx.metadata["launch_count"] = count
            if count <= 2:
                ctx.outputs["out"][0][0] = 1.0
            # Later launches intentionally leave the preallocated output alone.
        return launch

    record, *_ = _evaluate(build, tmp_path, repetitions=2)
    assert record.status == RecordStatus.CORRECTNESS_ERROR
    assert record.runtime_samples_ms == ()
    assert record.metadata["correctness"]["oracle_eligible"] is False
    assert record.metadata["correctness"]["post_measurement"]["status"] == "failed"


def test_input_mutation_fails_even_when_kernel_writes_expected_output(tmp_path):
    def build(ctx, tile=1):
        def launch():
            ctx.inputs["x"][0][0] = 9.0
            ctx.outputs["out"][0][0] = 1.0
        return launch

    record, *_ = _evaluate(build, tmp_path)
    assert record.status == RecordStatus.CORRECTNESS_ERROR
    assert record.metadata["correctness"]["correctness_origin"] == "platform_input_integrity"
    assert record.metadata["correctness"]["input_integrity"]["changed_inputs"]


def test_output_aliasing_input_is_rejected(tmp_path):
    def build(ctx, tile=1):
        def launch(): ctx.outputs["out"] = ctx.inputs["x"]
        return launch

    record, *_ = _evaluate(build, tmp_path)
    assert record.status == RecordStatus.CORRECTNESS_ERROR
    assert record.metadata["correctness"]["correctness_origin"] == "platform_input_integrity"
    assert record.metadata["correctness"]["outputs"]["__inputs__"]["aliased_outputs"]


def test_native_artifact_identity_is_resolved_before_cache_lookup(tmp_path):
    from dataclasses import replace
    from kernel_tuning_lab.production.environment import file_identity
    from kernel_tuning_lab.storage import ExperimentStore
    artifact=tmp_path/'native-component.bin';artifact.write_bytes(b'version-one')
    operator=_fixture._operator()
    kernel=replace(_fixture._kernel(operator),execution_identity=lambda:{'binary_sha256':file_identity(artifact)})
    fingerprint=kernel.fingerprint
    adapter,hardware,op,shape,config=_fixture._cpu_context(kernel,operator)
    hardware=adapter.probe_hardware();space=ConfigSpace({'tile':(1,)})
    with ExperimentStore(tmp_path/'cache.sqlite') as store:
        runner=BenchmarkRunner(adapter,store=store,warmup=0,repetitions=1)
        first=runner.measure(hardware,op,shape,config,space)
        assert first.correctness_verified and not runner.last_cache_hit
        assert runner.measure(hardware,op,shape,config,space).id==first.id and runner.last_cache_hit
        artifact.write_bytes(b'version-two')
        assert kernel.fingerprint==fingerprint # plugin declaration unchanged
        second=runner.measure(hardware,op,shape,config,space)
        assert second.correctness_verified and not runner.last_cache_hit
        assert first.cache_key!=second.cache_key
        assert first.environment['native_execution_identity']!=second.environment['native_execution_identity']


def test_native_unsupported_packing_is_not_fabricated_as_compilation_or_latency(tmp_path):
    from kernel_tuning_lab.sdk import UnsupportedPreparationError
    def build(ctx,tile=1):
        raise UnsupportedPreparationError('native selection has no logical packer',details={'native_template_id':0,'TileP':64})
    record,*_=_evaluate(build,tmp_path)
    assert record.status==RecordStatus.INVALID_CONFIG and not record.runtime_samples_ms
    assert record.metadata['failure_class']=='unsupported_preparation'
    assert record.metadata['native_preparation_details']['TileP']==64


def test_config_dependent_packed_input_mutation_is_checked_without_hiding_task_input_changes(tmp_path):
    def build(ctx,tile=1):
        packed=ctx.register_prepared_input('backend_packed',[1.0])
        def launch():
            packed[0]=2.0
            ctx.outputs['out'][0][0]=1.0
        return launch
    record,*_=_evaluate(build,tmp_path)
    assert record.status==RecordStatus.CORRECTNESS_ERROR
    assert 'prepared::backend_packed' in record.metadata['correctness']['input_integrity']['changed_inputs']


@pytest.mark.parametrize("value", [True, "not numeric"])
def test_malformed_python_outputs_are_persisted_as_correctness_failures(tmp_path, value):
    def build(ctx, tile=1):
        def launch(): ctx.outputs["out"] = [[value]]
        return launch

    record, *_ = _evaluate(build, tmp_path)
    assert record.status == RecordStatus.CORRECTNESS_ERROR
    assert record.failure_reason
    assert record.metadata["correctness_verified"] is False
    assert record.metadata["correctness"]["status"] == "failed"


def test_counterexample_revokes_scoped_claim_for_later_measurement(tmp_path):
    from dataclasses import replace

    operator = _fixture._operator()
    kernel = _fixture._kernel(operator)
    # The public helper creates a shape/config scoped assertion.
    hardware = _fixture.HardwareSpec("CPU", architecture="cpu", warp_size=1, max_threads_per_block=1)
    operator_spec = _fixture.OperatorSpec(name="toy", kind="toy", implementation=kernel.id, dtype="float32", version="1")
    shape = _fixture.ShapeSpec({"M": 1, "N": 1})
    config = KernelConfig({"tile": 1})
    def seed_dependent(ctx, tile=1):
        def launch(): ctx.outputs["out"][0][0] = 0.0 if ctx.seed == 0 else 1.0
        return launch
    kernel = replace(kernel, build=seed_dependent)
    claim = _fixture.declare_correctness(kernel, operator, id="lifecycle-claim", source="fixture",
        evidence=["CPU fixture"], scope={"runtimes": ["python"], "dtypes": ["float32"], "architectures": ["cpu"],
        "shapes": dict(shape.dims), "configs": dict(config.params), "parameters": {}})
    claimed_kernel = replace(kernel, claims=(claim,))
    first_adapter = _fixture.AssetKernelAdapter(claimed_kernel, operator, accept_claims=True, force_correctness=True)
    hardware = first_adapter.probe_hardware()
    runner = BenchmarkRunner(first_adapter, warmup=0, repetitions=1, seed=0, experiment_id="counterexample")
    failed = runner.measure(hardware, operator_spec, shape, config, ConfigSpace({"tile": (1,)}), resume=False)
    assert failed.status == RecordStatus.CORRECTNESS_ERROR
    assert failed.metadata["correctness"]["correctness_origin"] == "platform_reference"

    second_adapter = _fixture.AssetKernelAdapter(claimed_kernel, operator, accept_claims=True)
    assert second_adapter.remember_counterexample(failed) == ["lifecycle-claim"]
    prepared = second_adapter.prepare(hardware, operator_spec, shape, seed=1)
    compiled = second_adapter.compile(prepared, config)
    correct, details = second_adapter.check_correctness(compiled)
    assert correct
    assert details["status"] == "verified"


@pytest.mark.parametrize("bad_at", [2, 3])
def test_declared_correctness_rechecks_finite_structure_on_each_required_boundary(tmp_path, bad_at):
    from dataclasses import replace

    operator = _fixture._operator()
    def build(ctx, tile=1):
        def launch():
            count = ctx.metadata.get("declared_calls", 0) + 1
            ctx.metadata["declared_calls"] = count
            if count == bad_at:
                ctx.outputs["out"] = [[float("nan")]] if bad_at == 2 else [[1.0, 2.0]]
            else:
                ctx.outputs["out"] = [[1.0]]
        return launch
    kernel = _fixture._kernel(operator, build=build)
    hardware = _fixture.HardwareSpec("CPU", architecture="cpu", warp_size=1, max_threads_per_block=1)
    operator_spec = _fixture.OperatorSpec(name="toy", kind="toy", implementation=kernel.id, dtype="float32", version="1")
    shape, config = _fixture.ShapeSpec({"M": 1, "N": 1}), KernelConfig({"tile": 1})
    claim = _fixture.declare_correctness(kernel, operator, id=f"declared-struct-{bad_at}", source="fixture",
        evidence=["fixture claim"], scope={"runtimes": ["python"], "dtypes": ["float32"], "architectures": ["cpu"],
        "shapes": dict(shape.dims), "configs": dict(config.params), "parameters": {}})
    kernel = replace(kernel, claims=(claim,))
    adapter = _fixture.AssetKernelAdapter(kernel, operator, accept_claims=True)
    hardware = adapter.probe_hardware()
    runner = BenchmarkRunner(adapter, warmup=0, repetitions=1, experiment_id=f"declared-structure-{bad_at}")
    record = runner.measure(hardware, operator_spec, shape, config, ConfigSpace({"tile": (1,)}), resume=False)
    assert record.status == RecordStatus.CORRECTNESS_ERROR
    assert record.runtime_samples_ms == ()
    correctness = record.metadata["correctness"]
    if bad_at == 2:
        assert correctness["status"] == "failed"
        assert correctness["checked_invocations"][-1]["correct"] is False
    else:
        assert correctness["status"] == "failed"
        assert correctness["post_measurement"]["status"] == "failed"
