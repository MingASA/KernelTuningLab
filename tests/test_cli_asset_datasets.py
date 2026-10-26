"""Synthetic unit labels exercise SDK/CLI interoperability, never GPU evidence."""
from dataclasses import replace

from kernel_tuning_lab.catalog import get_catalog
from kernel_tuning_lab.cli import _infer_space, _l3_evaluate
from kernel_tuning_lab.models import HardwareSpec,KernelConfig,MeasurementRecord,OperatorSpec,RecordStatus,SearchBudget,ShapeSpec
from kernel_tuning_lab.replay import run_replay
from kernel_tuning_lab.search import GridSearch


def _record(kernel_ref, config, dtype='float32'):
    kernel=get_catalog().resolve('kernel',kernel_ref)
    return MeasurementRecord(HardwareSpec('unit-cpu',architecture='cpu',warp_size=1,max_threads_per_block=1),
        OperatorSpec('gemm',kind='gemm',implementation=kernel_ref,dtype=dtype,version='1.0.0'),
        ShapeSpec({'M':2,'N':3,'K':4}),KernelConfig(config),RecordStatus.SUCCESS,
        runtime_samples_ms=(1.,),provenance='mock',split='test',environment={'kernel_fingerprint':kernel.fingerprint},
        metadata={'correctness_verified':True,'protocol':{'contract':'asset_kernel_v1'}})


def test_sdk_cpu_gemm_replay_uses_its_tile_domain_instead_of_cuda_gemm_constraints():
    record=_record('python/gemm@1.0.0',{'tile_size':4})
    space=_infer_space([record])
    assert space.validate(record.config,record.hardware,record.operator,record.shape).valid
    result=run_replay([record],GridSearch(),space,SearchBudget(1))
    assert result.best_config==record.config and result.oracle_best_latency_ms==1.
    assert result.coverage_scope=='observed_domains_with_current_asset_constraints'


def test_fixed_sdk_kernel_reconstructs_exactly_one_config_without_a_dummy_parameter():
    from kernel_tuning_lab.hardware import get_hardware_profile
    record=replace(_record('torch/mm@1.0.0',{},'float16'),hardware=get_hardware_profile('5060ti16g'))
    # This unit fixture does not execute the CUDA implementation.
    space=_infer_space([record])
    assert space.cardinality==1 and list(space.iter_configs())==[KernelConfig({})]
    assert space.validate(record.config,record.hardware,record.operator,record.shape).valid


def test_old_or_unmatched_sdk_content_is_explicitly_observed_only():
    record=replace(_record('python/gemm@1.0.0',{'tile_size':4}),environment={'kernel_fingerprint':'older-python-or-source'})
    space=_infer_space([record])
    result=run_replay([record],GridSearch(),space,SearchBudget(1))
    assert result.best_config==record.config
    assert result.coverage_scope=='observed_configs_only'
    assert space.identity['reason']=='asset_content_unmatched'


def test_cli_l3_separates_timings_in_one_journal_instead_of_mixing_or_aborting():
    first=_record('python/gemm@1.0.0',{'tile_size':4})
    second=replace(first,measurement_mode='separate_unit_protocol',runtime_samples_ms=(2.,))
    results,unavailable=_l3_evaluate([first,second])
    assert results and not unavailable
    assert len({row['source_context_id'] for row in results})==2
    assert {row['reference_latency_ms'] for row in results}=={1.,2.}
