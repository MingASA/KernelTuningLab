"""Real CPU adapter regressions for non-overlapping tuning phase wall costs."""
import pytest
from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.catalog import get_catalog
from kernel_tuning_lab.kernel_runtime import AssetKernelAdapter
from kernel_tuning_lab.models import OperatorSpec,ShapeSpec,RecordStatus
from kernel_tuning_lab.runner import BenchmarkRunner
from kernel_tuning_lab.storage import ExperimentStore


def fixture():
    kernel=get_catalog().resolve('kernel','python/gemm')
    definition=get_catalog().resolve('operator','gemm')
    adapter=AssetKernelAdapter(kernel,definition,accept_claims=False)
    hardware=adapter.probe_hardware()
    operator=OperatorSpec('gemm',kind='gemm',implementation=kernel.ref,dtype='float32',version=definition.version)
    shape=ShapeSpec({'M':2,'N':3,'K':5})
    space=build_effective_space(kernel,hardware,operator,shape)
    return adapter,hardware,operator,shape,space,next(space.iter_configs())


def test_phase_costs_are_source_costs_preserved_by_cache(tmp_path):
    adapter,hw,op,shape,space,config=fixture()
    with ExperimentStore(tmp_path/'cost.sqlite') as store:
        runner=BenchmarkRunner(adapter,store,warmup=2,repetitions=3,seed=7)
        record=runner.measure(hw,op,shape,config,space)
        assert record.successful,record.failure_reason
        costs=record.metadata['phase_wall_time_ms']
        assert set(costs)=={'preflight_and_cache','preparation','compile','initial_correctness','warmup','measurement','post_correctness'}
        assert all(v>=0 for v in costs.values())
        assert sum(costs.values())<=record.wall_time_s*1000
        assert costs['compile']==record.compile_time_ms
        cached=runner.measure(hw,op,shape,config,space)
        assert runner.last_cache_hit and cached.id==record.id
        assert cached.metadata['phase_wall_time_ms']==costs


def test_failed_compilation_keeps_completed_phase_cost(tmp_path,monkeypatch):
    adapter,hw,op,shape,space,config=fixture()
    def fail(*args): raise RuntimeError('intentional unit-test compile failure')
    monkeypatch.setattr(adapter,'compile',fail)
    record=BenchmarkRunner(adapter,warmup=0,repetitions=1).measure(hw,op,shape,config,space)
    assert record.status==RecordStatus.COMPILE_ERROR
    assert record.metadata['phase_wall_time_ms']['compile']==record.compile_time_ms
    assert set(record.metadata['phase_wall_time_ms'])=={'preflight_and_cache','preparation','compile'}
