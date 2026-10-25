"""Analysis/reveal protocol tests; synthetic unit fixtures are not GPU evidence."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import pytest

from kernel_tuning_lab.l3 import PerformanceMatrix
from kernel_tuning_lab.models import HardwareSpec,KernelConfig,OperatorSpec,ShapeSpec,WorkloadDistribution,MeasurementRecord,RecordStatus

spec=importlib.util.spec_from_file_location("phase1_analysis",Path(__file__).resolve().parents[1]/'scripts/analyze_phase1.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def test_rank_statistics_and_pareto_are_finite_and_do_not_reverse_objective():
    assert analysis.correlation([1,2,3],[10,20,30])==pytest.approx(1)
    assert analysis.correlation([1,2,3],[30,20,10])==pytest.approx(-1)
    assert analysis.ranks([2,2,4])==[1.5,1.5,3.]
    assert analysis.pareto_indices(((1.,2.,3.),(3.,2.,4.)))==[0,1]


def test_joint_selection_cannot_use_unrevealed_test_label():
    configs=tuple(KernelConfig(c) for c in analysis.design.configs())
    shapes=tuple(ShapeSpec(dict(zip(('M','N','K'),s[2]))) for s in analysis.design.SHAPES)
    rows=tuple(tuple(1+(j*7+i*3)%17 for j in range(64)) for i in range(10))
    matrix=PerformanceMatrix(WorkloadDistribution(shapes,tuple([1]*10)),configs,rows,'unspecified')
    first=analysis.joint_study(matrix,[])['results']
    result=next(r for r in first if r['seed']==0 and r['budget']==32 and r['method']=='random_common_panel')
    acquired={identity for _,identity in result['acquired_cell_ids']}
    hidden=next(j for j,c in enumerate(configs) if c.id not in acquired)
    changed=[list(row) for row in rows];changed[6][hidden]=.001
    altered=replace(matrix,latencies_ms=tuple(tuple(row) for row in changed))
    second=analysis.joint_study(altered,[])['results']
    other=next(r for r in second if r['seed']==0 and r['budget']==32 and r['method']=='random_common_panel')
    assert result['acquired_cell_ids']==other['acquired_cell_ids']
    assert result['selected_config_ids']==other['selected_config_ids']
    assert result['mean_shape_regret']!=other['mean_shape_regret']
    assert all(r['revealed_cells']<=r['budget'] for r in first+second)


def test_baseline_comparison_rejects_dtype_and_protocol_mixing():
    record=MeasurementRecord(HardwareSpec('unit'),OperatorSpec('gemm',kind='gemm',dtype='float16'),
        ShapeSpec({'M':1}),KernelConfig({'tile':1}),RecordStatus.SUCCESS,runtime_samples_ms=(1.,),metadata={'protocol':{}})
    analysis.require_baseline_compatible(record,record)
    with pytest.raises(ValueError,match='mismatch'):
        analysis.require_baseline_compatible(record,replace(record,operator=replace(record.operator,dtype='bfloat16')))
    with pytest.raises(ValueError,match='protocol mismatch'):
        analysis.require_baseline_compatible(record,replace(record,metadata={'protocol':{'cuda_graph':{'calls_per_replay':128}}}))
