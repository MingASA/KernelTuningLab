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


def test_factorial_effects_separate_interaction_from_scale():
    import math
    configs=tuple(KernelConfig(c) for c in analysis.design.configs())
    def sign(c,k): return 1 if c.params[k]==analysis.design.SPACE[k][1] else -1
    values=[math.exp(2+.3*sign(c,'BLOCK_M')+.4*sign(c,'BLOCK_M')*sign(c,'num_stages')) for c in configs]
    effects=analysis.factorial_effects(configs,values,analysis.design.SPACE)
    by_term={tuple(t['parameters']):t for t in effects['terms']}
    assert by_term['BLOCK_M',]['log_variance_fraction']==pytest.approx(.36)
    assert by_term['BLOCK_M','num_stages']['log_variance_fraction']==pytest.approx(.64)
    assert effects['higher_order_or_noise_fraction']==pytest.approx(0,abs=1e-10)


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


def test_l3_research_report_keeps_all_solver_series_and_validates_dispatch():
    configs=tuple(KernelConfig(c) for c in analysis.design.configs()[:4])
    shapes=tuple(ShapeSpec(dict(zip(('M','N','K'),s[2]))) for s in analysis.design.SHAPES)
    matrix=PerformanceMatrix(WorkloadDistribution(shapes,tuple([1]*10)),configs,
        tuple(tuple(1.+(i+j)%4 for j in range(4)) for i in range(10)),'mock')
    # Clearly labelled synthetic fixture for report semantics, never GPU data.
    record=MeasurementRecord(HardwareSpec('unit',architecture='sm_120',smem_per_block=101376),
        OperatorSpec('gemm',kind='gemm',implementation='triton/t1_gemm@1.0.0',dtype='float16',version='1.0.0'),
        shapes[0],configs[0],RecordStatus.SUCCESS,runtime_samples_ms=(1.,),provenance='mock',
        metadata={'correctness_verified':True,'correctness':{'verified_on_cpu':True}})
    result=analysis.l3_study(matrix,[record])
    for weight in ('uniform','synthetic_decode_skew'):
        for k in (1,2,4,8):
            rows=[r for r in result['results'] if r['weights']==weight and r['k']==k]
            assert {r['method'] for r in rows}=={'greedy','local_swap','exact'}
            assert all(r['dispatch_legality'].startswith('passed') for r in rows)


def test_prior_freeze_refuses_retrospective_test_access(tmp_path):
    import json
    (tmp_path/'research_manifest.json').write_text(json.dumps({'jobs':[{'status':'running','experiment':{'split':'test'}}]}))
    with pytest.raises(ValueError,match='retrospective'):
        analysis.freeze_training_prior(tmp_path,tmp_path/'prior')
    assert not (tmp_path/'prior/training-prior.json').exists()
