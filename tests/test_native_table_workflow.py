"""CPU finite-workload contracts, not GPU/L3 performance research evidence."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import pytest

from kernel_tuning_lab.asset_runner import execute_asset_experiment
from kernel_tuning_lab.l3 import MissingMeasurementError,UnknownShapeError
from kernel_tuning_lab.models import ShapeSpec,KernelConfig
from kernel_tuning_lab.production.dispatch import ProductionTable
from kernel_tuning_lab.space import ConfigSpace


ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('native_table_workflow',ROOT/'examples/production_suite/native_table_workflow.py')
workflow=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(workflow)


@pytest.fixture
def measured_cpu(tmp_path):
    spec={'schema_version':2,'name':'native-table-CPU-contract','operator':'gemm@1.0.0','dtype':'float32',
        'kernels':[{'asset':'python/gemm@1.0.0','space':{'tile_size':[1,4]}}],'algorithms':['grid@1.0.0'],
        'shapes':[{'M':2,'N':3,'K':5},{'M':3,'N':4,'K':6}],'warmup':0,'repetitions':3,
        'budget':{'max_evaluations':2,'max_wall_time_s':10},'validation':{'max_configs':2,'max_cases':0,'max_wall_time_s':10,'trust_claims':False}}
    path=tmp_path/'spec.json';path.write_text(json.dumps(spec));output=tmp_path/'source-run'
    assert execute_asset_experiment(path,output)['status']=='passed'
    rows=workflow.load_records([output/'measurements.sqlite'])
    context=next(workflow.context_key(row) for row in rows if row.repetitions==3)
    return path,rows,context


def test_cpu_matrix_export_table_and_next_spec_actually_dispatch(tmp_path,measured_cpu):
    spec,rows,context=measured_cpu
    output=tmp_path/'tables'
    report=workflow.build_workflow(rows,spec,output,'python/gemm@1.0.0',context_id=context,allow_fixture=True)
    assert report['status']=='exported_not_executed' and not report['GPU_evidence']
    assert report['shape_count']==report['candidate_count']==2
    for entry in report['selections']:
        assert entry['physical_compiled_variants'] is None and entry['graphs'] is None
        assert entry['native_rule']['actual_config_count'] is None
    result=execute_asset_experiment(output/'dispatch-k1.json',tmp_path/'actual-dispatch')
    assert result['status']=='passed' and result['observations']==2 and not result['physical_gpu_validated']
    binding=json.loads((output/'config-table-k1.json').read_text());selected=report['selections'][0]
    policy=ProductionTable(table_artifact=output/'config-table-k1.json',table_sha256=selected['table_sha256'])
    first=next(row for row in rows if row.repetitions==3)
    space=ConfigSpace({'tile_size':(1,4)});space.asset_identity={'kernel_fingerprint':binding['kernel_fingerprint']}
    with pytest.raises(UnknownShapeError):policy.select(first.hardware,first.operator,ShapeSpec({'M':99,'N':3,'K':5}),space.read_only())


def test_missing_cell_protocol_and_semantic_mixing_are_refused(tmp_path,measured_cpu):
    spec,rows,context=measured_cpu
    research=[row for row in rows if workflow.context_key(row)==context]
    with pytest.raises(ValueError,match='multiple hardware/semantics/timing'):
        workflow.select_cohort(rows,'python/gemm@1.0.0',allow_fixture=True)
    with pytest.raises(MissingMeasurementError):
        workflow.build_workflow(research[:-1],spec,tmp_path/'missing','python/gemm@1.0.0',allow_fixture=True)
    changed=replace(research[0],operator=replace(research[0].operator,parameters={'different_semantics':True}))
    with pytest.raises(ValueError,match='multiple hardware/semantics/timing'):
        workflow.select_cohort((*research,changed),'python/gemm@1.0.0',allow_fixture=True)
    with pytest.raises(ValueError,match='at least two Shapes'):
        workflow.select_cohort([row for row in research if row.shape==research[0].shape],'python/gemm@1.0.0',allow_fixture=True)


def test_unknown_or_outside_native_config_does_not_become_equal_candidate_baseline(measured_cpu):
    _,rows,context=measured_cpu;explicit=tuple(row for row in rows if workflow.context_key(row)==context)
    row=explicit[0];metadata=deepcopy(row.metadata);metadata['correctness']['execution']['annotations']={}
    native=replace(row,provenance='mock',operator=replace(row.operator,implementation='native/rule-fixture'),config=KernelConfig({}),metadata=metadata)
    configs=tuple({r.config.id:r.config for r in explicit}.values())
    report=workflow.native_comparison((native,),explicit,configs,1)
    assert report['actual_config_count'] is None and report['same_candidate_coverage'] is None
    metadata=deepcopy(metadata);metadata['correctness']['execution']['annotations']={'native_selected_configuration':{'tile_size':999}}
    outside=replace(native,metadata=metadata)
    report=workflow.native_comparison((outside,),explicit,configs,1)
    assert report['actual_config_count']==1 and report['same_candidate_coverage'] is False
