"""Real dispatcher path with a synthetic binding; no GPU performance claims."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from kernel_tuning_lab.acceptance import check_algorithm
from kernel_tuning_lab.l3.dispatcher import UnknownShapeError
from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec,KernelConfig
from kernel_tuning_lab.space import ConfigSpace


def test_native_table_calls_existing_dispatcher_and_rejects_uncovered_or_changed_context(tmp_path):
    path=Path(__file__).resolve().parents[1]/'examples/space_production/native_dispatch.py'
    spec=importlib.util.spec_from_file_location('native_dispatch_fixture',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    h=HardwareSpec('synthetic');o=OperatorSpec('fixture');s=ShapeSpec({'M':1,'N':64,'K':128})
    config=KernelConfig({'USE_ATOMIC_ADD':True,'USE_FP32_REDUCE':True})
    space=ConfigSpace({'USE_ATOMIC_ADD':(False,True),'USE_FP32_REDUCE':(True,)})
    space.asset_identity={'kernel_fingerprint':'fixture-only'}
    artifact={'hardware_id':h.id,'operator_id':o.id,'kernel_fingerprint':'fixture-only','config_budget':1,
        'dispatch':{'selected_configs':[config.params],'provenance':'mock',
            'assignments':[{'shape':s.dims,'config':config.params,'config_id':config.id}]}}
    file=tmp_path/'table.json';file.write_text(json.dumps(artifact));options={'table_artifact':str(file),'table_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    assert check_algorithm(module.ASSETS[0],options,7,h,o,s,space)['status']=='passed'
    policy=module.NativeDispatch(**options);assert policy.select(h,o,s,space.read_only())==config
    assert policy.diagnostics['actual_table_lookups']==1
    with pytest.raises(UnknownShapeError):policy.select(h,o,ShapeSpec({'M':2,'N':64,'K':128}),space.read_only())
    with pytest.raises(ValueError,match='binding mismatch'):policy.select(HardwareSpec('different'),o,s,space.read_only())
    with pytest.raises(ValueError,match='SHA mismatch'):module.NativeDispatch(**{**options,'table_sha256':'bad'})
