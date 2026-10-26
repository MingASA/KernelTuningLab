"""CPU contract tests; synthetic priors are never GPU evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from kernel_tuning_lab.models import HardwareSpec,OperatorSpec,ShapeSpec,SearchBudget
from kernel_tuning_lab.search import SearchHistory
from kernel_tuning_lab.space import ConfigSpace

PATH=Path(__file__).resolve().parents[1]/'examples/crossgpu_anchor/online_search.py'
spec=importlib.util.spec_from_file_location('online_cohort_fixture',PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_existing_online_searchers_never_escape_sparse_cohort():
    configs=[{'BLOCK_M':16,'BLOCK_N':64},{'BLOCK_M':32,'BLOCK_N':128}]
    space=ConfigSpace({'BLOCK_M':[16,32],'BLOCK_N':[64,128]})
    context=(HardwareSpec('synthetic',architecture='mock'),OperatorSpec('gemm',kind='gemm'),ShapeSpec({'M':1,'N':64,'K':32}))
    for method in ('random','heuristic','surrogate'):
        search=module.FiniteSearch(method=method,configs=configs)
        suggestions=search.suggest(*context,space,SearchHistory(),SearchBudget(2),batch_size=2)
        assert len(suggestions)==2 and all(dict(c.params) in configs for c in suggestions)


def test_source_priority_requires_frozen_source_bytes_and_no_target_labels(tmp_path):
    configs=[{'BLOCK_M':16},{'BLOCK_M':32}];shape={'M':1,'N':64,'K':32}
    path=tmp_path/'source-only.json';path.write_text(json.dumps({'configs':configs,'workload':{'shapes':[shape]},'latencies_ms':[[2,1]]}))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    search=module.FiniteSearch(method='source_static_priority',configs=configs,prior_matrix=path,prior_sha256=digest)
    hw,op=HardwareSpec('synthetic'),OperatorSpec('gemm',kind='gemm')
    space=ConfigSpace({'BLOCK_M':[16,32]});history=SearchHistory()
    suggestion=search.suggest(hw,op,ShapeSpec(shape),space,history,SearchBudget(1))
    assert dict(suggestion[0].params)==configs[1] and len(history)==0
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='SHA256'):
        module.FiniteSearch(method='source_static_priority',configs=configs,prior_matrix=path,prior_sha256=digest)
    with pytest.raises(ValueError,match='lacks this Shape'):
        search.suggest(hw,op,ShapeSpec({'M':2,'N':64,'K':32}),space,history,SearchBudget(1))
