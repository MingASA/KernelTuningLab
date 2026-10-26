"""Finite analysis math and CPU-only target preparation, using synthetic costs."""
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_phase2_supplements import mixed_selection


def test_physical_poc_selector_matches_exhaustive_binary_subsets():
    generic=[[8,2,9,3],[4,7,3,6]]
    specialized=[1,1,2,2];weights=[4,1,2,1]
    variants=generic+[[value if i==j else float('inf') for i in range(4)] for j,value in enumerate(specialized)]
    for k in (1,2,3,4,6):
        scores=[]
        for size in range(1,k+1):
            for subset in itertools.combinations(variants,size):
                scores.append(sum(w*min(v[i] for v in subset) for i,w in enumerate(weights))/sum(weights))
        result=mixed_selection(generic,specialized,weights,k)
        assert result['weighted_latency_ms']==pytest.approx(min(scores))
        assert result['used_compiler_identities']<=k
    assert mixed_selection(generic,specialized,weights,2,'specialized') is None


def test_target_preparation_is_static_and_has_no_guessed_h20_profile(tmp_path):
    output=tmp_path/'h20'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_target_gpu.py'),'--target','h20','--output',str(output)],
        capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['executed'] is False
    source=json.loads((output/'target-smoke.json').read_text())
    assert source['defaults']['hardware'] is None
    manifest=json.loads((output/'study/research_manifest.json').read_text())
    assert manifest['status']=='generated'
    assert all(job['status']=='not_run' for job in manifest['jobs'])
    assert manifest['planned']['max_observations']==16
    assert not list(output.rglob('measurements.sqlite'))
