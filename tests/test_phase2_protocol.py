"""Frozen split and provenance guards; synthetic fixtures are not GPU data."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('phase2_analysis_for_test',ROOT/'scripts/analyze_phase2.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def test_historical_geometry_is_never_presented_as_new_heldout_and_prior_is_in_cohort():
    design=analysis.design
    old={analysis.core.ShapeSpec(s['shape']).id for s in design.HISTORICAL}
    new={analysis.core.ShapeSpec(s['shape']).id for s in design.SHAPES if s['role']=='new_heldout'}
    assert not old&new
    assert all(s['split']=='train' for s in design.SHAPES if s['role']=='historically_explored')
    assert analysis.OLD_CONSTANT.id in {c.id for c in analysis.CONFIGS}
    assert len(analysis.CONFIGS)==48 and len(new)==14


def test_phase2_prior_cannot_be_frozen_after_test_acquisition(tmp_path):
    (tmp_path/'test').mkdir()
    (tmp_path/'test/research_manifest.json').write_text(json.dumps({'jobs':[{'status':'running'}]}))
    with pytest.raises(ValueError,match='retrospective'):
        analysis.freeze_prior(tmp_path)
    assert not (tmp_path/'pretest/prior.json').exists()
