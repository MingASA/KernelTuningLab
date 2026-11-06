"""Pre-measurement finite cohort and resource accounting, with no CUDA imports."""
import importlib.util
import itertools
from pathlib import Path

from kernel_tuning_lab.catalog import get_catalog
from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.hardware import get_hardware_profile
from kernel_tuning_lab.models import KernelConfig,OperatorSpec,ShapeSpec
from kernel_tuning_lab.research import expand_research,read_json

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('crossgpu_design',ROOT/'examples/crossgpu_anchor/design.py')
design=importlib.util.module_from_spec(spec);spec.loader.exec_module(design)


def test_frozen_anchor_counts_match_physical_and_acceptance_budgets():
    folder=ROOT/'examples/crossgpu_anchor'
    for name,observations,acceptance in [('anchor',192,48),('pilot',24,24)]:
        expanded=expand_research(read_json(folder/(name+'.json')),directory=folder)
        assert expanded['planned']['max_observations']==observations
        assert expanded['planned']['max_validation_evaluations']==acceptance
    assert len({KernelConfig(c).id for c in design.CONFIGS})==16


def test_candidate_configs_fit_declared_sdk_domains_and_include_synthetic_first_point():
    catalog=get_catalog();kernel=catalog.resolve('kernel','triton/t1_gemm@1.0.0')
    definition=catalog.resolve('operator',kernel.operator)
    operator=OperatorSpec(definition.id,kind=definition.id,implementation=kernel.ref,version=definition.version,dtype='float16')
    # Profiles here are only CPU static schema/resource checks, never GPU evidence.
    for profile,entry in itertools.product(('5060ti16g','4090','H800'),design.SHAPES):
        hw=get_hardware_profile(profile);shape=ShapeSpec(entry['dims'])
        space=build_effective_space(kernel,hw,operator,shape,design.SPACE)
        assert space.config_at(0).id in {KernelConfig(c).id for c in design.CONFIGS}
        assert all(space.validate(KernelConfig(c),hw,operator,shape).valid for c in design.CONFIGS)
