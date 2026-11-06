"""Generic asset remains CPU-discoverable; actual reuse is a GPU experiment."""
from kernel_tuning_lab.catalog import get_catalog
from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.hardware import get_hardware_profile
from kernel_tuning_lab.models import OperatorSpec,ShapeSpec


def test_generic_and_specialized_t1_have_distinct_identities_and_same_legal_config_domains():
    catalog=get_catalog();special=catalog.resolve('kernel','triton/t1_gemm@1.0.0');generic=catalog.resolve('kernel','triton/t1_gemm_generic@1.0.0')
    assert special.fingerprint!=generic.fingerprint
    hardware=get_hardware_profile('5060ti16g');shape=ShapeSpec({'M':33,'N':129,'K':65})
    domains=[]
    for kernel in (special,generic):
        operator=OperatorSpec('gemm',kind='gemm',implementation=kernel.ref,dtype='float16',version='1.0.0')
        space=build_effective_space(kernel,hardware,operator,shape)
        domains.append(space.domains)
        assert len(list(space.legal_configs(hardware,operator,shape)))==192
    assert domains[0]==domains[1]
