"""CPU-only contracts and actual compiled identity provenance, no fake latency."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.catalog import get_catalog
from kernel_tuning_lab.kernel_runtime import ExecutionContext
from kernel_tuning_lab.models import HardwareSpec, OperatorSpec, ShapeSpec


def test_t1_controlled_space_remains_legal_at_small_m_and_tails():
    path=Path(__file__).resolve().parents[1]/"examples/research_phase1/design.py"
    module_spec=importlib.util.spec_from_file_location("phase1_design",path)
    design=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(design)
    kernel=get_catalog().resolve("kernel","triton/t1_gemm")
    hardware=HardwareSpec("physical-profile-test",architecture="sm_120",smem_per_block=101376)
    operator=OperatorSpec("gemm",kind="gemm",implementation=kernel.ref,dtype="float16")
    for _,_,dims,_ in design.SHAPES:
        shape=ShapeSpec(dict(zip(("M","N","K"),dims)))
        space=build_effective_space(kernel,hardware,operator,shape,design.SPACE)
        assert space.cardinality==64
        assert all(space.validate(c,hardware,operator,shape).valid for c in space.iter_configs())


def test_compiled_artifact_identity_is_separate_from_configuration():
    ctx=ExecutionContext(HardwareSpec("test"),OperatorSpec("gemm"),ShapeSpec({"M":1}),"python")
    compiled=SimpleNamespace(metadata=SimpleNamespace(shared=32,num_warps=4),n_regs=16,
                             hash="real-compiler-key",asm={"cubin":b"binary","ptx":"code","ttir":"not an executable"})
    ctx.record_compiled_kernel("stage",compiled)
    resource=ctx.metadata["execution_resources"]["stage"]
    assert resource["compiler_cache_identity"]=="real-compiler-key"
    assert resource["artifacts"]["cubin"]=={"sha256":hashlib.sha256(b"binary").hexdigest(),"bytes":6}
    assert set(resource["artifacts"])=={"cubin","ptx"}
    ctx.record_compiled_kernel("unknown",SimpleNamespace())
    assert ctx.metadata["execution_resources"]["unknown"]["compiler_cache_identity"] is None
    assert ctx.metadata["execution_resources"]["unknown"]["artifacts"]=={}
