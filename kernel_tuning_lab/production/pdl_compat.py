"""Scoped optional-PDL symbols for older Triton dependency hashing.

Triton3.2 resolves even constexpr-disabled attribute references while hashing
the original SGLang JIT body. Private module projections provide a rejecting
JIT symbol, never a no-op, without changing the installed language module or
any numerical function body. Actual PDL execution remains unsupported there.
"""
from functools import lru_cache
from types import ModuleType


def _unsupported_pdl():
    tl.static_assert(False,'PDL needs real gdc compiler intrinsics; this compatibility profile supports USE_PDL=False only')


@lru_cache(maxsize=1)
def _rejecting_symbol():
    global tl
    import triton
    import triton.language as tl
    return triton.jit(_unsupported_pdl)


def language_with_optional_pdl(language):
    cuda=language.extra.cuda
    missing=tuple(name for name in ('gdc_launch_dependents','gdc_wait') if not hasattr(cuda,name))
    if not missing:return language,()
    def project(module):
        result=ModuleType(module.__name__);result.__dict__.update(vars(module));return result
    projected=project(language);extra=project(language.extra);cuda_copy=project(cuda)
    for name in missing:setattr(cuda_copy,name,_rejecting_symbol())
    extra.cuda=cuda_copy;projected.extra=extra
    return projected,missing
