"""Lazy kernel backend plugins; importing the package never requires CUDA."""

from .base import BackendUnavailableError, KernelAdapter
from .mock import MockAdapter

_PLUGIN_FACTORIES = {}


def register_adapter(name: str, factory, *, replace: bool = False) -> None:
    """Register an application-owned adapter factory without importing a compiler SDK."""
    builtins = {"mock", "triton", "triton_gemm", "triton_batched_gemm", "triton_softmax", "triton_layernorm"}
    if not name or name in builtins:
        raise ValueError("plugin backend name must be nonempty and distinct from built-in backends")
    if name in _PLUGIN_FACTORIES and not replace:
        raise ValueError(f"backend {name!r} is already registered")
    if not callable(factory):
        raise TypeError("adapter factory must be callable")
    _PLUGIN_FACTORIES[name] = factory


def get_adapter(name: str, **kwargs):
    if name in _PLUGIN_FACTORIES:
        return _PLUGIN_FACTORIES[name](**kwargs)
    if name == "mock":
        return MockAdapter(**kwargs)
    if name in {"triton", "triton_gemm", "triton_batched_gemm"}:
        from .triton_gemm import TritonGemmAdapter
        if name == "triton_batched_gemm":
            kwargs["batched"] = True
        return TritonGemmAdapter(**kwargs)
    if name in {"triton_softmax", "triton_layernorm"}:
        from .triton_reductions import TritonLayerNormAdapter, TritonSoftmaxAdapter
        return (TritonSoftmaxAdapter if name == "triton_softmax" else TritonLayerNormAdapter)(**kwargs)
    raise ValueError(f"unknown backend {name!r}; use mock, triton, or register your own KernelAdapter")


__all__ = ["BackendUnavailableError", "KernelAdapter", "MockAdapter", "get_adapter", "register_adapter"]
