"""Stable contributor-facing imports, available without CUDA or optional SDKs."""

from .assets import (KernelDefinition, OperatorDefinition, ParameterSpec, SearchAlgorithmDefinition,
                     SearchCapabilities, SpaceContext, SpaceDefinition)
from .asset_space import SpaceView
from .catalog import AssetCatalog, get_catalog
from .models import (HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec,
                     RecordStatus, SearchBudget, ShapeSpec, WorkloadDistribution)
from .search import SearchHistory
from .storage import ExperimentStore
from .space import ValidationResult


def __getattr__(name):
    if name in {"ExecutionContext", "ExecutionPlan", "ExecutionStep", "AssetKernelAdapter"}:
        from . import kernel_runtime
        return getattr(kernel_runtime, name)
    if name in {"CorrectnessClaim", "declare_correctness", "claim_matches"}:
        from . import claims
        return getattr(claims, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["KernelDefinition", "OperatorDefinition", "ParameterSpec", "SearchAlgorithmDefinition",
           "SearchCapabilities", "SpaceContext", "SpaceDefinition", "SpaceView", "AssetCatalog", "get_catalog",
           "HardwareSpec", "KernelConfig", "OperatorSpec", "SearchBudget", "ShapeSpec", "ValidationResult",
           "MeasurementRecord", "RecordStatus", "SearchHistory", "WorkloadDistribution", "ExperimentStore",
           "ExecutionContext", "ExecutionPlan", "ExecutionStep", "AssetKernelAdapter",
           "CorrectnessClaim", "declare_correctness", "claim_matches"]
