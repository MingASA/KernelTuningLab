"""L3 finite config-budget selection and observed Shape dispatch.

No constraint on physical compiled specializations or binary size is implied.
"""

from .dispatcher import ShapeDispatcher, UnknownShapeError
from .matrix import MissingMeasurementError, PerformanceMatrix
from .solver import (
    OracleLimitError,
    SelectionResult,
    bucket_baseline,
    evaluate_selection,
    greedy,
    greedy_local_swap,
    k_loss_curve,
    oracle,
)

__all__ = [
    "PerformanceMatrix", "SelectionResult", "ShapeDispatcher", "MissingMeasurementError",
    "UnknownShapeError", "OracleLimitError", "oracle", "greedy", "greedy_local_swap",
    "bucket_baseline", "evaluate_selection", "k_loss_curve",
]
