"""CPU-only checks for the declared native Blackwell Triton runtime baseline."""

from __future__ import annotations

import re

from ..space import ValidationResult


def _version(value: str | None) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(value or ""))
    return tuple(int(part or 0) for part in match.groups()) if match else None


def validate_triton_runtime(architecture: str, *, torch_version: str | None,
                            cuda_version: str | None, triton_version: str | None) -> ValidationResult:
    """Require the project's native sm_12x baseline, not speculative PTX fallback.

    These are minimum runtime declarations, not proof that a kernel compiles or
    is numerically correct. Newer CUDA 13.x/Torch/Triton versions are accepted.
    Other architectures retain their existing runtime contract.
    """
    architecture = architecture.lower().replace("_", "")
    if not re.fullmatch(r"sm12\d", architecture):
        return ValidationResult(True)
    reasons = []
    for label, value, minimum in (("PyTorch", torch_version, (2, 7, 0)),
                                  ("PyTorch CUDA runtime", cuda_version, (12, 8, 0)),
                                  ("Triton", triton_version, (3, 3, 0))):
        parsed = _version(value)
        if parsed is None or parsed < minimum:
            reasons.append(f"{architecture} requires {label} >= {minimum[0]}.{minimum[1]} for the declared native backend; found {value!r}")
    return ValidationResult(not reasons, tuple(reasons))
