"""Versioned, content-bound submitter assertions; never measurement evidence.

An accepted claim can avoid a repeated comparison.  It cannot turn an assertion
into a platform check, and cannot create a measurement without executing code.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
import math
from types import MappingProxyType
from typing import Any, Mapping

from .models import canonical_json, stable_id
from .space import ValidationResult


_REQUIRED_SCOPE = {"runtimes", "dtypes", "architectures", "shapes", "configs", "parameters"}
_OPTIONAL_SCOPE = {"environment", "hardware_names", "seeds"}


def _scope_json(value):
    if isinstance(value, Mapping):
        return {key: _scope_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_scope_json(item) for item in value]
    return value


def _freeze_scope(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_scope(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_scope(item) for item in value)
    return value


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("claim timestamps must be ISO 8601 strings")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("claim timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class CorrectnessClaim:
    id: str
    source: str
    evidence: tuple[str, ...]
    kernel_ref: str
    kernel_fingerprint: str
    operator_ref: str
    operator_fingerprint: str
    scope: Mapping[str, Any]
    version: str = "1"
    created_at: str | None = None
    expires_at: str | None = None

    def __post_init__(self) -> None:
        for name in ("id", "source", "kernel_ref", "kernel_fingerprint", "operator_ref", "operator_fingerprint", "version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"claim {name} must be a nonempty string")
        if isinstance(self.evidence, str) or not self.evidence or any(not isinstance(item, str) or not item.strip() for item in self.evidence):
            raise ValueError("claim evidence must contain nonempty source references")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not isinstance(self.scope, Mapping):
            raise ValueError("claim scope must be an object")
        if any(not isinstance(key, str) for key in self.scope):
            raise ValueError("claim scope field names must be strings")
        missing = _REQUIRED_SCOPE - self.scope.keys()
        unknown = self.scope.keys() - _REQUIRED_SCOPE - _OPTIONAL_SCOPE
        if missing or unknown:
            raise ValueError(f"claim scope missing={sorted(missing)}, unknown={sorted(unknown)}")
        canonical_json(_scope_json(self.scope))
        object.__setattr__(self, "scope", _freeze_scope(self.scope))
        for name in ("created_at", "expires_at"):
            if getattr(self, name) is not None:
                _timestamp(getattr(self, name))
        if self.created_at and self.expires_at and _timestamp(self.expires_at) <= _timestamp(self.created_at):
            raise ValueError("claim expiry must follow creation")

    @property
    def fingerprint(self) -> str:
        return stable_id(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        value = {entry.name: getattr(self, entry.name) for entry in fields(self)}
        value["scope"] = _scope_json(self.scope)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CorrectnessClaim:
        return cls(**dict(value))


def declare_correctness(kernel, operator_definition, *, id: str, source: str,
                        evidence: tuple[str, ...] | list[str], scope: Mapping[str, Any],
                        version: str = "1", created_at: str | None = None,
                        expires_at: str | None = None) -> CorrectnessClaim:
    """Bind an assertion to the current implementation and reference contents."""
    return CorrectnessClaim(id=id, source=source, evidence=tuple(evidence),
                            kernel_ref=kernel.ref, kernel_fingerprint=kernel.fingerprint,
                            operator_ref=operator_definition.ref,
                            operator_fingerprint=operator_definition.fingerprint,
                            scope=dict(scope), version=version,
                            created_at=created_at, expires_at=expires_at)


def _matches(value: Any, declaration: Any) -> bool:
    if declaration == "*":
        return True
    if isinstance(declaration, Mapping):
        if not declaration or declaration.keys() - {"min", "max"}:
            return False
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
        bounds = list(declaration.values())
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in bounds):
            return False
        return ("min" not in declaration or value >= declaration["min"]) and ("max" not in declaration or value <= declaration["max"])
    choices = declaration if isinstance(declaration, (list, tuple)) else (declaration,)
    # JSON distinguishes booleans from integers, unlike Python equality.
    return any(canonical_json(_scope_json(value)) == canonical_json(_scope_json(item)) for item in choices)


def _mapping_matches(values: Mapping[str, Any], declaration: Any) -> bool:
    if declaration == "*":
        return True
    if not isinstance(declaration, Mapping) or values.keys() != declaration.keys():
        return False
    return all(_matches(value, declaration[key]) for key, value in values.items())


def claim_matches(claim: CorrectnessClaim | Mapping[str, Any], kernel, operator_definition,
                  ctx, config, environment: Mapping[str, Any] | None = None,
                  *, now: datetime | None = None) -> ValidationResult:
    """Match every declared dimension, with no implicit missing-field wildcard."""
    try:
        claim = claim if isinstance(claim, CorrectnessClaim) else CorrectnessClaim.from_dict(claim)
    except (TypeError, ValueError) as exc:
        return ValidationResult(False, (f"invalid correctness claim: {exc}",))
    reasons = []
    for name, actual in (("kernel_ref", kernel.ref), ("kernel_fingerprint", kernel.fingerprint),
                         ("operator_ref", operator_definition.ref), ("operator_fingerprint", operator_definition.fingerprint)):
        if getattr(claim, name) != actual:
            reasons.append(f"claim {name} differs from current asset")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("claim matching time must include a timezone")
    if claim.created_at and _timestamp(claim.created_at) > current:
        reasons.append("claim creation is in the future")
    if claim.expires_at and _timestamp(claim.expires_at) <= current:
        reasons.append("claim has expired")
    scope = claim.scope
    for key, actual in (("runtimes", ctx.runtime), ("dtypes", ctx.operator.dtype),
                        ("architectures", ctx.hardware.architecture), ("hardware_names", ctx.hardware.name),
                        ("seeds", ctx.seed)):
        if key in scope and not _matches(actual, scope[key]):
            reasons.append(f"claim scope does not cover {key}")
    params = config.params if hasattr(config, "params") else config
    for key, actual in (("shapes", ctx.shape.dims), ("configs", params), ("parameters", ctx.operator.parameters)):
        if not _mapping_matches(actual, scope[key]):
            reasons.append(f"claim scope does not cover {key}")
    if "environment" in scope:
        declared = scope["environment"]
        if environment is None:
            reasons.append("claim requires environment evidence which was not supplied")
        elif declared != "*":
            if not isinstance(declared, Mapping) or any(key not in environment or not _matches(environment[key], value) for key, value in declared.items()):
                reasons.append("claim environment does not match current runtime")
    return ValidationResult(not reasons, tuple(reasons))


__all__ = ["CorrectnessClaim", "declare_correctness", "claim_matches"]
