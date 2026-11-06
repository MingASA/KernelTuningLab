"""Public, CPU-safe declarations for independently reusable extension assets.

The declarations contain no device allocation or compiler imports.  A manifest
records content identities rather than relying on a contributor's version alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from collections.abc import Sequence
from copy import deepcopy
import functools
import hashlib
import inspect
import keyword
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .models import HardwareSpec, KernelConfig, OperatorSpec, ShapeSpec, canonical_json, stable_id


_UNSET = object()
_PARAMETER_KINDS = ("integer", "real", "boolean", "categorical")
_FILE_HASHES: dict[tuple[str, int, int, int], str] = {}


def _file_digest(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    if key not in _FILE_HASHES:
        _FILE_HASHES[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _FILE_HASHES[key]


def _jsonable(value: Any) -> Any:
    """Convert declaration values without deepcopying mapping proxies/callables."""
    if isinstance(value, SpaceContext):
        return value.to_dict()
    if isinstance(value, _Asset):
        return {"asset": value.ref, "fingerprint": value.fingerprint}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return callable_identity(value)
    canonical_json(value)
    return value


def _capture_identity(value: Any) -> Any:
    try:
        return _jsonable(value)
    except (TypeError, ValueError, RecursionError):
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def callable_identity(function: Callable) -> dict[str, Any]:
    """Bind callable identity to its module contents and serializable captures.

    Explicit ``source_files`` on an asset cover helper files imported by that
    module.  Python cannot reliably infer an arbitrary transitive import graph.
    """
    if isinstance(function, functools.partial):
        return {"partial": callable_identity(function.func),
                "args": _capture_identity(function.args),
                "keywords": _capture_identity(function.keywords or {})}
    target = function if inspect.isfunction(function) or inspect.ismethod(function) else type(function)
    result: dict[str, Any] = {"entry": getattr(target, "__qualname__", type(function).__qualname__)}
    try:
        filename = inspect.getsourcefile(target) or inspect.getfile(target)
    except (TypeError, OSError):
        filename = None
    if filename and Path(filename).is_file():
        source = Path(filename)
        result["source"] = {"name": source.name, "sha256": _file_digest(source)}
    else:
        result["module"] = getattr(target, "__module__", type(function).__module__)
    code = getattr(function, "__code__", None)
    if code is not None:
        result["bytecode_sha256"] = hashlib.sha256(code.co_code).hexdigest()
        result["constants"] = [_capture_identity(item) for item in code.co_consts]
        result["defaults"] = _capture_identity(getattr(function, "__defaults__", None))
        result["keyword_defaults"] = _capture_identity(getattr(function, "__kwdefaults__", None))
        if function.__closure__:
            captured = []
            for cell in function.__closure__:
                try:
                    captured.append(_capture_identity(cell.cell_contents))
                except ValueError:
                    captured.append({"empty_cell": True})
            result["closure"] = captured
    elif not inspect.isclass(function):
        state = getattr(function, "__dict__", None)
        if state:
            result["state"] = _capture_identity(state)
    return result


def _source_manifest(source_files: tuple[str, ...], callbacks: tuple[Callable | None, ...]) -> list[dict[str, str]]:
    base = None
    for callback in callbacks:
        if callback is None:
            continue
        try:
            filename = inspect.getsourcefile(callback)
        except (TypeError, OSError):
            filename = None
        if filename:
            base = Path(filename).resolve().parent
            break
    result = []
    for filename in source_files:
        path = Path(filename).expanduser()
        if not path.is_absolute() and base is not None:
            path = base / path
        if not path.is_file():
            raise ValueError(f"declared source file does not exist: {path}")
        result.append({"name": path.name, "sha256": _file_digest(path)})
    return sorted(result, key=lambda entry: (entry["name"], entry["sha256"]))


def _check_asset_header(asset) -> None:
    if not isinstance(asset.id, str) or not asset.id.strip() or any(character.isspace() for character in asset.id) or "@" in asset.id:
        raise ValueError("asset id must be a nonempty string without '@'")
    if not isinstance(asset.version, str) or not asset.version.strip() or any(character.isspace() for character in asset.version) or "@" in asset.version:
        raise ValueError("asset version must be a nonempty string without '@'")
    if type(asset.api_version) is not int or asset.api_version != 1:
        raise ValueError("unsupported asset API version; expected 1")


class _Asset:
    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    @property
    def fingerprint(self) -> str:
        return stable_id(self.manifest())


def _freeze_space_value(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("space environment keys must be strings")
        return MappingProxyType({key: _freeze_space_value(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_space_value(item) for item in value)
    canonical_json(value)
    return deepcopy(value)


class SpaceContext:
    """Immutable resolution inputs, separate from physical hardware identity.

    Hardware/operator/shape are defensive snapshots: reading those properties
    returns a copy, including their mutable metadata and parameter mappings.
    Environment mappings and sequences are recursively read-only. The context
    contains software facts and the execution workspace budget without adding
    them to a HardwareSpec or changing its identity.
    """

    __slots__ = ("_hardware", "_operator", "_shape", "_environment", "_workspace_memory_limit_bytes")

    def __init__(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec,
                 environment: Mapping[str, Any] | None = None,
                 workspace_memory_limit_bytes: int = 256 * 1024**2):
        if not isinstance(hardware, HardwareSpec) or not isinstance(operator, OperatorSpec) or not isinstance(shape, ShapeSpec):
            raise TypeError("SpaceContext requires HardwareSpec, OperatorSpec and ShapeSpec inputs")
        if environment is not None and not isinstance(environment, Mapping):
            raise TypeError("SpaceContext environment must be a mapping")
        if (isinstance(workspace_memory_limit_bytes, bool) or not isinstance(workspace_memory_limit_bytes, int)
                or workspace_memory_limit_bytes <= 0):
            raise ValueError("workspace_memory_limit_bytes must be a positive integer")
        object.__setattr__(self, "_hardware", deepcopy(hardware))
        object.__setattr__(self, "_operator", deepcopy(operator))
        object.__setattr__(self, "_shape", deepcopy(shape))
        object.__setattr__(self, "_environment", _freeze_space_value(environment or {}))
        object.__setattr__(self, "_workspace_memory_limit_bytes", workspace_memory_limit_bytes)

    def __setattr__(self, name, value):
        raise AttributeError("SpaceContext is read-only")

    def __deepcopy__(self, memo):
        # Public reads cannot mutate the captured snapshot.
        return self

    @property
    def hardware(self) -> HardwareSpec:
        return deepcopy(self._hardware)

    @property
    def operator(self) -> OperatorSpec:
        return deepcopy(self._operator)

    @property
    def shape(self) -> ShapeSpec:
        return deepcopy(self._shape)

    @property
    def environment(self) -> Mapping[str, Any]:
        return self._environment

    @property
    def workspace_memory_limit_bytes(self) -> int:
        return self._workspace_memory_limit_bytes

    def to_dict(self) -> dict[str, Any]:
        return {"hardware": _jsonable(self._hardware), "operator": _jsonable(self._operator),
                "shape": _jsonable(self._shape), "environment": _jsonable(self._environment),
                "workspace_memory_limit_bytes": self._workspace_memory_limit_bytes}

    @property
    def fingerprint(self) -> str:
        return stable_id(self.to_dict())


@dataclass(frozen=True)
class ParameterSpec:
    values: tuple[Any, ...]
    kind: str = "auto"
    active_when: Mapping[str, Any] = field(default_factory=dict)
    default: Any = field(default=_UNSET)
    description: str = ""
    unit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.description, str) or (self.unit is not None and not isinstance(self.unit, str)):
            raise ValueError("parameter description/unit must be strings")
        if isinstance(self.values, (str, bytes)) or not isinstance(self.values, Sequence):
            raise ValueError("parameter values must be an explicit finite sequence")
        values = tuple(deepcopy(value) for value in self.values)
        if not values:
            raise ValueError("parameter values must be nonempty")
        identities = tuple(canonical_json(value) for value in values)
        if len(set(identities)) != len(values):
            raise ValueError("parameter values must be distinct")
        kind = self.kind
        if kind == "auto":
            kind = ("boolean" if all(isinstance(value, bool) for value in values) else
                    "integer" if all(isinstance(value, int) and not isinstance(value, bool) for value in values) else
                    "real" if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values) else
                    "categorical")
        if kind not in _PARAMETER_KINDS:
            raise ValueError(f"unsupported parameter kind: {kind!r}")
        if kind == "boolean" and not all(isinstance(value, bool) for value in values):
            raise ValueError("boolean parameters require boolean values")
        if kind == "integer" and not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise ValueError("integer parameters require integer values")
        if kind == "real" and not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values):
            raise ValueError("real parameters require finite numeric values")
        default = values[0] if self.default is _UNSET else self.default
        if canonical_json(default) not in identities:
            raise ValueError("parameter default must belong to its declared values")
        active_when = deepcopy(dict(self.active_when))
        canonical_json(active_when)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "active_when", MappingProxyType(active_when))

    def to_dict(self) -> dict[str, Any]:
        return {"values": _jsonable(self.values), "kind": self.kind,
                "active_when": _jsonable(self.active_when), "default": _jsonable(self.default),
                "description": self.description, "unit": self.unit}


@dataclass(frozen=True)
class SpaceDefinition:
    parameters: Mapping[str, ParameterSpec]
    constraints: tuple[Callable, ...] = ()
    name: str = "custom"
    resolver: Callable | None = None
    rules: tuple = ()
    advisories: tuple[Callable, ...] = ()
    candidates: tuple | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("space name must be a nonempty string")
        parameters = {}
        for name, specification in self.parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("parameter names must be nonempty strings")
            if not isinstance(specification, ParameterSpec):
                specification = ParameterSpec(**specification) if isinstance(specification, Mapping) else ParameterSpec(specification)
            parameters[name] = specification
        # A fixed implementation legitimately has no knobs: the empty Cartesian
        # product contains one configuration, {}, rather than zero candidates.
        for name, specification in parameters.items():
            for parent, condition in specification.active_when.items():
                if parent not in parameters or parent == name:
                    raise ValueError(f"invalid activation dependency {name!r} -> {parent!r}")
                choices = condition if isinstance(condition, (tuple, list)) else (condition,)
                domain = {canonical_json(value) for value in parameters[parent].values}
                if not choices or any(canonical_json(value) not in domain for value in choices):
                    raise ValueError(f"activation condition for {name!r} is outside {parent!r}'s domain")
        visited, active = set(), set()
        def visit(name):
            if name in active:
                raise ValueError("conditional parameter dependencies must not contain cycles")
            if name in visited:
                return
            active.add(name)
            for parent in parameters[name].active_when:
                visit(parent)
            active.remove(name)
            visited.add(name)
        for name in parameters:
            visit(name)
        if any(not callable(constraint) for constraint in self.constraints):
            raise TypeError("space constraints must be callable")
        if self.resolver is not None and not callable(self.resolver):
            raise TypeError("space resolver must be callable")
        if any(not callable(r) or not callable(getattr(r, 'check_partial', None)) or set(r.dependencies)-parameters.keys() for r in self.rules):
            raise ValueError('structured rules require declared dependencies and explicit partial checks')
        if any(not callable(callback) for callback in self.advisories):
            raise TypeError('space advisories must be callable')
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, 'rules', tuple(self.rules))
        object.__setattr__(self, 'advisories', tuple(self.advisories))
        if self.candidates is not None:
            candidates = tuple(deepcopy(c) if isinstance(c, KernelConfig) else KernelConfig(c) for c in self.candidates)
            if len({c.id for c in candidates}) != len(candidates):
                raise ValueError('native candidates must be distinct')
            if any(set(c.params) != set(parameters) or any(canonical_json(c.get(k)) not in {canonical_json(v) for v in parameters[k].values} for k in parameters) for c in candidates):
                raise ValueError('native candidates must lie in the declared parameter domains')
            object.__setattr__(self, 'candidates', candidates)

    @property
    def domains(self) -> dict[str, tuple[Any, ...]]:
        return {name: specification.values for name, specification in self.parameters.items()}

    @property
    def fingerprint(self) -> str:
        return stable_id(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name,
                "parameters": {name: specification.to_dict() for name, specification in self.parameters.items()},
                "constraints": [callable_identity(constraint) for constraint in self.constraints],
                "resolver": callable_identity(self.resolver) if self.resolver else None,
                "rules": [r.to_dict() for r in self.rules],
                "advisories": [callable_identity(c) for c in self.advisories],
                "candidates": [dict(c.params) for c in self.candidates] if self.candidates is not None else None}


@dataclass(frozen=True)
class SearchCapabilities:
    parameter_kinds: tuple[str, ...] = _PARAMETER_KINDS
    supports_constraints: bool = True
    supports_conditional: bool = True
    required_parameters: tuple[str, ...] = ()
    max_dimensions: int | None = None
    max_cardinality: int | None = None
    min_cardinality: int = 1

    def __post_init__(self) -> None:
        if not self.parameter_kinds or set(self.parameter_kinds) - set(_PARAMETER_KINDS):
            raise ValueError("algorithm parameter_kinds must contain supported finite parameter kinds")
        for value in (self.max_dimensions, self.max_cardinality,self.min_cardinality):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError("algorithm dimension/cardinality limits must be positive integers")
        if not isinstance(self.supports_constraints, bool) or not isinstance(self.supports_conditional, bool):
            raise ValueError("algorithm space capability flags must be booleans")
        object.__setattr__(self, "parameter_kinds", tuple(self.parameter_kinds))
        object.__setattr__(self, "required_parameters", tuple(self.required_parameters))

    def to_dict(self) -> dict[str, Any]:
        return {item.name: _jsonable(getattr(self, item.name)) for item in fields(self)}


@dataclass(frozen=True)
class OperatorDefinition(_Asset):
    id: str
    version: str
    shape_dims: tuple[str, ...]
    dtypes: tuple[str, ...]
    generate_inputs: Callable
    reference: Callable
    tolerances: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    validation_shapes: tuple[ShapeSpec, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    validate: Callable | None = None
    source_files: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    api_version: int = 1

    # factory(parameters: Mapping, requested_shapes: tuple[ShapeSpec, ...])
    #     -> tuple[ShapeSpec | Mapping[str, int], ...]
    validation_case_factory: Callable | None = None

    def __post_init__(self) -> None:
        _check_asset_header(self)
        if not self.shape_dims or any(not isinstance(name, str) or not name for name in self.shape_dims) or len(set(self.shape_dims)) != len(self.shape_dims):
            raise ValueError("operator shape_dims must be nonempty and distinct")
        if not self.dtypes or isinstance(self.dtypes, str) or any(not isinstance(name, str) or not name for name in self.dtypes) or len(set(self.dtypes)) != len(self.dtypes):
            raise ValueError("operator dtypes must be nonempty and distinct")
        if not callable(self.generate_inputs) or not callable(self.reference) or (self.validate is not None and not callable(self.validate)):
            raise TypeError("operator input/reference/validation entries must be callable")
        if self.validation_case_factory is not None and not callable(self.validation_case_factory):
            raise TypeError("operator validation_case_factory must be callable or None")
        tolerances = {}
        for dtype in self.dtypes:
            if dtype not in self.tolerances:
                raise ValueError(f"operator must declare tolerances for dtype {dtype!r}")
            pair = tuple(self.tolerances[dtype])
            if len(pair) != 2 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in pair):
                raise ValueError("tolerances are finite nonnegative (atol, rtol) pairs")
            tolerances[dtype] = pair
        shapes = tuple(shape if isinstance(shape, ShapeSpec) else ShapeSpec(dict(shape)) for shape in self.validation_shapes)
        if any(set(shape.dims) != set(self.shape_dims) for shape in shapes):
            raise ValueError("operator validation shapes must match shape_dims")
        canonical_json(_jsonable(self.parameters))
        canonical_json(_jsonable(self.metadata))
        object.__setattr__(self, "shape_dims", tuple(self.shape_dims))
        object.__setattr__(self, "dtypes", tuple(self.dtypes))
        object.__setattr__(self, "tolerances", MappingProxyType(tolerances))
        object.__setattr__(self, "validation_shapes", shapes)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "source_files", tuple(str(path) for path in self.source_files))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def tolerance(self, dtype: str) -> tuple[float, float]:
        if dtype not in self.tolerances:
            raise ValueError(f"operator {self.ref} has no tolerance for dtype {dtype!r}")
        return self.tolerances[dtype]

    def manifest(self) -> dict[str, Any]:
        return {"kind": "operator", "id": self.id, "version": self.version, "api_version": self.api_version,
                "shape_dims": list(self.shape_dims), "dtypes": list(self.dtypes),
                "parameters": _jsonable(self.parameters), "tolerances": _jsonable(self.tolerances),
                "metadata": _jsonable(self.metadata),
                "validation_shapes": [dict(shape.dims) for shape in self.validation_shapes],
                "generate_inputs": callable_identity(self.generate_inputs), "reference": callable_identity(self.reference),
                "validate": callable_identity(self.validate) if self.validate else None,
                "validation_case_factory": callable_identity(self.validation_case_factory) if self.validation_case_factory is not None else None,
                "source_files": _source_manifest(self.source_files, (self.generate_inputs, self.reference, self.validate,
                                                                    self.validation_case_factory))}


@dataclass(frozen=True)
class KernelDefinition(_Asset):
    id: str
    version: str
    operator: str
    build: Callable
    space: SpaceDefinition
    runtime: str = "cuda"
    dtypes: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()
    parameter_map: Mapping[str, str] = field(default_factory=dict)
    supports: Callable | None = None
    source_files: tuple[str, ...] = ()
    claims: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    api_version: int = 1
    execution_identity: Callable | None = None

    def __post_init__(self) -> None:
        _check_asset_header(self)
        if not isinstance(self.operator, str) or not self.operator or not isinstance(self.runtime, str) or not self.runtime:
            raise ValueError("Kernel operator reference and runtime are required")
        if not callable(self.build) or (self.supports is not None and not callable(self.supports)):
            raise TypeError("Kernel build/supports entries must be callable")
        if self.execution_identity is not None and not callable(self.execution_identity):
            raise TypeError('execution_identity must be a CPU-safe no-argument callback')
        if not isinstance(self.space, SpaceDefinition):
            raise TypeError("Kernel space must be a SpaceDefinition")
        if isinstance(self.dtypes, str) or isinstance(self.architectures, str) or any(not isinstance(value, str) or not value for value in (*self.dtypes, *self.architectures)):
            raise ValueError("Kernel dtypes/architectures must be sequences of nonempty strings")
        mappings = dict(self.parameter_map)
        if mappings.keys() - self.space.parameters.keys():
            raise ValueError("parameter_map contains undeclared configuration keys")
        targets = [mappings.get(name, name) for name in self.space.parameters]
        if any(not isinstance(target, str) or not target.isidentifier() or keyword.iskeyword(target) for target in targets):
            raise ValueError("parameter_map targets must be valid build keyword names")
        if len(set(targets)) != len(targets):
            raise ValueError("parameter_map must not map multiple parameters to the same build keyword")
        object.__setattr__(self, "parameter_map", MappingProxyType(mappings))
        object.__setattr__(self, "dtypes", tuple(self.dtypes))
        object.__setattr__(self, "architectures", tuple(self.architectures))
        object.__setattr__(self, "source_files", tuple(str(path) for path in self.source_files))
        object.__setattr__(self, "claims", tuple(self.claims))
        canonical_json(_jsonable(self.metadata))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def manifest(self) -> dict[str, Any]:
        # Claims bind to this identity; including claims here would be circular.
        return {"kind": "kernel", "id": self.id, "version": self.version, "api_version": self.api_version,
                "operator": self.operator, "runtime": self.runtime, "dtypes": list(self.dtypes),
                "architectures": list(self.architectures), "parameter_map": dict(self.parameter_map),
                "metadata": _jsonable(self.metadata),
                "space": self.space.to_dict(), "build": callable_identity(self.build),
                **({'execution_identity':callable_identity(self.execution_identity)} if self.execution_identity else {}),
                "supports": callable_identity(self.supports) if self.supports else None,
                "source_files": _source_manifest(self.source_files, (self.build, self.supports))}


@dataclass(frozen=True)
class SearchAlgorithmDefinition(_Asset):
    id: str
    version: str
    factory: Callable
    capabilities: SearchCapabilities = field(default_factory=SearchCapabilities)
    source_files: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    api_version: int = 1

    def __post_init__(self) -> None:
        _check_asset_header(self)
        if not callable(self.factory):
            raise TypeError("algorithm factory must be callable")
        if not isinstance(self.capabilities, SearchCapabilities):
            raise TypeError("algorithm capabilities must be SearchCapabilities")
        object.__setattr__(self, "source_files", tuple(str(path) for path in self.source_files))
        canonical_json(_jsonable(self.metadata))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def manifest(self) -> dict[str, Any]:
        return {"kind": "algorithm", "id": self.id, "version": self.version, "api_version": self.api_version,
                "capabilities": self.capabilities.to_dict(), "factory": callable_identity(self.factory),
                "metadata": _jsonable(self.metadata),
                "source_files": _source_manifest(self.source_files, (self.factory,))}


__all__ = ["ParameterSpec", "SpaceContext", "SpaceDefinition", "SearchCapabilities", "OperatorDefinition",
           "KernelDefinition", "SearchAlgorithmDefinition", "callable_identity"]
