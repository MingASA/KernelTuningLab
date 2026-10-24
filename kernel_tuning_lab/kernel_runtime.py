"""Shared operator services and complete-flow execution for registered Kernels.

Discovery imports no GPU SDK.  Both Python and CUDA use the same execution plan
for correctness, warmup and timing; CUDA timings synchronize around that plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import fnmatch
import hashlib
import inspect
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Callable

from .backends.base import BackendUnavailableError
from .claims import CorrectnessClaim, claim_matches
from .models import HardwareSpec, KernelConfig, OperatorSpec, RecordStatus, ShapeSpec
from .space import ValidationResult


def _dimensions(shape: int | Sequence[int]) -> tuple[int, ...]:
    result = (shape,) if isinstance(shape, int) else tuple(shape)
    if not result or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in result):
        raise ValueError("allocation shape must contain positive integer dimensions")
    return result


def _filled(shape: tuple[int, ...], fill: Any) -> list:
    return [fill for _ in range(shape[0])] if len(shape) == 1 else [_filled(shape[1:], fill) for _ in range(shape[0])]


@dataclass
class ExecutionContext:
    hardware: HardwareSpec
    operator: OperatorSpec
    shape: ShapeSpec
    runtime: str
    seed: int = 0
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    workspace: dict[str, Any] = field(default_factory=dict)
    config: KernelConfig | None = None
    torch: Any = None
    device: int = 0
    workspace_memory_limit_bytes: int = 256 * 1024**2
    workspace_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    _workspace_allocation_open: bool = field(default=True, repr=False)

    def _allocate(self, shape, dtype, fill):
        shape = _dimensions(shape)
        dtype = dtype or self.operator.dtype
        if self.runtime == "cuda":
            if self.torch is None:
                raise BackendUnavailableError("CUDA context has no loaded PyTorch runtime")
            torch_dtype = getattr(self.torch, dtype) if isinstance(dtype, str) else dtype
            value = self.torch.empty(shape, dtype=torch_dtype, device=f"cuda:{self.device}")
            if fill is not None:
                value.fill_(fill)
            return value
        if self.runtime != "python":
            raise ValueError(f"unsupported execution runtime: {self.runtime}")
        return _filled(shape, math.nan if fill is None else fill)

    def allocate_workspace(self, name: str, shape: int | Sequence[int], dtype: str | None = None,
                           fill: Any = 0) -> Any:
        """Declare an allocation once, outside repeated execution and timing."""
        if not self._workspace_allocation_open:
            raise ValueError("allocate workspace during Kernel build/setup, before repeated invocation")
        if not isinstance(name, str) or not name or name in self.workspace:
            raise ValueError("workspace names must be nonempty and allocated exactly once")
        dims = _dimensions(shape)
        widths = {"bool": 1, "uint8": 1, "int8": 1, "float16": 2, "bfloat16": 2,
                  "int16": 2, "float32": 4, "int32": 4, "float64": 8, "int64": 8}
        selected = dtype or self.operator.dtype
        if selected not in widths:
            raise ValueError(f"unsupported workspace dtype {selected!r}")
        # Python lists and boxed numbers cost more than their logical dtype.
        width = widths[selected] if self.runtime == "cuda" else max(32, widths[selected])
        size = math.prod(dims) * width
        if self.runtime == "python":
            size += 64 * sum(math.prod(dims[:depth]) for depth in range(len(dims)))
        if size > self.workspace_memory_limit_bytes - self.workspace_bytes:
            raise ValueError("declared workspace exceeds workspace_memory_limit_bytes")
        value = self._allocate(dims, selected, fill)
        self.workspace[name] = value
        self.workspace_bytes += size
        self.metadata.setdefault("workspace_allocations", []).append({"name": name, "shape": dims, "dtype": selected, "accounted_bytes": size})
        return value

    def allocate_output(self, name: str, shape: int | Sequence[int], dtype: str | None = None,
                        fill: Any = None) -> Any:
        if not isinstance(name, str) or not name or name in self.outputs:
            raise ValueError("output names must be nonempty and allocated exactly once")
        value = self._allocate(shape, dtype, fill)
        self.outputs[name] = value
        return value

    def record_compiled_kernel(self, name: str, compiled) -> None:
        """Archive actual compiler resources and reject known launch-limit violations."""
        if not isinstance(name, str) or not name:
            raise ValueError("compiled kernel resource entry needs a name")
        metadata = getattr(compiled, "metadata", None)
        shared = getattr(metadata, "shared", None)
        registers = getattr(compiled, "n_regs", None)
        warps = getattr(metadata, "num_warps", None)
        self.metadata.setdefault("execution_resources", {})[name] = {
            "shared_memory_bytes": shared, "registers_per_thread": registers, "num_warps": warps}
        if shared is not None and self.hardware.smem_per_block and shared > self.hardware.smem_per_block:
            raise RuntimeError(f"{name} compiled shared memory {shared} exceeds {self.hardware.smem_per_block} bytes")
        if warps is not None and warps * self.hardware.warp_size > self.hardware.max_threads_per_block:
            raise RuntimeError(f"{name} compiled warp count exceeds physical thread limit")
        if (registers is not None and warps is not None and self.hardware.registers_per_sm
                and registers * warps * self.hardware.warp_size > self.hardware.registers_per_sm):
            raise RuntimeError(f"{name} compiled registers exceed the complete SM register file")


@dataclass(frozen=True)
class ExecutionStep:
    name: str
    launch: Callable[[], Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not callable(self.launch):
            raise ValueError("execution steps need a name and a callable launch")
        inspect.signature(self.launch).bind()


@dataclass(frozen=True)
class ExecutionPlan:
    steps: tuple[ExecutionStep, ...]
    initialize: Callable[[], Any] | None = None
    setup: Callable[[], Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.steps or any(not isinstance(step, ExecutionStep) for step in self.steps):
            raise ValueError("execution plan needs at least one ExecutionStep")
        if len({step.name for step in self.steps}) != len(self.steps):
            raise ValueError("execution step names must be unique")
        for name in ("initialize", "setup"):
            callback = getattr(self, name)
            if callback is not None:
                if not callable(callback):
                    raise ValueError(f"execution plan {name} must be callable")
                inspect.signature(callback).bind()


@dataclass
class PreparedAsset:
    context: ExecutionContext
    reference: dict[str, Any]


@dataclass
class CompiledAsset:
    prepared: PreparedAsset
    plan: ExecutionPlan
    build_time_ms: float
    setup_time_ms: float

    @property
    def context(self) -> ExecutionContext:
        return self.prepared.context


def _validation(value: Any) -> ValidationResult:
    if isinstance(value, ValidationResult):
        return value
    if value is None or value is True:
        return ValidationResult(True)
    if value is False:
        return ValidationResult(False, ("asset capability validator rejected this input",))
    if isinstance(value, str):
        return ValidationResult(not value, (value,) if value else ())
    raise TypeError("capability validators must return ValidationResult, a reason string, bool or None")


def _copy_output(value: Any, torch_module: Any = None) -> Any:
    if torch_module is not None and isinstance(value, torch_module.Tensor):
        return value.detach().clone()
    return deepcopy(value)


def _flatten(value: Any):
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_flatten(item) for item in value))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Python operator outputs must be nested numeric lists or scalar numbers")
    return float(value)


def _compare_python(actual: Any, expected: Any, atol: float, rtol: float) -> tuple[bool, float, bool]:
    left, right = _flatten(actual), _flatten(expected)
    worst = 0.0
    nonfinite = False

    def compare(a, b):
        nonlocal worst, nonfinite
        if isinstance(a, tuple) or isinstance(b, tuple):
            return (isinstance(a, tuple) and isinstance(b, tuple) and len(a[1]) == len(b[1])
                    and all([compare(x, y) for x, y in zip(a[1], b[1])]))
        if not math.isfinite(a) or not math.isfinite(b):
            nonfinite = True
            return False
        difference = abs(a - b)
        worst = max(worst, difference)
        return difference <= atol + rtol * abs(b)

    matched = bool(compare(left, right))
    return matched, worst if math.isfinite(worst) else None, nonfinite


class AssetKernelAdapter:
    """Bridge a Kernel's build function to the existing benchmark lifecycle."""

    provenance = "measured"

    def __init__(self, kernel, operator_definition, *, accept_claims: bool = True,
                 force_correctness: bool = False, device: int = 0,
                 workspace_memory_limit_bytes: int = 256 * 1024**2):
        self.kernel, self.operator_definition = kernel, operator_definition
        self.name = kernel.ref
        self.accept_claims, self.force_correctness = accept_claims, force_correctness
        self.device = device
        self.workspace_memory_limit_bytes = workspace_memory_limit_bytes
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise ValueError("CUDA device must be a nonnegative integer")
        if isinstance(workspace_memory_limit_bytes, bool) or not isinstance(workspace_memory_limit_bytes, int) or workspace_memory_limit_bytes <= 0:
            raise ValueError("workspace_memory_limit_bytes must be positive")
        if kernel.runtime not in {"python", "cuda"}:
            raise ValueError(f"unsupported Kernel runtime {kernel.runtime!r}")
        self.measurement_mode = "cpu_full_flow_perf_counter" if kernel.runtime == "python" else "cuda_synchronized_full_flow_perf_counter"
        self._torch = None
        self._triton = None
        self._hardware = None
        self._driver_probed = False
        self._driver_version = None
        self._failed_cases: set[tuple[str, str, str]] = set()
        self._revoked_claims: dict[str, str] = {}
        self._kernel_identity = kernel.fingerprint
        self._operator_identity = operator_definition.fingerprint

    def _load_cuda(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except ImportError as exc:
            raise BackendUnavailableError("CUDA asset execution requires an installed CUDA-enabled PyTorch build") from exc
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise BackendUnavailableError("CUDA asset execution requires an available NVIDIA CUDA GPU")
        if self.device >= torch.cuda.device_count():
            raise BackendUnavailableError(f"CUDA device index {self.device} is unavailable")
        if self.kernel.metadata.get("requires_triton", False):
            try:
                import triton
            except ImportError as exc:
                raise BackendUnavailableError("this Kernel requires the optional Triton runtime") from exc
            from .backends.compatibility import validate_triton_runtime
            major, minor = torch.cuda.get_device_capability(self.device)
            result = validate_triton_runtime(f"sm_{major}{minor}", torch_version=torch.__version__,
                                            cuda_version=torch.version.cuda, triton_version=triton.__version__)
            if not result.valid:
                raise BackendUnavailableError("; ".join(result.reasons))
            self._triton = triton
        torch.cuda.set_device(self.device)
        self._torch = torch
        return torch

    def probe_hardware(self) -> HardwareSpec:
        if self._hardware is None:
            if self.kernel.runtime == "python":
                self._hardware = HardwareSpec(name=f"CPU {platform.processor() or platform.machine() or 'unknown'}",
                                              architecture="cpu", warp_size=1, max_threads_per_block=1,
                                              features=("python",), metadata={"source": "platform/os", "logical_cpus": os.cpu_count(),
                                                                                 "platform": platform.platform(), "physical_runtime": "python"})
            else:
                torch = self._load_cuda()
                # Reuse the established physical-device probe without requiring
                # all CUDA asset implementations to depend on Triton.
                from .backends.triton_gemm import TritonGemmAdapter
                probe = object.__new__(TritonGemmAdapter)
                probe.torch, probe.device = torch, self.device
                self._hardware = probe.probe_hardware()
        return self._hardware

    def environment(self) -> dict[str, Any]:
        if self.kernel.fingerprint != self._kernel_identity or self.operator_definition.fingerprint != self._operator_identity:
            raise ValueError("asset source/declaration changed after resolution; restart acceptance with the new content")
        actual = self.probe_hardware()
        result = {"python": platform.python_version(), "platform": platform.platform(), "runtime": self.kernel.runtime,
                  "physical_device": actual.name, "architecture": actual.architecture,
                  "kernel_ref": self.kernel.ref, "kernel_fingerprint": self.kernel.fingerprint,
                  "operator_ref": self.operator_definition.ref, "operator_fingerprint": self.operator_definition.fingerprint,
                  "runtime_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if self.kernel.runtime == "cuda":
            torch = self._load_cuda()
            if not self._driver_probed:
                self._driver_probed = True
                try:
                    probe = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=5, check=False)
                    versions = set(probe.stdout.strip().splitlines()) if probe.returncode == 0 else set()
                    if len(versions) == 1:
                        self._driver_version = next(iter(versions))
                except (OSError, subprocess.TimeoutExpired):
                    pass
            result.update({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device_index": self.device,
                           "nvidia_driver": self._driver_version,
                           "gpu": actual.name, "torch_compiled_arches": list(torch.cuda.get_arch_list())})
            if self._triton is not None:
                result["triton"] = self._triton.__version__
        return result

    def protocol_metadata(self) -> dict[str, Any]:
        return {"contract": "asset_kernel_v1", "kernel_ref": self.kernel.ref,
                "kernel_fingerprint": self.kernel.fingerprint, "operator_ref": self.operator_definition.ref,
                "operator_fingerprint": self.operator_definition.fingerprint,
                "measurement_scope": "initialize followed by all ordered execution steps",
                "timing": self.measurement_mode, "excluded_from_timing": ["input_generation", "reference", "build", "one_time_setup", "workspace_allocation"],
                "claim_policy": {"accept_claims": self.accept_claims, "force_correctness": self.force_correctness,
                                 "claims": [getattr(claim, "fingerprint", None) or claim for claim in self.kernel.claims],
                                 "counterexamples": sorted(self._revoked_claims.items())},
                "tolerances": dict(self.operator_definition.tolerances),
                "workspace_memory_limit_bytes": self.workspace_memory_limit_bytes}

    def _context(self, hardware, operator, shape, seed=0):
        # Contributor code may mutate its context, but must not mutate the case
        # identities retained by the runner and its append-only result journal.
        return ExecutionContext(deepcopy(hardware), deepcopy(operator), deepcopy(shape), self.kernel.runtime, seed=seed,
                                torch=self._load_cuda() if self.kernel.runtime == "cuda" else None,
                                device=self.device, workspace_memory_limit_bytes=self.workspace_memory_limit_bytes)

    def _mapped_parameters(self, config: KernelConfig) -> dict[str, Any]:
        mapping = dict(self.kernel.parameter_map)
        unknown = mapping.keys() - config.params.keys()
        if unknown:
            # Conditional inactive parameters need not occur in this config.
            declared = getattr(self.kernel.space, "parameters", {})
            unexpected = unknown - declared.keys()
            if unexpected:
                raise ValueError(f"parameter_map refers to unknown parameters: {sorted(unexpected)}")
        result = {}
        for key, value in config.params.items():
            target = mapping.get(key, key)
            if not isinstance(target, str) or not target:
                raise ValueError("parameter_map targets must be nonempty keyword names")
            if target in result:
                raise ValueError(f"multiple parameters map to {target!r}")
            result[target] = value
        return result

    def validate(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec,
                 config: KernelConfig) -> ValidationResult:
        reasons = []
        actual = self.probe_hardware()
        if (hardware.name, hardware.architecture) != (actual.name, actual.architecture):
            reasons.append("requested hardware does not match the physical execution device")
        if self.kernel.operator not in {self.operator_definition.ref, self.operator_definition.id}:
            reasons.append("Kernel operator reference differs from the resolved operator asset")
        if operator.kind != self.operator_definition.id or operator.version != self.operator_definition.version:
            reasons.append("OperatorSpec kind/version differs from the resolved operator asset")
        if operator.implementation not in {self.kernel.id, self.kernel.ref}:
            reasons.append("OperatorSpec implementation differs from the selected Kernel")
        if set(shape.dims) != set(self.operator_definition.shape_dims):
            reasons.append(f"operator requires shape dimensions {self.operator_definition.shape_dims}")
        if operator.dtype not in self.operator_definition.dtypes or (self.kernel.dtypes and operator.dtype not in self.kernel.dtypes):
            reasons.append(f"dtype {operator.dtype!r} is unsupported by operator or Kernel")
        if self.kernel.architectures and not any(fnmatch.fnmatchcase(hardware.architecture, pattern) for pattern in self.kernel.architectures):
            reasons.append(f"architecture {hardware.architecture!r} is unsupported by Kernel")
        if self.kernel.runtime == "python" and hardware.architecture != "cpu":
            reasons.append("Python execution requires the physical CPU hardware spec")
        unknown = operator.parameters.keys() - self.operator_definition.parameters.keys()
        if unknown:
            reasons.append(f"unknown operator parameters: {sorted(unknown)}")
        try:
            context = self._context(hardware, operator, shape)
            context.config = deepcopy(config)
            callback = self.operator_definition.validate
            if callback is not None:
                result = _validation(callback(context))
                reasons.extend(result.reasons or (() if result.valid else ("operator validator rejected input",)))
            supports = self.kernel.supports
            if supports is not None:
                result = _validation(supports(hardware, operator, shape))
                reasons.extend(result.reasons or (() if result.valid else ("Kernel validator rejected input",)))
            inspect.signature(self.kernel.build).bind(context, **self._mapped_parameters(config))
        except Exception as exc:
            reasons.append(f"Kernel contract: {type(exc).__name__}: {exc}")
        return ValidationResult(not reasons, tuple(reasons))

    def prepare(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec, seed: int) -> PreparedAsset:
        context = self._context(hardware, operator, shape, seed)
        generated = self.operator_definition.generate_inputs(context, seed)
        if generated is not None:
            if not isinstance(generated, Mapping):
                raise TypeError("generate_inputs must return an input mapping or populate ctx.inputs and return None")
            context.inputs.update(generated)
        expected = self.operator_definition.reference(context)
        if not isinstance(expected, Mapping):
            expected = {"out": expected}
        if not expected or any(not isinstance(key, str) or not key for key in expected):
            raise ValueError("reference must define at least one named output")
        baseline = {key: _copy_output(value, context.torch) for key, value in expected.items()}
        self._synchronize()
        return PreparedAsset(context, baseline)

    def compile(self, prepared: PreparedAsset, config: KernelConfig) -> CompiledAsset:
        prepared.context.config = deepcopy(config)
        started = time.perf_counter()
        built = self.kernel.build(prepared.context, **self._mapped_parameters(config))
        if isinstance(built, ExecutionPlan):
            plan = built
        elif callable(built):
            plan = ExecutionPlan((ExecutionStep("kernel", built),))
        else:
            raise TypeError("Kernel build must return a callable or ExecutionPlan")
        self._synchronize()
        build_ms = (time.perf_counter() - started) * 1000
        setup_started = time.perf_counter()
        if plan.setup is not None:
            plan.setup()
        self._synchronize()
        setup_ms = (time.perf_counter() - setup_started) * 1000
        prepared.context._workspace_allocation_open = False
        return CompiledAsset(prepared, plan, build_ms, setup_ms)

    def _synchronize(self):
        if self.kernel.runtime == "cuda":
            self._load_cuda().cuda.synchronize(self.device)

    def _invoke(self, compiled: CompiledAsset):
        if self.kernel.runtime == "cuda":
            self._load_cuda().cuda.set_device(self.device)
        if compiled.plan.initialize is not None:
            compiled.plan.initialize()
        for step in compiled.plan.steps:
            result = step.launch()
            if result is not None:
                if isinstance(result, Mapping):
                    compiled.context.outputs.update(result)
                elif (isinstance(result, (int, float, list, tuple)) or
                      (compiled.context.torch is not None and isinstance(result, compiled.context.torch.Tensor))):
                    compiled.context.outputs["out"] = result
                # JIT launches may return a compiler handle while writing their
                # declared output buffers. That handle is not a tensor result.

    def _matching_claim(self, compiled: CompiledAsset):
        ctx = compiled.context
        blocked = (ctx.operator.id, ctx.shape.id, ctx.config.id) in self._failed_cases
        evaluated = []
        matched_claim = None
        for raw in self.kernel.claims:
            try:
                claim = raw if isinstance(raw, CorrectnessClaim) else CorrectnessClaim.from_dict(raw)
            except (TypeError, ValueError) as exc:
                evaluated.append({"id": raw.get("id") if isinstance(raw, Mapping) else None,
                                  "valid": False, "reasons": [f"invalid correctness claim: {exc}"]})
                continue
            result = claim_matches(claim, self.kernel, self.operator_definition, ctx, ctx.config, self.environment())
            reasons = list(result.reasons)
            if blocked:
                reasons.append("a direct correctness failure overrides submitter assertions for this case/configuration")
            if self.force_correctness:
                reasons.append("force_correctness requests an independent platform comparison")
            if not self.accept_claims:
                reasons.append("submitter assertions are disabled by the experiment policy")
            revoked = self._revoked_claims.get(claim.fingerprint)
            if revoked:
                reasons.append(f"recorded platform counterexample overrides this declaration: {revoked}")
            usable = result.valid and not blocked and not revoked and self.accept_claims and not self.force_correctness
            evaluated.append({"id": claim.id, "fingerprint": claim.fingerprint, "valid": result.valid,
                              "accepted": usable, "reasons": reasons})
            if usable and matched_claim is None:
                matched_claim = claim
        return matched_claim, evaluated

    def remember_counterexample(self, record) -> list[str]:
        """Apply persisted, matching reference failures before trusting assertions."""
        if (record.status != RecordStatus.CORRECTNESS_ERROR or record.provenance != "measured"
                or record.environment.get("kernel_fingerprint") != self._kernel_identity
                or record.environment.get("operator_fingerprint") != self._operator_identity
                or record.metadata.get("correctness", {}).get("correctness_origin") not in {"platform_reference", "platform_structure"}):
            return []
        context = ExecutionContext(record.hardware, record.operator, record.shape, self.kernel.runtime, seed=record.seed)
        revoked = []
        for raw in self.kernel.claims:
            try:
                claim = raw if isinstance(raw, CorrectnessClaim) else CorrectnessClaim.from_dict(raw)
                matched = claim_matches(claim, self.kernel, self.operator_definition, context, record.config, record.environment)
            except (TypeError, ValueError):
                continue
            if matched.valid:
                self._revoked_claims[claim.fingerprint] = record.id
                revoked.append(claim.id)
        return revoked

    @staticmethod
    def _poison_outputs(ctx: ExecutionContext):
        """Expose missing writes without changing output object identities."""
        def poison(value):
            if isinstance(value, list):
                for index, child in enumerate(value):
                    if isinstance(child, list):
                        poison(child)
                    elif isinstance(child, (int, float)) and not isinstance(child, bool):
                        value[index] = math.nan
            elif ctx.torch is not None and isinstance(value, ctx.torch.Tensor) and value.is_floating_point():
                value.fill_(math.nan)
        for output in ctx.outputs.values():
            poison(output)

    def _compare_outputs(self, compiled: CompiledAsset, atol: float, rtol: float):
        ctx = compiled.context
        expected = compiled.prepared.reference
        correct = ctx.outputs.keys() == expected.keys()
        errors = {}
        for name, reference in expected.items():
            if name not in ctx.outputs:
                errors[name] = {"reason": "expected output is missing"}
                continue
            actual = ctx.outputs[name]
            if self.kernel.runtime == "cuda":
                torch = self._load_cuda()
                if not isinstance(actual, torch.Tensor) or not isinstance(reference, torch.Tensor):
                    matched, maximum, nonfinite = False, None, False
                elif actual.shape != reference.shape or actual.device != reference.device or actual.dtype != reference.dtype:
                    matched, maximum, nonfinite = False, None, False
                else:
                    finite = bool(torch.isfinite(actual).all().item()) and bool(torch.isfinite(reference).all().item())
                    matched = finite and bool(torch.allclose(actual, reference, atol=atol, rtol=rtol))
                    maximum = float((actual.double() - reference.double()).abs().max().item()) if actual.numel() else 0.0
                    nonfinite = not finite
                    if not math.isfinite(maximum):
                        maximum = None
            else:
                matched, maximum, nonfinite = _compare_python(actual, reference, atol, rtol)
            correct = correct and matched
            errors[name] = {"matches": matched, "max_absolute_error": maximum, "nonfinite_output": nonfinite}
        if ctx.outputs.keys() - expected.keys():
            errors["__unexpected_outputs__"] = {"names": sorted(ctx.outputs.keys() - expected.keys())}
        return bool(correct), errors

    def _structural_outputs(self, compiled: CompiledAsset):
        """Claims do not excuse missing buffers, wrong shapes or NaN outputs."""
        ctx, expected = compiled.context, compiled.prepared.reference
        correct = ctx.outputs.keys() == expected.keys()
        errors = {}

        def python_structure(value):
            if isinstance(value, (list, tuple)):
                return tuple(python_structure(item) for item in value)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("output contains a nonnumeric or nonfinite scalar")
            return None

        for name, baseline in expected.items():
            if name not in ctx.outputs:
                errors[name] = "expected output is missing"
                correct = False
                continue
            actual = ctx.outputs[name]
            if ctx.runtime == "cuda":
                torch = self._load_cuda()
                valid = (isinstance(actual, torch.Tensor) and isinstance(baseline, torch.Tensor)
                         and actual.shape == baseline.shape and actual.dtype == baseline.dtype
                         and actual.device == baseline.device and bool(torch.isfinite(actual).all().item()))
                if not valid:
                    errors[name] = "output tensor shape, dtype, device or finite-value check failed"
            else:
                try:
                    valid = python_structure(actual) == python_structure(baseline)
                    if not valid:
                        errors[name] = "output shape differs from operator output shape"
                except ValueError as exc:
                    valid = False
                    errors[name] = str(exc)
            correct = correct and valid
        if ctx.outputs.keys() - expected.keys():
            errors["__unexpected_outputs__"] = sorted(ctx.outputs.keys() - expected.keys())
        return bool(correct), errors

    def check_correctness(self, compiled: CompiledAsset) -> tuple[bool, dict[str, Any]]:
        ctx = compiled.context
        claim, evaluated_claims = self._matching_claim(compiled)
        execution = {"steps": [step.name for step in compiled.plan.steps], "initialize_each_invocation": compiled.plan.initialize is not None,
                     "setup_once": compiled.plan.setup is not None, "setup_time_ms": compiled.setup_time_ms,
                     "build_time_ms": compiled.build_time_ms, "workspace_bytes": ctx.workspace_bytes,
                     "workspace_allocations": ctx.metadata.get("workspace_allocations", []),
                     "compiled_resources": ctx.metadata.get("execution_resources", {})}
        if claim is not None:
            self._poison_outputs(ctx)
            self._invoke(compiled)
            self._synchronize()
            structurally_valid, structural_errors = self._structural_outputs(compiled)
            if not structurally_valid:
                self._failed_cases.add((ctx.operator.id, ctx.shape.id, ctx.config.id))
                self._revoked_claims[claim.fingerprint] = f"output structure failure for {ctx.shape.id}/{ctx.config.id}"
                for evaluated in evaluated_claims:
                    if evaluated.get("accepted"):
                        evaluated["accepted"] = False
                        evaluated["reasons"].append("direct output structure/finite-value failure overrides this claim")
                return False, {"status": "failed", "correctness_origin": "platform_structure",
                               "verified_on_gpu": False, "verified_on_cpu": False,
                               "comparison_performed": False, "outputs": structural_errors,
                               "claims_evaluated": evaluated_claims, "execution": execution}
            return True, {"status": "declared", "correctness_origin": "submitter_claim", "claim_id": claim.id,
                          "claim_fingerprint": claim.fingerprint, "claim_source": claim.source,
                          "claim_evidence": list(claim.evidence), "verified_on_gpu": False,
                          "verified_on_cpu": False, "comparison_performed": False,
                          "structural_check": "passed",
                          "claims_evaluated": evaluated_claims, "execution": execution}
        atol, rtol = self.operator_definition.tolerance(ctx.operator.dtype)
        correct = True
        runs = []
        for index in range(2):
            self._poison_outputs(ctx)
            self._invoke(compiled)
            self._synchronize()
            run_correct, errors = self._compare_outputs(compiled, atol, rtol)
            runs.append({"invocation": index + 1, "correct": run_correct, "outputs": errors})
            correct = correct and run_correct
        if not correct:
            self._failed_cases.add((ctx.operator.id, ctx.shape.id, ctx.config.id))
            for evaluated in evaluated_claims:
                if evaluated.get("valid") and evaluated.get("fingerprint"):
                    self._revoked_claims[evaluated["fingerprint"]] = f"reference failure for {ctx.shape.id}/{ctx.config.id}"
        return bool(correct), {"status": "verified" if correct else "failed", "correctness_origin": "platform_reference",
                               "verified_on_gpu": bool(correct and self.kernel.runtime == "cuda"),
                               "verified_on_cpu": bool(correct and self.kernel.runtime == "python"),
                               "comparison_performed": True, "atol": atol, "rtol": rtol,
                               "reference_fingerprint": self.operator_definition.fingerprint,
                               "outputs": runs[-1]["outputs"], "checked_invocations": runs,
                               "claims_evaluated": evaluated_claims, "execution": execution}

    def warmup(self, compiled: CompiledAsset, iterations: int):
        for _ in range(iterations):
            self._invoke(compiled)
        self._synchronize()

    def measure(self, compiled: CompiledAsset, repetitions: int) -> tuple[float, ...]:
        samples = []
        for _ in range(repetitions):
            self._synchronize()
            started = time.perf_counter()
            self._invoke(compiled)
            self._synchronize()
            samples.append((time.perf_counter() - started) * 1000)
        return tuple(samples)


__all__ = ["ExecutionContext", "ExecutionStep", "ExecutionPlan", "AssetKernelAdapter", "PreparedAsset", "CompiledAsset"]
