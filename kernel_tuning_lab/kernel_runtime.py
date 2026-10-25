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
import struct
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
    input_fingerprints: dict[str, str] = field(default_factory=dict)


@dataclass
class CompiledAsset:
    prepared: PreparedAsset
    plan: ExecutionPlan
    build_time_ms: float
    setup_time_ms: float
    correctness_details: dict[str, Any] = field(default_factory=dict)
    cuda_graph: Any = None

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


def _compare_python(actual: Any, expected: Any, atol: float, rtol: float) -> tuple[bool, float, bool]:
    worst = 0.0
    nonfinite = False

    def compare(a, b):
        nonlocal worst, nonfinite
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != len(b):
                return False
            matched = True
            for left, right in zip(a, b):
                matched = compare(left, right) and matched
            return matched
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (a, b)):
            raise TypeError("Python operator outputs must be nested numeric lists or scalar numbers")
        if not math.isfinite(a) or not math.isfinite(b):
            nonfinite = True
            return False
        difference = abs(a - b)
        worst = max(worst, difference)
        return difference <= atol + rtol * abs(b)

    matched = bool(compare(actual, expected))
    return matched, worst if math.isfinite(worst) else None, nonfinite


def _tensor_chunks(value, max_elements: int):
    """Visit logical tensor contents without materializing a large strided copy."""
    if value.numel() <= max_elements:
        yield value
    elif value.is_contiguous():
        flat = value.view(-1)
        for start in range(0, flat.numel(), max_elements):
            yield flat[start:start + max_elements]
    else:
        axis = next(index for index, size in enumerate(value.shape) if size > 1)
        width = max(1, max_elements // max(1, value.numel() // value.shape[axis]))
        for start in range(0, value.shape[axis], width):
            selection = [slice(None)] * value.ndim
            selection[axis] = slice(start, start + width)
            yield from _tensor_chunks(value[tuple(selection)], max_elements)


def _paired_tensor_chunks(left, right, max_elements: int):
    if left.shape != right.shape:
        raise ValueError("tensor output shape differs from its reference")
    if left.numel() <= max_elements:
        yield left, right
    elif left.is_contiguous() and right.is_contiguous():
        left, right = left.view(-1), right.view(-1)
        for start in range(0, left.numel(), max_elements):
            yield left[start:start + max_elements], right[start:start + max_elements]
    else:
        axis = next(index for index, size in enumerate(left.shape) if size > 1)
        width = max(1, max_elements // max(1, left.numel() // left.shape[axis]))
        for start in range(0, left.shape[axis], width):
            selection = [slice(None)] * left.ndim
            selection[axis] = slice(start, start + width)
            yield from _paired_tensor_chunks(left[tuple(selection)], right[tuple(selection)], max_elements)


def _input_fingerprints(ctx: ExecutionContext) -> dict[str, str]:
    """Hash every logical input value; GPU host staging stays below one MiB.

    This is an execution-boundary check, outside the timed flow. It intentionally
    reads every input element rather than presenting a sample as full integrity.
    """
    def fingerprint(value):
        digest = hashlib.sha256()

        def visit(item):
            if ctx.torch is not None and isinstance(item, ctx.torch.Tensor):
                if item.layout != ctx.torch.strided:
                    raise TypeError("input integrity currently requires strided tensors")
                digest.update(repr(("tensor", tuple(item.shape), tuple(item.stride()), str(item.dtype), str(item.device))).encode())
                maximum = max(1, 1024**2 // item.element_size())
                for chunk in _tensor_chunks(item.detach(), maximum):
                    raw = chunk.contiguous().reshape(-1).view(ctx.torch.uint8).cpu()
                    try:
                        payload = raw.numpy().tobytes()
                    except (ImportError, RuntimeError):
                        payload = bytes(raw.tolist())
                    digest.update(payload)
            elif isinstance(item, Mapping):
                digest.update(b"mapping:")
                for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                    visit(key)
                    visit(item[key])
                digest.update(b";")
            elif isinstance(item, (list, tuple)):
                digest.update((type(item).__name__ + f":{len(item)}:").encode())
                for child in item:
                    visit(child)
                digest.update(b";")
            elif isinstance(item, float):
                digest.update(b"float:" + struct.pack("!d", item))
            elif item is None or isinstance(item, (bool, int, str)):
                digest.update((type(item).__name__ + ":" + repr(item) + ";").encode())
            elif isinstance(item, (bytes, bytearray, memoryview)):
                digest.update((type(item).__name__ + f":{len(item)}:").encode())
                digest.update(item)
            else:
                raise TypeError(f"input integrity does not support {type(item).__name__}; use nested Python values or strided tensors")
        visit(value)
        return digest.hexdigest()

    if any(not isinstance(name, str) or not name for name in ctx.inputs):
        raise ValueError("input names must be nonempty strings")
    return {name: fingerprint(value) for name, value in ctx.inputs.items()}


class AssetKernelAdapter:
    """Bridge a Kernel's build function to the existing benchmark lifecycle."""

    provenance = "measured"

    def __init__(self, kernel, operator_definition, *, accept_claims: bool = True,
                 force_correctness: bool = False, device: int = 0,
                 workspace_memory_limit_bytes: int = 256 * 1024**2,
                 timing_mode: str = "full_flow", cuda_graph_batch_size: int = 128):
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
        if timing_mode not in {"full_flow", "cuda_graph"}:
            raise ValueError("timing_mode must be full_flow or cuda_graph")
        if timing_mode == "cuda_graph" and kernel.runtime != "cuda":
            raise ValueError("cuda_graph timing requires a CUDA Kernel")
        if type(cuda_graph_batch_size) is not int or not 1 <= cuda_graph_batch_size <= 4096:
            raise ValueError("cuda_graph_batch_size must be an integer in [1, 4096]")
        self.timing_mode, self.cuda_graph_batch_size = timing_mode, cuda_graph_batch_size
        self.measurement_mode = "cpu_full_flow_perf_counter" if kernel.runtime == "python" else "cuda_synchronized_full_flow_perf_counter"
        if timing_mode == "cuda_graph":
            self.measurement_mode = "cuda_graph_events_full_flow_amortized"
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
                "input_contract": {"mutation": "forbidden", "integrity": "full_logical_sha256_at_execution_boundaries",
                                   "output_input_aliasing": "forbidden", "checks_excluded_from_timing": True},
                "cuda_graph": ({"calls_per_replay": self.cuda_graph_batch_size, "host_steps_replayed": False,
                                "scope": "all captured GPU initialize and ordered steps; reused buffers; hot cache",
                                "setup_excluded": True} if self.timing_mode == "cuda_graph" else None),
                "output_poisoning": {"before_each_invocation": self.timing_mode != "cuda_graph",
                                     "before_each_graph_replay": self.timing_mode == "cuda_graph", "timed": False,
                                     "floating": "nan", "boolean_integer": "elementwise_reference_distinct_sentinel"},
                "post_measurement_check": "final output and input integrity, without another invocation",
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
        inputs = _input_fingerprints(context)
        expected = self.operator_definition.reference(context)
        if not isinstance(expected, Mapping):
            expected = {"out": expected}
        if not expected or any(not isinstance(key, str) or not key for key in expected):
            raise ValueError("reference must define at least one named output")
        baseline = {key: _copy_output(value, context.torch) for key, value in expected.items()}
        if _input_fingerprints(context) != inputs:
            raise ValueError("operator reference changed read-only inputs")
        self._synchronize()
        return PreparedAsset(context, baseline, inputs)

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
        compiled = CompiledAsset(prepared, plan, build_ms, setup_ms)
        if self.timing_mode == "cuda_graph":
            torch = self._load_cuda()
            graph_started = time.perf_counter()
            # Warm lazy launch machinery on a side stream before capture.
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                self._invoke(compiled)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(self.cuda_graph_batch_size):
                    self._invoke(compiled)
            self._synchronize()
            compiled.cuda_graph = graph
            prepared.context.metadata["cuda_graph_setup_time_ms"] = (time.perf_counter() - graph_started) * 1000
        return compiled

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
                or record.metadata.get("correctness", {}).get("correctness_origin") not in {
                    "platform_reference", "platform_structure", "platform_input_integrity"}):
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
    def _output_input_aliases(ctx: ExecutionContext) -> list[str]:
        """Outputs may not share mutable containers or tensor storage with inputs."""
        if not isinstance(ctx.inputs, Mapping) or not isinstance(ctx.outputs, Mapping):
            return ["__invalid_input_or_output_mapping__"]

        def identities(value):
            if ctx.torch is not None and isinstance(value, ctx.torch.Tensor):
                if value.numel():
                    yield ("tensor", str(value.device), value.untyped_storage().data_ptr())
            elif isinstance(value, (Mapping, list, tuple)):
                if not isinstance(value, tuple):
                    yield ("python", id(value))
                children = value.values() if isinstance(value, Mapping) else value
                for child in children:
                    yield from identities(child)
            elif isinstance(value, (bytearray, memoryview)):
                yield ("python", id(value))

        inputs = {identity for value in ctx.inputs.values() for identity in identities(value)}
        return [name for name, value in ctx.outputs.items() if any(identity in inputs for identity in identities(value))]

    def _input_integrity(self, compiled: CompiledAsset):
        try:
            actual = _input_fingerprints(compiled.context)
            expected = compiled.prepared.input_fingerprints
            changed = sorted(name for name in actual.keys() | expected.keys() if actual.get(name) != expected.get(name))
            return not changed, {"status": "passed" if not changed else "failed", "method": "sha256",
                                 "scope": "all logical input values at execution boundaries",
                                 "changed_inputs": changed, "expected": expected, "actual": actual}
        except Exception as exc:
            return False, {"status": "failed", "method": "sha256",
                           "reason": f"{type(exc).__name__}: {exc}"}

    def _remember_failure(self, compiled: CompiledAsset, evaluated_claims):
        ctx = compiled.context
        self._failed_cases.add((ctx.operator.id, ctx.shape.id, ctx.config.id))
        for evaluated in evaluated_claims:
            if evaluated.get("valid") and evaluated.get("fingerprint"):
                self._revoked_claims[evaluated["fingerprint"]] = f"platform failure for {ctx.shape.id}/{ctx.config.id}"
            if evaluated.get("accepted"):
                evaluated["accepted"] = False
                evaluated["reasons"].append("direct platform correctness failure overrides this declaration")

    @staticmethod
    def _poison_outputs(ctx: ExecutionContext, references: Mapping[str, Any] | None = None):
        """Poison mutable buffers in place, replacing immutable scalar/tuple wrappers."""
        def poison(value, reference=None):
            if isinstance(value, list):
                for index, child in enumerate(value):
                    value[index] = poison(child, reference[index] if isinstance(reference, (list, tuple)) and index < len(reference) else None)
            elif isinstance(value, tuple):
                return tuple(poison(child, reference[index] if isinstance(reference, (list, tuple)) and index < len(reference) else None)
                             for index, child in enumerate(value))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                return math.nan
            elif ctx.torch is not None and isinstance(value, ctx.torch.Tensor):
                if value.is_floating_point() or value.is_complex():
                    value.fill_(math.nan)
                else:
                    if not isinstance(reference, ctx.torch.Tensor) or value.dtype != reference.dtype:
                        raise ValueError("integer/boolean output poisoning requires a matching operator reference")
                    for destination, baseline in _paired_tensor_chunks(value, reference, 1024**2):
                        if value.dtype == ctx.torch.bool:
                            destination.copy_(ctx.torch.logical_not(baseline))
                        else:
                            bounds = ctx.torch.iinfo(value.dtype)
                            sentinel = ctx.torch.where(baseline == bounds.min,
                                                      ctx.torch.full_like(baseline, bounds.max),
                                                      ctx.torch.full_like(baseline, bounds.min))
                            destination.copy_(sentinel)
            return value
        for name, output in tuple(ctx.outputs.items()):
            poisoned = poison(output, references.get(name) if references else None)
            if poisoned is not output:
                ctx.outputs[name] = poisoned

    def _compare_outputs(self, compiled: CompiledAsset, atol: float, rtol: float):
        ctx = compiled.context
        expected = compiled.prepared.reference
        if not isinstance(ctx.outputs, Mapping):
            return False, {"__outputs__": {"reason": "outputs must be a named mapping"}}
        correct = ctx.outputs.keys() == expected.keys()
        errors = {}
        for name, reference in expected.items():
            if name not in ctx.outputs:
                errors[name] = {"reason": "expected output is missing"}
                continue
            actual = ctx.outputs[name]
            try:
                if self.kernel.runtime == "cuda":
                    torch = self._load_cuda()
                    if not isinstance(actual, torch.Tensor) or not isinstance(reference, torch.Tensor):
                        matched, maximum, nonfinite = False, None, False
                    elif actual.shape != reference.shape or actual.device != reference.device or actual.dtype != reference.dtype:
                        matched, maximum, nonfinite = False, None, False
                    else:
                        matched, maximum, nonfinite = self._compare_tensor_outputs(actual, reference, atol, rtol)
                else:
                    matched, maximum, nonfinite = _compare_python(actual, reference, atol, rtol)
            except (TypeError, ValueError, OverflowError) as exc:
                correct = False
                errors[name] = {"matches": False, "reason": f"{type(exc).__name__}: {exc}"}
                continue
            correct = correct and matched
            errors[name] = {"matches": matched, "max_absolute_error": maximum, "nonfinite_output": nonfinite}
        if ctx.outputs.keys() - expected.keys():
            errors["__unexpected_outputs__"] = {"names": sorted(ctx.outputs.keys() - expected.keys(), key=repr)}
        return bool(correct), errors

    def _compare_tensor_outputs(self, actual, expected, atol, rtol):
        torch = self._load_cuda()
        matched, nonfinite, maximum = True, False, 0.0
        integer_output = not actual.is_floating_point() and not actual.is_complex()
        for left, right in _paired_tensor_chunks(actual, expected, 1024**2):
            finite = bool(torch.isfinite(left).all().item()) and bool(torch.isfinite(right).all().item())
            nonfinite = nonfinite or not finite
            comparison = torch.equal(left, right) if integer_output else torch.allclose(left, right, atol=atol, rtol=rtol)
            matched = finite and bool(comparison) and matched
            if finite and left.numel() and not integer_output:
                promoted_left = left.to(torch.complex128) if left.is_complex() else left.double()
                promoted_right = right.to(torch.complex128) if right.is_complex() else right.double()
                difference = float((promoted_left - promoted_right).abs().max().item())
                maximum = max(maximum, difference)
        maximum = None if integer_output and not matched else maximum
        return matched, maximum if maximum is not None and not nonfinite and math.isfinite(maximum) else None, nonfinite

    def _structural_outputs(self, compiled: CompiledAsset):
        """Claims do not excuse missing buffers, wrong shapes or NaN outputs."""
        ctx, expected = compiled.context, compiled.prepared.reference
        if not isinstance(ctx.outputs, Mapping):
            return False, {"__outputs__": "outputs must be a named mapping"}
        correct = ctx.outputs.keys() == expected.keys()
        errors = {}

        def python_structure(left, right):
            if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
                if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != len(right):
                    return False
                return all(python_structure(a, b) for a, b in zip(left, right))
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in (left, right)):
                raise ValueError("output contains a nonnumeric or nonfinite scalar")
            return True

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
                         and actual.device == baseline.device
                         and all(bool(torch.isfinite(chunk).all().item()) for chunk in _tensor_chunks(actual, 1024**2))
                         and all(bool(torch.isfinite(chunk).all().item()) for chunk in _tensor_chunks(baseline, 1024**2)))
                if not valid:
                    errors[name] = "output tensor shape, dtype, device or finite-value check failed"
            else:
                try:
                    valid = python_structure(actual, baseline)
                    if not valid:
                        errors[name] = "output shape differs from operator output shape"
                except (TypeError, ValueError, OverflowError) as exc:
                    valid = False
                    errors[name] = str(exc)
            correct = correct and valid
        if ctx.outputs.keys() - expected.keys():
            errors["__unexpected_outputs__"] = sorted(ctx.outputs.keys() - expected.keys(), key=repr)
        return bool(correct), errors

    def check_correctness(self, compiled: CompiledAsset) -> tuple[bool, dict[str, Any]]:
        ctx = compiled.context
        claim, evaluated_claims = self._matching_claim(compiled)
        execution = {"steps": [step.name for step in compiled.plan.steps], "initialize_each_invocation": compiled.plan.initialize is not None,
                     "setup_once": compiled.plan.setup is not None, "setup_time_ms": compiled.setup_time_ms,
                     "build_time_ms": compiled.build_time_ms, "workspace_bytes": ctx.workspace_bytes,
                     "cuda_graph_setup_time_ms": ctx.metadata.get("cuda_graph_setup_time_ms"),
                     "workspace_allocations": ctx.metadata.get("workspace_allocations", []),
                     "compiled_resources": ctx.metadata.get("execution_resources", {})}
        origin = "platform_structure" if claim is not None else "platform_reference"
        input_ok, integrity = self._input_integrity(compiled)
        aliases = self._output_input_aliases(ctx)
        runs, errors = [], {}
        correct = input_ok and not aliases
        atol, rtol = self.operator_definition.tolerance(ctx.operator.dtype)
        if not correct:
            origin = "platform_input_integrity"
            errors["__inputs__"] = {"integrity": integrity, "aliased_outputs": aliases}
        else:
            for index in range(2):
                self._poison_outputs(ctx, compiled.prepared.reference)
                self._invoke(compiled)
                self._synchronize()
                aliases = self._output_input_aliases(ctx)
                if claim is not None:
                    output_ok, errors = self._structural_outputs(compiled)
                else:
                    output_ok, errors = self._compare_outputs(compiled, atol, rtol)
                input_ok, integrity = self._input_integrity(compiled)
                run_correct = output_ok and input_ok and not aliases
                if not input_ok or aliases:
                    origin = "platform_input_integrity"
                    errors["__inputs__"] = {"integrity": integrity, "aliased_outputs": aliases}
                runs.append({"invocation": index + 1, "correct": run_correct, "outputs": errors,
                             "input_integrity": integrity})
                correct = correct and run_correct
                if not run_correct:
                    break
        if not correct:
            self._remember_failure(compiled, evaluated_claims)
        details = {"status": ("declared" if claim is not None else "verified") if correct else "failed",
                   "correctness_origin": "submitter_claim" if correct and claim is not None else origin,
                   "verified_on_gpu": bool(correct and claim is None and ctx.runtime == "cuda"),
                   "verified_on_cpu": bool(correct and claim is None and ctx.runtime == "python"),
                   "comparison_performed": claim is None and bool(runs),
                   "outputs": errors, "checked_invocations": runs, "input_integrity": integrity,
                   "claims_evaluated": evaluated_claims, "execution": execution}
        if not correct:
            details["oracle_eligible"] = False
        if claim is None:
            details.update(atol=atol, rtol=rtol, reference_fingerprint=self.operator_definition.fingerprint)
        else:
            details.update(claim_id=claim.id, claim_fingerprint=claim.fingerprint, claim_source=claim.source,
                           claim_evidence=list(claim.evidence), structural_check="passed" if correct else "failed")
        compiled.correctness_details = details
        return bool(correct), details

    def check_after_measurement(self, compiled: CompiledAsset) -> tuple[bool, dict[str, Any]]:
        """Check the final timed output, without executing or timing another call."""
        self._synchronize()
        initial = compiled.correctness_details
        declared = initial.get("status") == "declared"
        if declared:
            correct, errors = self._structural_outputs(compiled)
        else:
            atol, rtol = self.operator_definition.tolerance(compiled.context.operator.dtype)
            correct, errors = self._compare_outputs(compiled, atol, rtol)
        input_ok, integrity = self._input_integrity(compiled)
        aliases = self._output_input_aliases(compiled.context)
        correct = bool(correct and input_ok and not aliases)
        origin = "platform_structure" if declared else "platform_reference"
        if not input_ok or aliases:
            origin = "platform_input_integrity"
            errors["__inputs__"] = {"integrity": integrity, "aliased_outputs": aliases}
        if not correct:
            self._remember_failure(compiled, initial.get("claims_evaluated", ()))
        return correct, {"status": "passed" if correct else "failed", "correctness_origin": origin,
                         "scope": "final timed invocation outputs and all input values after warmup/measurement",
                         "comparison_performed": not declared, "additional_invocations": 0,
                         "outputs": errors, "input_integrity": integrity}

    def warmup(self, compiled: CompiledAsset, iterations: int):
        for _ in range(iterations):
            if not self._output_input_aliases(compiled.context):
                self._poison_outputs(compiled.context, compiled.prepared.reference)
            self._invoke(compiled)
        self._synchronize()

    def measure(self, compiled: CompiledAsset, repetitions: int) -> tuple[float, ...]:
        samples = []
        if self.timing_mode == "cuda_graph":
            torch = self._load_cuda()
            compiled.cuda_graph.replay()
            self._synchronize()
            for _ in range(repetitions):
                if not self._output_input_aliases(compiled.context):
                    self._poison_outputs(compiled.context, compiled.prepared.reference)
                self._synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                compiled.cuda_graph.replay()
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)) / self.cuda_graph_batch_size)
            return tuple(samples)
        for _ in range(repetitions):
            if not self._output_input_aliases(compiled.context):
                self._poison_outputs(compiled.context, compiled.prepared.reference)
            self._synchronize()
            started = time.perf_counter()
            self._invoke(compiled)
            self._synchronize()
            samples.append((time.perf_counter() - started) * 1000)
        return tuple(samples)


__all__ = ["ExecutionContext", "ExecutionStep", "ExecutionPlan", "AssetKernelAdapter", "PreparedAsset", "CompiledAsset"]
