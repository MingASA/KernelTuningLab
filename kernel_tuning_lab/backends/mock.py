"""Deterministic synthetic performance landscape, never a real GPU benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import math
import platform
import random
from typing import Any

from ..models import HardwareSpec, KernelConfig, OperatorSpec, ShapeSpec, stable_id
from ..space import ValidationResult


@dataclass
class MockCompiled:
    hardware: HardwareSpec
    shape: ShapeSpec
    config: KernelConfig
    seed: int
    token: int
    latency_ms: float


class MockAdapter:
    name = "mock"
    provenance = "mock"
    measurement_mode = "synthetic_roofline_v2"

    def __init__(self, compile_failure_rate: float = 0.025, correctness_failure_rate: float = 0.02, runtime_failure_rate: float = 0.01):
        self.rates = (compile_failure_rate, correctness_failure_rate, runtime_failure_rate)
        if any(not math.isfinite(r) or not 0 <= r <= 1 for r in self.rates):
            raise ValueError("mock failure rates must be in [0,1]")

    def environment(self) -> dict[str, Any]:
        return {"python": platform.python_version(), "backend": "mock", "model_version": "2"}

    def protocol_metadata(self) -> dict[str, Any]:
        return {"failure_rates": list(self.rates), "synthetic": True,
                "note": "runtime samples are simulated; compile and wall times are measured CPU overhead"}

    def validate(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec, config: KernelConfig) -> ValidationResult:
        if operator.kind not in {"gemm", "batched_gemm", "softmax", "layernorm"}:
            return ValidationResult(False, ("unsupported mock operator",))
        required = {"M", "N", "K", "Batch"} if operator.kind == "batched_gemm" else {"M", "N", "K"} if operator.kind == "gemm" else {"M", "N"}
        if not required <= shape.dims.keys():
            return ValidationResult(False, (f"{operator.kind} requires {sorted(required)}",))
        if operator.implementation != f"mock_{operator.kind}":
            return ValidationResult(False, ("MockAdapter implementation does not match operator kind",))
        if operator.kind == "gemm" and (shape.get("Batch") != 1 or shape.get("batch") != 1):
            return ValidationResult(False, ("v0.1 GEMM adapter supports unbatched GEMM only",))
        return ValidationResult(True)

    def prepare(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec, seed: int) -> tuple[HardwareSpec, OperatorSpec, ShapeSpec, int]:
        return hardware, operator, shape, seed

    def compile(self, prepared: tuple[HardwareSpec, OperatorSpec, ShapeSpec, int], config: KernelConfig) -> MockCompiled:
        hardware, operator, shape, seed = prepared
        token = int(stable_id([hardware.id, operator.kind, shape.id, config.id, seed]), 16)
        rng = random.Random(token)
        if rng.random() < self.rates[0]:
            raise RuntimeError("injected mock compilation failure")
        if operator.kind in {"softmax", "layernorm"}:
            rows, cols = shape.get("M"), shape.get("N")
            block = config.get("BLOCK_SIZE")
            warps, stages = config.get("num_warps"), config.get("num_stages")
            bandwidth = hardware.bandwidth_gbps or 600.0
            byte_width = 4 if operator.dtype == "float32" else 2
            passes = 3 if operator.kind == "softmax" else 5
            memory_ms = rows * cols * byte_width * passes / (bandwidth * 1e6)
            utilization = max(0.1, min(1.0, rows / max(1, hardware.sm_count)))
            padding = block / cols
            warp_penalty = 1 + 0.15 * abs(warps - (4 if block <= 1024 else 8)) / 4
            latency = (0.002 + memory_ms / utilization) * padding**0.35 * warp_penalty * (1 + 0.03 * (stages - 1))
            latency *= 0.94 + 0.12 * rng.random()
            return MockCompiled(hardware, shape, config, seed, token, latency)
        m, n, k = (shape.get(d) for d in ("M", "N", "K"))
        bm, bn, bk = (config.get(d) for d in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        warps, stages = config.get("num_warps"), config.get("num_stages")
        compute = hardware.compute_tflops or 40.0
        bandwidth = hardware.bandwidth_gbps or 600.0
        batch = shape.get("Batch")
        compute_ms = 2.0 * batch * m * n * k / (compute * 1e9)
        memory_ms = 2.0 * batch * (m * k + k * n + m * n) / (bandwidth * 1e6)
        useful = (m * n) / (math.ceil(m / bm) * bm * math.ceil(n / bn) * bn)
        blocks = batch * math.ceil(m / bm) * math.ceil(n / bn)
        occupancy = min(1.0, blocks / max(1.0, hardware.sm_count * 1.5))
        tile_efficiency = min(1.0, math.sqrt(bm * bn) / 90.0)
        ratio = compute / bandwidth
        ideal_stages = 2.0 if ratio < 0.07 else 3.5
        stage_penalty = 1.0 + 0.10 * abs(stages - ideal_stages)
        warp_penalty = 1.0 + 0.15 * abs(warps - (4 if bm * bn <= 4096 else 8)) / 4
        efficiency = max(0.06, useful * (0.4 + 0.6 * occupancy) * tile_efficiency)
        latency = 0.003 + max(compute_ms, memory_ms) * stage_penalty * warp_penalty / efficiency
        latency *= 1.0 + (bk / 32 - 1) * (0.1 if k < 512 else -0.06)
        latency *= 0.93 + 0.14 * rng.random()
        return MockCompiled(hardware, shape, config, seed, token, latency)

    def check_correctness(self, compiled: MockCompiled) -> tuple[bool, dict[str, Any]]:
        valid = random.Random(compiled.token + 1).random() >= self.rates[1]
        return valid, {"correctness_mode": "synthetic", "verified_on_gpu": False,
                       "description": "mock correctness flag, no real tensor calculation"}

    def warmup(self, compiled: MockCompiled, iterations: int) -> None:
        # Synthetic samples have no runtime state to warm up.
        return None

    def measure(self, compiled: MockCompiled, repetitions: int) -> tuple[float, ...]:
        rng = random.Random(compiled.token + 2)
        if rng.random() < self.rates[2]:
            raise RuntimeError("injected mock runtime failure")
        return tuple(compiled.latency_ms * max(0.8, rng.gauss(1.0, 0.012)) for _ in range(repetitions))
