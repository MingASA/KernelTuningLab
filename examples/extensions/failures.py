"""Deliberately invalid contributions for automatic acceptance examples.

These entries load normally.  Kernel correctness, proposal validation and space
capability matching must reject the corresponding experiments with evidence.
"""

from kernel_tuning_lab.sdk import (KernelConfig, KernelDefinition, ParameterSpec,
                                   SearchAlgorithmDefinition, SearchCapabilities, SpaceDefinition)


def build_wrong_gemm(ctx, tile_size=1):
    def launch():
        ctx.outputs["out"] = [[0.0] * ctx.shape.get("N") for _ in range(ctx.shape.get("M"))]
    return launch


class IllegalSearch:
    name = "demo/illegal"

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if budget.exhausted(0, 0):
            return []
        parameters = dict(space.config_at(0).params)
        key = "__undeclared_demo_parameter__"
        while key in space.domains:
            key += "_"
        parameters[key] = 1
        return [KernelConfig(parameters)]


class NumericOnlySearch:
    name = "demo/numeric_only"

    def __init__(self, seed=0):
        self.seed = seed
        self._cursors = {}

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if budget.exhausted(0, 0):
            return []
        seen = {record.config.id for record in history}
        seen.update(getattr(history, "attempted_config_ids", ()))
        key = (hardware.id, operator.id, shape.id, space.fingerprint)
        cursor = self._cursors.get(key, 0)
        result = []
        stop = min(space.cardinality, cursor + 256)
        while cursor < stop and len(result) < min(batch_size, budget.max_evaluations):
            candidate = space.config_at(cursor)
            cursor += 1
            if candidate.id not in seen and space.validate(candidate, hardware, operator, shape).valid:
                result.append(candidate)
                seen.add(candidate.id)
        self._cursors[key] = cursor
        return result


def create_illegal(seed=0):
    return IllegalSearch()


def create_numeric_only(seed=0):
    return NumericOnlySearch(seed)


ASSETS = (
    KernelDefinition(
        id="demo/wrong_gemm", version="1.0.0", operator="gemm@1.0.0",
        build=build_wrong_gemm, space=SpaceDefinition({"tile_size": ParameterSpec((1, 4))}),
        runtime="python", dtypes=("float32", "float64"),
        metadata={"description": "Intentional wrong output: automatic correctness checks must fail"}),
    SearchAlgorithmDefinition(
        id="demo/illegal", version="1.0.0", factory=create_illegal,
        metadata={"description": "Intentional undeclared parameter: automatic proposal checks must fail"}),
    SearchAlgorithmDefinition(
        id="demo/numeric_only", version="1.0.0", factory=create_numeric_only,
        capabilities=SearchCapabilities(parameter_kinds=("integer",)),
        metadata={"description": "Integer-only capability declaration: categorical spaces must be rejected"}),
)
