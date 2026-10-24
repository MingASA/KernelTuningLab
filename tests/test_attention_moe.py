"""Independent semantic and edge-case tests for attention and grouped MoE GEMM."""

import math

import pytest

from kernel_tuning_lab.builtin_assets import attention, moe
from kernel_tuning_lab.catalog import get_catalog


def test_attention_bottom_right_causal_non_square_and_gqa_reference():
    # Sq=2, Sk=4: bottom-right alignment permits keys <= i+2.
    class Shape:
        def get(self, name):
            return {"Batch": 1, "Heads": 2, "KVHeads": 1, "SeqQ": 2,
                    "SeqKV": 4, "D": 1}[name]

    class Context:
        shape = Shape()
        operator = type("Operator", (), {"parameters": {"causal": True, "softmax_scale": 1.0},
                                         "dtype": "float64"})()
        runtime = "python"
        inputs = {
            "q": [[[[1.0], [2.0]], [[1.0], [2.0]]]],
            "k": [[[[0.0], [1.0], [2.0], [3.0]]]],
            "v": [[[[1.0], [2.0], [4.0], [8.0]]]],
        }

    out = attention._reference(Context())["out"]
    assert len(out) == 1 and len(out[0]) == 2 and len(out[0][0]) == 2
    # For query row 0, keys 0..2 are visible; row 1 sees all four.
    def weighted(q, count):
        logits = [q * float(i) for i in range(count)]
        peak = max(logits)
        probs = [math.exp(x - peak) for x in logits]
        return sum(p * v for p, v in zip(probs, (1.0, 2.0, 4.0, 8.0))) / sum(probs)
    assert out[0][0][0][0] == pytest.approx(weighted(1.0, 3))
    assert out[0][0][1][0] == pytest.approx(weighted(2.0, 4))
    assert out[0][1] == out[0][0]


@pytest.mark.parametrize("scale,dtype,dim,expected", [
    (1.0, "float16", 64, True),
    (2.5e38, "bfloat16", 1, False),
    (-2.5e38, "bfloat16", 1, False),
    (7e37, "float16", 256, False),
])
def test_attention_exp2_prescale_overflow_guard(scale, dtype, dim, expected):
    assert attention._exp2_can_prescale(scale, dtype, dim) is expected


@pytest.mark.parametrize("parameters", [
    {"dropout": 0.1}, {"causal": 1}, {"causal_alignment": "sideways"},
    {"softmax_scale": float("inf")}, {"unsupported": True},
])
def test_attention_rejects_invalid_semantic_parameters(parameters):
    shape = type("Shape", (), {"dims": {"Batch": 1, "Heads": 2, "KVHeads": 1,
                                       "SeqQ": 3, "SeqKV": 5, "D": 7},
                               "get": lambda self, name: self.dims[name]})()
    operator = type("Operator", (), {"kind": "attention", "parameters": parameters})()
    assert attention._semantics_reason(operator, shape)


def test_attention_and_moe_assets_auto_discover_all_runtime_pairs():
    catalog = get_catalog()
    for op in ("attention@1.0.0", "moe_grouped_gemm@1.0.0"):
        catalog.resolve("operator", op)
        stem = op.split("@")[0]
        catalog.resolve("kernel", f"python/{stem}@1.0.0")
        catalog.resolve("kernel", f"triton/{stem}@1.0.0")


def test_moe_group_count_and_offsets_contracts_include_empty_experts():
    assert moe.group_counts(7, 4, "balanced") == (2, 2, 2, 1)
    assert moe.group_counts(7, 4, "skewed") == (7, 0, 0, 0)
    assert moe.group_counts(7, 4, "empty_first")[0] == 0
    assert moe.group_counts(7, 4, "empty_last")[-1] == 0
    assert moe.validate_group_offsets([0, 2, 2, 5], 5, 3) == (0, 2, 2, 5)
    for offsets in ([1, 2, 5], [0, 4, 3, 5], [0, 2, 6], [0, True, 5]):
        with pytest.raises(ValueError):
            moe.validate_group_offsets(offsets, 5, len(offsets) - 1)
    with pytest.raises(ValueError):
        moe.group_counts(2, 1, "empty_last")


def test_moe_explicit_group_sizes_validate_and_validation_factory_adapts():
    catalog = get_catalog()
    op = catalog.resolve("operator", "moe_grouped_gemm@1.0.0")
    assert op.validation_case_factory({"group_sizes": [0, 3, 0, 2]}, ())
    with pytest.raises(ValueError):
        moe._counts({"group_sizes": [1, -1]}, 0, 2)
    with pytest.raises(ValueError):
        moe._counts({"group_sizes": [2, 2]}, 3, 2)
