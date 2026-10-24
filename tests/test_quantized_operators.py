"""Contract tests for the canonical W4/W3/W2 packed affine format."""

import pytest

from kernel_tuning_lab.catalog import get_catalog
from kernel_tuning_lab.quantization import pack_bits, unpack_bits, dequantize_packed


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("k", [1, 7, 11, 22, 35, 65])
def test_bitstream_known_roundtrip_crosses_int32_words(bits, k):
    codes = [[((i * 7 + 1) % (1 << bits))] for i in range(k)]
    packed = pack_bits(codes, bits)
    assert unpack_bits(packed, bits, k) == codes
    assert len(packed) == (k * bits + 31) // 32
    if bits == 3 and k >= 11:
        # Hand-constructed low bits ensure a code starts at bit 30 and crosses a word.
        codes = [[0] for _ in range(k)]
        codes[10][0] = 7
        packed = pack_bits(codes, 3)
        assert unpack_bits(packed, 3, k) == codes


def test_pack_bits_fixed_known_vectors_and_validation():
    assert pack_bits([[0], [1], [2], [3]], 2) == [[0b11100100]]
    assert pack_bits([[1], [2], [3], [4]], 4) == [[0x4321]]
    with pytest.raises(ValueError):
        pack_bits([[4]], 2)
    with pytest.raises(ValueError):
        pack_bits([[True]], 2)
    with pytest.raises(ValueError):
        unpack_bits([[0]], 3, 33)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_dequantize_uses_group_zero_point_and_partial_tail(bits):
    codes = [[x % (1 << bits)] for x in [0, 1, (1 << bits) - 1, 3, 5, 2, 7]]
    packed = pack_bits(codes, bits)
    scales = [[0.5], [2.0], [1.0]]
    zeros = [[0], [1], [2]]
    actual = dequantize_packed(packed, scales, zeros, bits, 3, len(codes), dtype="float64")
    expected = [((codes[i][0] - zeros[i // 3][0]) * scales[i // 3][0]) for i in range(len(codes))]
    assert [row[0] for row in actual] == expected


def test_quantized_assets_are_auto_discovered_as_six_pairs():
    catalog = get_catalog()
    for bits in (4, 3, 2):
        for op in (f"dequantize_w{bits}@1.0.0", f"w{bits}a16_gemm@1.0.0"):
            assert catalog.resolve("operator", op)
            assert catalog.resolve("kernel", f"python/{op.split('@')[0]}@1.0.0")
            assert catalog.resolve("kernel", f"triton/{op.split('@')[0]}@1.0.0")
