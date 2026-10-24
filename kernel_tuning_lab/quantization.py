"""CPU-safe tools for the built-in unsigned affine quantization format.

The format is intentionally explicit and does not claim GPTQ/AWQ compatibility.
``codes`` has shape [K, N]. Each column is a continuous least-significant-bit
first stream along K, stored in a row-major signed int32 matrix with shape
[ceil(K * bits / 32), N]. A 3-bit code may occupy two adjacent words. Unused
high bits of the last word are zero when produced by :func:`pack_bits`.

These functions accept rectangular nested Python sequences and return lists.
Their results can be passed to ``torch.tensor(..., dtype=torch.int32)`` without
unsigned-integer overflow. No optional tensor or GPU package is imported here.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import struct


SUPPORTED_BITS = (2, 3, 4)
PACKING_FORMAT = "canonical_unsigned_affine_k_bitstream"
_WORD_MASK = (1 << 32) - 1


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bits(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in SUPPORTED_BITS:
        raise ValueError("bits must be one of 2, 3, 4")
    return value


def _matrix(value, name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{name} must be a nonempty rectangular nested sequence")
    if not isinstance(value[0], Sequence) or isinstance(value[0], (str, bytes)) or not value[0]:
        raise ValueError(f"{name} must have nonempty rows")
    columns = len(value[0])
    if any(not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != columns for row in value):
        raise ValueError(f"{name} must be rectangular")
    return len(value), columns


def _signed_word(value: int) -> int:
    value &= _WORD_MASK
    return value if value < (1 << 31) else value - (1 << 32)


def pack_bits(codes, bits: int) -> list[list[int]]:
    """Pack integer ``codes[K][N]`` into signed int32 words along K.

    Codes must be integers in ``[0, 2**bits - 1]``; booleans are rejected.
    This is an unsigned code format even when its int32 storage is negative.
    """
    bits = _bits(bits)
    k, n = _matrix(codes, "codes")
    maximum = (1 << bits) - 1
    packed = [[0] * n for _ in range((k * bits + 31) // 32)]
    for row, values in enumerate(codes):
        word, shift = divmod(row * bits, 32)
        for column, code in enumerate(values):
            if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= maximum:
                raise ValueError(f"codes[{row}][{column}] must be an integer in [0, {maximum}]")
            packed[word][column] |= (code << shift) & _WORD_MASK
            if shift + bits > 32:
                packed[word + 1][column] |= code >> (32 - shift)
    return [[_signed_word(word) for word in row] for row in packed]


def unpack_bits(packed, bits: int, k: int) -> list[list[int]]:
    """Unpack signed int32 ``packed[ceil(k*bits/32)][N]`` into codes[K][N].

    ``k`` is required because padding bits do not encode the original length.
    Padding is ignored, allowing externally supplied int32 words with any tail
    padding while preserving exactly the same logical codes.
    """
    bits = _bits(bits)
    k = _positive_integer(k, "k")
    words, n = _matrix(packed, "packed")
    if words != (k * bits + 31) // 32:
        raise ValueError("packed first dimension must equal ceil(k * bits / 32)")
    for row in packed:
        if any(isinstance(word, bool) or not isinstance(word, int) or not -(1 << 31) <= word < (1 << 31) for word in row):
            raise ValueError("packed values must be signed int32 integers")
    maximum = (1 << bits) - 1
    output = []
    for row in range(k):
        word, shift = divmod(row * bits, 32)
        values = []
        for column in range(n):
            code = (packed[word][column] & _WORD_MASK) >> shift
            if shift + bits > 32:
                code |= (packed[word + 1][column] & _WORD_MASK) << (32 - shift)
            values.append(code & maximum)
        output.append(values)
    return output


def dequantize_packed(packed, scales, zero_points, bits: int, group_size: int,
                      k: int, dtype: str = "float64") -> list[list[float]]:
    """Return ``(code - zero_points[k//group_size,n]) * scales[...]``.

    Scales and zero points have shape [ceil(k/group_size), N]. Scales must be
    finite and positive, and zero points are integer unsigned codes. Setting a
    zero point to ``2**(bits - 1)`` expresses a midpoint symmetric convention.
    ``dtype`` selects float32 rounding or Python's float64 calculation.
    """
    bits = _bits(bits)
    k = _positive_integer(k, "k")
    group_size = _positive_integer(group_size, "group_size")
    if dtype not in {"float32", "float64"}:
        raise ValueError("stdlib dequantization supports float32/float64")
    codes = unpack_bits(packed, bits, k)
    groups, n = (k + group_size - 1) // group_size, len(codes[0])
    if _matrix(scales, "scales") != (groups, n) or _matrix(zero_points, "zero_points") != (groups, n):
        raise ValueError("scales and zero_points must have shape [ceil(k/group_size), N]")
    maximum = (1 << bits) - 1
    for row in scales:
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in row):
            raise ValueError("scales must contain finite positive numbers")
    for row in zero_points:
        if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum for value in row):
            raise ValueError(f"zero_points must contain integers in [0, {maximum}]")
    def cast(value):
        return struct.unpack("f", struct.pack("f", value))[0] if dtype == "float32" else float(value)
    return [[cast((code - zero_points[row // group_size][column]) * cast(scales[row // group_size][column]))
             for column, code in enumerate(values)] for row, values in enumerate(codes)]


__all__ = ["SUPPORTED_BITS", "PACKING_FORMAT", "pack_bits", "unpack_bits", "dequantize_packed"]
