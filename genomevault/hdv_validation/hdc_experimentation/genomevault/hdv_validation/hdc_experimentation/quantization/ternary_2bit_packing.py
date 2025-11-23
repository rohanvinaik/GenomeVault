"""
2-Bit Ternary Packing for HDC Pipeline Optimization

This module provides LOSSLESS 2-bit packing for ternary {-1, 0, +1} values.

Encoding scheme:
  -1 → 0b00 (00)
   0 → 0b01 (01)
  +1 → 0b10 (10)

Packing: 4 ternary values per byte (2 bits × 4 = 8 bits)

Storage reduction:
  - Original: int8 = 1 byte per value
  - Packed: 2 bits per value = 0.25 bytes per value
  - Reduction: 4× per dimension

For D=5,120 per bank × 3 banks:
  - Original: 15,360 bytes per chunk
  - Packed: 3,840 bytes per chunk
  - Reduction: 4×

Reference: COMPREHENSIVE_OPTIMIZATION_ROADMAP.md Lines 577-650

Author: Claude Code
Date: November 21, 2025
"""

import numpy as np
from typing import Tuple
from numba import njit


@njit(cache=True, fastmath=True)
def pack_ternary_to_2bit(ternary: np.ndarray) -> np.ndarray:
    """
    Pack ternary {-1, 0, +1} values into 2-bit representation.

    Encoding:
      -1 → 0b00
       0 → 0b01
      +1 → 0b10

    Args:
        ternary: int8 array with values {-1, 0, +1}, shape (N,)

    Returns:
        uint8 array with 4 values packed per byte, shape (N // 4,)

    Example:
        >>> ternary = np.array([1, -1, 0, 1], dtype=np.int8)
        >>> packed = pack_ternary_to_2bit(ternary)
        >>> # Result: 0b10_00_01_10 = 0x8A
    """
    N = len(ternary)
    assert N % 4 == 0, f"Array length must be multiple of 4, got {N}"

    packed_size = N // 4
    packed = np.zeros(packed_size, dtype=np.uint8)

    for i in range(packed_size):
        byte_val = 0
        for j in range(4):
            idx = 4 * i + j
            val = ternary[idx]

            # Encode ternary to 2-bit
            if val == -1:
                bits = 0b00
            elif val == 0:
                bits = 0b01
            else:  # val == 1
                bits = 0b10

            # Pack into byte (MSB first)
            shift = 6 - 2 * j  # Shifts: 6, 4, 2, 0
            byte_val |= (bits << shift)

        packed[i] = byte_val

    return packed


@njit(cache=True, fastmath=True)
def unpack_2bit_to_ternary(packed: np.ndarray) -> np.ndarray:
    """
    Unpack 2-bit representation back to ternary {-1, 0, +1}.

    LOSSLESS: This function MUST return bit-identical values to the original.

    Args:
        packed: uint8 array with 4 values packed per byte, shape (M,)

    Returns:
        int8 array with ternary values {-1, 0, +1}, shape (M * 4,)

    Example:
        >>> packed = np.array([0x8A], dtype=np.uint8)  # 0b10_00_01_10
        >>> ternary = unpack_2bit_to_ternary(packed)
        >>> # Result: [1, -1, 0, 1]
    """
    packed_size = len(packed)
    ternary_size = packed_size * 4
    ternary = np.zeros(ternary_size, dtype=np.int8)

    for i in range(packed_size):
        byte_val = packed[i]

        for j in range(4):
            # Extract 2 bits (MSB first)
            shift = 6 - 2 * j  # Shifts: 6, 4, 2, 0
            bits = (byte_val >> shift) & 0b11

            # Decode 2-bit to ternary
            if bits == 0b00:
                val = -1
            elif bits == 0b01:
                val = 0
            else:  # bits == 0b10 (or 0b11, doesn't matter)
                val = 1

            idx = 4 * i + j
            ternary[idx] = val

    return ternary


def pack_3bank_chunk(
    bank1: np.ndarray,
    bank2: np.ndarray,
    bank3: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack all 3 banks of a chunk to 2-bit representation.

    Args:
        bank1: int8 array, shape (D,), Hydrophobic bank
        bank2: int8 array, shape (D,), Major Groove bank
        bank3: int8 array, shape (D,), Hinge bank

    Returns:
        Tuple of 3 uint8 arrays, each shape (D // 4,)

    Storage savings:
        - Input: 3 × D bytes (D=5,120 → 15,360 bytes)
        - Output: 3 × D/4 bytes (D=5,120 → 3,840 bytes)
        - Reduction: 4×
    """
    D = len(bank1)
    assert len(bank2) == D and len(bank3) == D, "All banks must have same dimension"
    assert D % 4 == 0, f"Dimension D must be multiple of 4, got {D}"

    packed_bank1 = pack_ternary_to_2bit(bank1)
    packed_bank2 = pack_ternary_to_2bit(bank2)
    packed_bank3 = pack_ternary_to_2bit(bank3)

    return packed_bank1, packed_bank2, packed_bank3


def unpack_3bank_chunk(
    packed_bank1: np.ndarray,
    packed_bank2: np.ndarray,
    packed_bank3: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack all 3 banks of a chunk from 2-bit representation.

    LOSSLESS: Returns bit-identical values to original unpacked data.

    Args:
        packed_bank1: uint8 array, shape (D // 4,)
        packed_bank2: uint8 array, shape (D // 4,)
        packed_bank3: uint8 array, shape (D // 4,)

    Returns:
        Tuple of 3 int8 arrays, each shape (D,)
    """
    bank1 = unpack_2bit_to_ternary(packed_bank1)
    bank2 = unpack_2bit_to_ternary(packed_bank2)
    bank3 = unpack_2bit_to_ternary(packed_bank3)

    return bank1, bank2, bank3


def validate_packing_lossless(ternary_original: np.ndarray) -> bool:
    """
    Validate that packing/unpacking is truly lossless.

    This is a CRITICAL validation - any loss would invalidate the
    lens confidence trajectory analysis.

    Args:
        ternary_original: int8 array with values {-1, 0, +1}

    Returns:
        True if pack → unpack is bit-identical to original

    Raises:
        AssertionError if ANY mismatch detected
    """
    # Pad to multiple of 4 if needed
    N = len(ternary_original)
    if N % 4 != 0:
        pad_size = 4 - (N % 4)
        ternary_padded = np.pad(ternary_original, (0, pad_size), constant_values=0)
    else:
        ternary_padded = ternary_original

    # Pack
    packed = pack_ternary_to_2bit(ternary_padded)

    # Unpack
    unpacked = unpack_2bit_to_ternary(packed)

    # Validate bit-identical (exclude padding)
    unpacked_unpadded = unpacked[:N]
    is_identical = np.array_equal(ternary_original, unpacked_unpadded)

    if not is_identical:
        # Find first mismatch for debugging
        mismatch_idx = np.where(ternary_original != unpacked_unpadded)[0]
        if len(mismatch_idx) > 0:
            idx = mismatch_idx[0]
            raise AssertionError(
                f"Packing is NOT lossless!\n"
                f"  Mismatch at index {idx}:\n"
                f"    Original: {ternary_original[idx]}\n"
                f"    Unpacked: {unpacked_unpadded[idx]}"
            )

    return is_identical


# Storage size calculations
def calculate_storage_reduction(num_chunks: int, D: int = 5120) -> dict:
    """
    Calculate storage savings from 2-bit packing.

    Args:
        num_chunks: Number of chunks in encoded genome
        D: Dimension per bank (default: 5,120)

    Returns:
        Dictionary with storage statistics

    Example:
        For full genome (3,370,053 chunks):
          - Original: 51.8 GB
          - Packed: 12.9 GB
          - Reduction: 4×
          - With compression (gzip 2.5×): 5.2 GB
          - Total reduction: 10×
    """
    bytes_per_chunk_original = 3 * D * 1  # 3 banks × D × int8
    bytes_per_chunk_packed = 3 * (D // 4) * 1  # 3 banks × D/4 × uint8

    total_original_bytes = num_chunks * bytes_per_chunk_original
    total_packed_bytes = num_chunks * bytes_per_chunk_packed

    # Estimate with compression (gzip achieves ~2.5× on packed data)
    compression_ratio = 2.5
    total_compressed_bytes = total_packed_bytes / compression_ratio

    return {
        'num_chunks': num_chunks,
        'D': D,
        'bytes_per_chunk_original': bytes_per_chunk_original,
        'bytes_per_chunk_packed': bytes_per_chunk_packed,
        'total_original_GB': total_original_bytes / (1024**3),
        'total_packed_GB': total_packed_bytes / (1024**3),
        'total_compressed_GB': total_compressed_bytes / (1024**3),
        'reduction_packing': bytes_per_chunk_original / bytes_per_chunk_packed,
        'reduction_with_compression': total_original_bytes / total_compressed_bytes,
    }


if __name__ == '__main__':
    # Quick validation test
    print("=" * 80)
    print("2-Bit Ternary Packing Validation")
    print("=" * 80)

    # Test 1: Single chunk validation
    print("\nTest 1: Lossless validation on random chunk")
    np.random.seed(42)
    D = 5120
    test_bank = np.random.choice([-1, 0, 1], size=D).astype(np.int8)

    try:
        is_lossless = validate_packing_lossless(test_bank)
        print(f"  ✓ Packing is LOSSLESS: {is_lossless}")
    except AssertionError as e:
        print(f"  ✗ Packing FAILED: {e}")
        exit(1)

    # Test 2: Storage reduction calculation
    print("\nTest 2: Storage reduction for full genome")
    stats = calculate_storage_reduction(num_chunks=3_370_053, D=5120)
    print(f"  Original size: {stats['total_original_GB']:.2f} GB")
    print(f"  Packed size: {stats['total_packed_GB']:.2f} GB")
    print(f"  Compressed size: {stats['total_compressed_GB']:.2f} GB")
    print(f"  Reduction (packing only): {stats['reduction_packing']:.1f}×")
    print(f"  Reduction (with compression): {stats['reduction_with_compression']:.1f}×")

    # Test 3: 3-bank chunk packing
    print("\nTest 3: 3-bank chunk packing")
    bank1 = np.random.choice([-1, 0, 1], size=D).astype(np.int8)
    bank2 = np.random.choice([-1, 0, 1], size=D).astype(np.int8)
    bank3 = np.random.choice([-1, 0, 1], size=D).astype(np.int8)

    packed1, packed2, packed3 = pack_3bank_chunk(bank1, bank2, bank3)
    unpacked1, unpacked2, unpacked3 = unpack_3bank_chunk(packed1, packed2, packed3)

    assert np.array_equal(bank1, unpacked1), "Bank 1 mismatch!"
    assert np.array_equal(bank2, unpacked2), "Bank 2 mismatch!"
    assert np.array_equal(bank3, unpacked3), "Bank 3 mismatch!"

    print(f"  ✓ All 3 banks pack/unpack correctly")
    print(f"  Original size: {(len(bank1) + len(bank2) + len(bank3))} bytes")
    print(f"  Packed size: {(len(packed1) + len(packed2) + len(packed3))} bytes")
    print(f"  Reduction: {(len(bank1) + len(bank2) + len(bank3)) / (len(packed1) + len(packed2) + len(packed3)):.1f}×")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED - 2-bit packing is production-ready!")
    print("=" * 80)
