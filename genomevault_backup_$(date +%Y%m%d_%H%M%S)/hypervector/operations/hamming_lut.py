"""Simplified Hamming distance utilities.

This module replaces a previously corrupted implementation. It provides a
deterministic popcount look-up table and a CPU based Hamming distance routine
that the rest of the hypervector package can rely on during testing.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - numba is optional
    from numba import jit
except Exception:  # pragma: no cover - executed when numba is absent
    jit = None  # type: ignore[assignment]

from genomevault.hypervector.types import VectorUInt64

_POPCOUNT_LUT_16: NDArray[np.uint8] | None = None


def generate_popcount_lut() -> NDArray[np.uint8]:
    """Generate a 16-bit population-count look-up table."""
    global _POPCOUNT_LUT_16
    if _POPCOUNT_LUT_16 is None:
        _POPCOUNT_LUT_16 = np.array(
            [bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8
        )
    return _POPCOUNT_LUT_16


def popcount_u64(arr: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Compute the population count for each element of ``arr``."""
    lut = generate_popcount_lut()
    result = np.zeros_like(arr, dtype=np.uint64)
    for index, val in np.ndenumerate(arr):
        result[index] = (
            lut[(val >> 0) & 0xFFFF]
            + lut[(val >> 16) & 0xFFFF]
            + lut[(val >> 32) & 0xFFFF]
            + lut[(val >> 48) & 0xFFFF]
        )
    return result


if jit is not None:  # pragma: no cover - jitted path

    @jit(nopython=True, cache=True)
    def _popcount_u64_fast(
        arr: NDArray[np.uint64], lut: NDArray[np.uint8]
    ) -> NDArray[np.uint64]:
        result = np.zeros(arr.shape, dtype=np.uint64)
        for i in range(arr.size):
            val = arr.flat[i]
            result.flat[i] = (
                lut[(val >> 0) & 0xFFFF]
                + lut[(val >> 16) & 0xFFFF]
                + lut[(val >> 32) & 0xFFFF]
                + lut[(val >> 48) & 0xFFFF]
            )
        return result

else:

    def _popcount_u64_fast(
        arr: NDArray[np.uint64], lut: NDArray[np.uint8]
    ) -> NDArray[np.uint64]:
        return popcount_u64(arr)


def hamming_distance_cpu(
    vec1: VectorUInt64, vec2: VectorUInt64, lut: NDArray[np.uint8] | None = None
) -> int:
    """Return the Hamming distance between ``vec1`` and ``vec2``."""
    if lut is None:
        lut = generate_popcount_lut()
    xor = np.bitwise_xor(vec1, vec2)
    return int(_popcount_u64_fast(xor, lut).sum())


class HammingLUT:
    """Compute Hamming distances using a cached look-up table."""

    def __init__(self) -> None:
        self.lut = generate_popcount_lut()

    def distance(self, vec1: VectorUInt64, vec2: VectorUInt64) -> int:
        """Compute the distance between two vectors."""
        return hamming_distance_cpu(vec1, vec2, self.lut)

    def distance_batch(
        self, vecs1: NDArray[np.uint64], vecs2: NDArray[np.uint64]
    ) -> NDArray[np.int32]:
        """Compute pairwise Hamming distances for two batches of vectors."""
        result = np.zeros((vecs1.shape[0], vecs2.shape[0]), dtype=np.int32)
        for i, v1 in enumerate(vecs1):
            for j, v2 in enumerate(vecs2):
                result[i, j] = hamming_distance_cpu(v1, v2, self.lut)
        return result


def export_platform_implementations() -> Dict[str, str]:
    """Report available accelerator implementations."""
    impls: Dict[str, str] = {"cpu": "lookup-table"}
    if jit is not None:
        impls["numba"] = "jit"
    return impls


__all__ = [
    "HammingLUT",
    "export_platform_implementations",
    "generate_popcount_lut",
    "hamming_distance_cpu",
    "popcount_u64",
]

