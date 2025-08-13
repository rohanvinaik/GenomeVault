"""Binding module."""
from __future__ import annotations

from collections.abc import Iterable
import warnings

import numpy as np

from genomevault.hypervector.types import VectorF64

warnings.warn(
    "genomevault.hypervector.operations.binding is deprecated. "
    "Use genomevault.hypervector_transform.binding_operations instead.",
    DeprecationWarning,
    stacklevel=2,
)


def circular_convolution(a: VectorF64, b: VectorF64) -> VectorF64:
    """Bind two hypervectors via circular convolution (fft-based).

    Args:
        a, b: 1-D float arrays of equal length
    Returns:
        1-D float array of same length
    """
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError("a and b must be 1-D arrays with equal length")
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    return np.real(np.fft.ifft(fa * fb))


def element_wise_multiply(a: VectorF64, b: VectorF64) -> VectorF64:
    """Bind two hypervectors via element-wise multiplication."""
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError("a and b must be 1-D arrays with equal length")
    return a * b


def permutation_binding(v: VectorF64, shift: int = 1) -> VectorF64:
    """Apply cyclic permutation (roll) used for sequence encoding."""
    if v.ndim != 1:
        raise ValueError("v must be a 1-D array")
    if not isinstance(shift, int):
        raise ValueError("shift must be an integer")
    L = v.shape[0]
    return np.roll(v, shift % L)


def _safe_norm(x: VectorF64) -> float:
    n = float(np.linalg.norm(x))
    return n if n > 0.0 else 1.0


def bundle(vectors: Iterable[VectorF64]) -> VectorF64:
    """Bundle multiple hypervectors via normalized addition.

    Args:
        vectors: iterable of 1-D float arrays of equal length
    Returns:
        normalized sum (L2) of inputs; if all-zero, returns the zero vector
    """
    vectors = list(vectors)
    if not vectors:
        raise ValueError("vectors must be non-empty")
    L = vectors[0].shape[0]
    for v in vectors:
        if v.ndim != 1 or v.shape[0] != L:
            raise ValueError("all vectors must be 1-D and equal length")
    s = np.sum(np.stack(vectors, axis=0), axis=0)
    n = np.linalg.norm(s)
    return s if n == 0.0 else s / n


def _cosine(a: VectorF64, b: VectorF64) -> float:
    denom = _safe_norm(a) * _safe_norm(b)
    return float(a @ b / denom)


def unbundle(
    bundled: VectorF64, item_memory: dict[str, VectorF64], threshold: float = 0.3
) -> list[tuple[str, float]]:
    """Retrieve components from a bundled vector using cosine similarity threshold.

    Args:
        bundled: 1-D array
        item_memory: mapping name -> prototype vector (1-D)
        threshold: minimum cosine similarity to include
    Returns:
        list of (label, similarity) sorted by similarity desc
    """
    if bundled.ndim != 1:
        raise ValueError("bundled must be a 1-D array")
    out: list[tuple[str, float]] = []
    for label, proto in item_memory.items():
        if proto.ndim != 1 or proto.shape[0] != bundled.shape[0]:
            raise ValueError("prototypes must be 1-D and equal length to bundled")
        sim = _cosine(bundled, proto)
        if sim >= threshold:
            out.append((label, float(sim)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
