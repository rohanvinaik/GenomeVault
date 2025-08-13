from __future__ import annotations

import numpy as np

from genomevault.hdc.core import D, bundle, encode, similarity


class TestHDC:
    """TestHDC class implementation."""

    def test_encode_basic(self):
        """Test basic encoding functionality."""
        X = np.random.randn(5, 10)
        V = encode(X, seed=42)
        assert V.shape == (5, D)
        assert np.all(np.abs(V) == 1)  # Binary vectors

    def test_encode_deterministic(self):
        """Test encoding is deterministic with seed."""
        X = np.random.randn(3, 8)
        V1 = encode(X, seed=123)
        V2 = encode(X, seed=123)
        np.testing.assert_array_equal(V1, V2)

    def test_bundle(self):
        """Test bundling operation."""
        vectors = np.array([[1, -1, 1], [-1, 1, 1], [1, 1, -1]])
        bundled = bundle(vectors)
        assert bundled.shape == (3,)
        assert np.all(np.abs(bundled) == 1)

    def test_similarity(self):
        """Test similarity computation."""
        a = np.array([1, -1, 1, -1])
        b = np.array([1, -1, 1, -1])
        c = np.array([-1, 1, -1, 1])

        assert similarity(a, b) == 1.0  # Identical
        assert 0 <= similarity(a, c) <= 1  # In range
