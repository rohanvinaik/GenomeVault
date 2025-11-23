"""Sparse Projection module."""

from __future__ import annotations

from typing import cast

from numpy.typing import NDArray
import numpy as np
import torch

from genomevault.hypervector.types import MatrixF32


def sparse_random_matrix(rows: int, cols: int, sparsity: float = 0.1) -> torch.Tensor:
    """Achlioptas sparse projection in {-1,0,+1}, scaled."""
    probs = [sparsity / 2, 1 - sparsity, sparsity / 2]
    vals: NDArray[np.float32] = np.random.choice(
        [-1.0, 0.0, 1.0], size=(rows, cols), p=probs
    ).astype(np.float32, copy=False)
    mat = torch.from_numpy(vals)
    if sparsity > 0:
        mat = mat / np.sqrt(sparsity * cols)
    return mat


class SparseRandomProjection:
    """Compatibility class for unified encoder."""

    def __init__(self, n_components: int, density: float = 0.1, seed: int = None):
        """Initialize instance.

        Args:
            n_components: N components.
            density: Density.
            seed: Seed.
        """
        self.n_components = n_components
        self.density = density
        if seed is not None:
            np.random.seed(seed)
        self._matrix = None

    def fit(self, n_features: int):
        """Fit.

        Args:
            n_features: Feature array.

        Returns:
            Operation result.
        """
        self._matrix = sparse_random_matrix(self.n_components, n_features, self.density)
        return self

    def transform(self, X: MatrixF32) -> MatrixF32:
        """Transform.

        Args:
            X: X.

        Returns:
            Operation result.

        Raises:
            ValueError: When operation fails.
        """
        if self._matrix is None:
            raise ValueError("Must call fit() before transform()")
        # Explicit type conversion with no-copy when possible
        X_tensor = torch.from_numpy(X.astype(np.float32, copy=False))
        result = torch.matmul(X_tensor, self._matrix.T)
        # Cast the result back with explicit type annotation
        out: MatrixF32 = cast(MatrixF32, result.numpy())
        return out
