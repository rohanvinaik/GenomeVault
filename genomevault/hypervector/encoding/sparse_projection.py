from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from genomevault.core.exceptions import ProjectionError

if TYPE_CHECKING:
    pass


class SparseRandomProjection:
    """Sparse random projection for mapping features to hypervector space.

    Implementation avoids heavy dependencies; stores per-component sparse indices and signs.
    For testability, dimensions in tests are small. Production tiers (10k/15k/20k) can be used later.
    """

    def __init__(
        self, n_components: int, density: float = 0.1, seed: int | None = None
    ) -> None:
        if not isinstance(n_components, int) or n_components <= 0:
            raise ProjectionError(
                "n_components must be a positive integer",
                context={"n_components": n_components},
            )
        if not (0.0 < float(density) <= 1.0):
            raise ProjectionError(
                "density must be in (0, 1]", context={"density": density}
            )
        self.n_components = int(n_components)
        self.density = float(density)
        self.rng = np.random.default_rng(seed)
        self._indices: list[np.ndarray] | None = None
        self._signs: list[np.ndarray] | None = None
        self._n_features: int | None = None
        self._scale: float = 1.0

    def fit(self, n_features: int) -> SparseRandomProjection:
        """Create sparse pattern per component.

        Each component selects k = max(1, round(density * n_features)) unique feature indices
        with random ±1 signs. A scaling of 1/sqrt(k) is applied at transform-time.
        """
        if not isinstance(n_features, int) or n_features <= 0:
            raise ProjectionError(
                "n_features must be a positive integer",
                context={"n_features": n_features},
            )

        k = max(1, int(round(self.density * n_features)))
        indices: list[np.ndarray] = []
        signs: list[np.ndarray] = []
        for _ in range(self.n_components):
            idx = self.rng.choice(n_features, size=k, replace=False)
            s = self.rng.choice([-1.0, 1.0], size=k)
            indices.append(idx.astype(np.int64, copy=False))
            signs.append(s.astype(np.float64, copy=False))
        self._indices = indices
        self._signs = signs
        self._n_features = n_features
        self._scale = 1.0 / np.sqrt(k)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._indices is None or self._signs is None or self._n_features is None:
            raise ProjectionError("fit() must be called before transform()")
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ProjectionError(
                "X must be a 2-D numpy array",
                context={"ndim": getattr(X, "ndim", None)},
            )
        if X.shape[1] != self._n_features:
            raise ProjectionError(
                "X has mismatched n_features",
                context={
                    "X_n_features": int(X.shape[1]),
                    "fit_n_features": int(self._n_features),
                },
            )
        n_samples = X.shape[0]
        Y = np.empty((n_samples, self.n_components), dtype=np.float64)
        # Compute each component as a sparse weighted sum
        for i, (idx, sgn) in enumerate(zip(self._indices, self._signs)):
            Xi = X[:, idx]  # shape (n_samples, k)
            Y[:, i] = (Xi * sgn).sum(axis=1) * self._scale
        return Y
