from __future__ import annotations

import numpy as np

from genomevault.core.exceptions import ProjectionError


class OrthogonalProjection:
    """Orthogonal projection preserving inner products/angles approximately."""

    def __init__(self, n_components: int, seed: int | None = None) -> None:
        if not isinstance(n_components, int) or n_components <= 0:
            raise ProjectionError(
                "n_components must be a positive integer",
                context={"n_components": n_components},
            )
        self.n_components = int(n_components)
        self.rng = np.random.default_rng(seed)
        self._P: np.ndarray | None = None  # shape: (n_components, n_features)
        self._n_features: int | None = None

    def fit(self, n_features: int) -> OrthogonalProjection:
        if not isinstance(n_features, int) or n_features <= 0:
            raise ProjectionError(
                "n_features must be a positive integer",
                context={"n_features": n_features},
            )
        if n_features < self.n_components:
            raise ProjectionError(
                "n_features must be >= n_components",
                context={"n_features": n_features, "n_components": self.n_components},
            )
        # Gaussian random matrix, then QR
        A = self.rng.standard_normal(
            (n_features, self.n_components)
        )  # (n_features, n_components)
        Q, _ = np.linalg.qr(
            A, mode="reduced"
        )  # Q: (n_features, n_components) with orthonormal columns
        self._P = Q.T  # (n_components, n_features)
        self._n_features = n_features
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._P is None or self._n_features is None:
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
        # P has shape (n_components, n_features); projecting is X @ P.T == X @ Q
        return X @ self._P.T
