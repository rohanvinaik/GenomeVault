"""Calibrators module."""

from __future__ import annotations

from numpy.typing import NDArray
import numpy as np


class PlattCalibrator:
    """Logistic regression (1D) via IRLS to map scores -> probability."""

    def __init__(self, max_iter: int = 100, tol: float = 1e-8, reg: float = 1e-6):
        """Initialize instance.

        Args:
            max_iter: Max iter.
            tol: Tol.
            reg: Reg.
        """
        self.coef_: float | None = None
        self.intercept_: float | None = None
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg = float(reg)

    @staticmethod
    def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
        """sigmoid.
        Args:        z: Parameter value.
        Returns:
            NDArray[np.float64]"""
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, y_true: NDArray[np.float64], y_score: NDArray[np.float64]) -> PlattCalibrator:
        """Fit.

        Args:
            y_true: Y true.
            y_score: Y score.

        Returns:
            PlattCalibrator instance.
        """
        y = np.asarray(y_true, dtype=np.float64)
        x = np.asarray(y_score, dtype=np.float64)
        X = np.c_[np.ones_like(x), x]  # intercept, slope
        w = np.zeros(2)  # [b, a]
        for _ in range(self.max_iter):
            z = X @ w
            p = self._sigmoid(z)
            # IRLS: W = diag(p*(1-p)); update: w += (X^T W X + reg I)^-1 X^T (y-p)
            W = p * (1.0 - p) + 1e-12
            grad = X.T @ (y - p) - self.reg * w
            H = (X.T * W) @ X + self.reg * np.eye(2)
            delta = np.linalg.solve(H, grad)
            w_new = w + delta
            if np.linalg.norm(delta) < self.tol:
                w = w_new
                break
            w = w_new
        self.intercept_, self.coef_ = float(w[0]), float(w[1])
        return self

    def predict_proba(self, y_score: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict proba.

        Args:
            y_score: Y score.

        Returns:
            Operation result.
        """
        assert self.coef_ is not None and self.intercept_ is not None, "fit first"
        x = np.asarray(y_score, dtype=np.float64)
        z = self.intercept_ + self.coef_ * x
        return self._sigmoid(z)


class IsotonicCalibrator:
    """Isotonic regression via pair-adjacent violators (PAV) algorithm."""

    def __init__(self):
        """Initialize instance."""
        self.x_: NDArray[np.float64] | None = None  # breakpoints (sorted scores)
        self.y_: NDArray[np.float64] | None = None  # fitted (piecewise-constant) probabilities

    def fit(self, y_true: NDArray[np.float64], y_score: NDArray[np.float64]) -> IsotonicCalibrator:
        """Fit.

        Args:
            y_true: Y true.
            y_score: Y score.

        Returns:
            IsotonicCalibrator instance.
        """
        y = np.asarray(y_true, dtype=np.float64)
        x = np.asarray(y_score, dtype=np.float64)
        order = np.argsort(x)
        x_s = x[order]
        y_s = y[order]

        # PAV
        g = y_s.copy()
        w = np.ones_like(g)
        i = 0
        while i < len(g) - 1:
            if g[i] <= g[i + 1]:
                i += 1
                continue
            # merge blocks until monotone
            j = i
            while j >= 0 and g[j] > g[j + 1]:
                new_w = w[j] + w[j + 1]
                new_g = (w[j] * g[j] + w[j + 1] * g[j + 1]) / new_w
                g[j] = new_g
                w[j] = new_w
                # remove position j+1 by shifting left
                g = np.delete(g, j + 1)
                w = np.delete(w, j + 1)
                x_s = np.delete(x_s, j + 1)
                y_s = np.delete(y_s, j + 1)
                j -= 1
            i = max(j, 0)

        self.x_ = x_s
        self.y_ = g
        return self

    def predict_proba(self, y_score: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict proba.

        Args:
            y_score: Y score.

        Returns:
            Operation result.
        """
        assert self.x_ is not None and self.y_ is not None, "fit first"
        x = np.asarray(y_score, dtype=np.float64)
        # piecewise-constant: for each x, find last breakpoint <= x
        idx = np.searchsorted(self.x_, x, side="right") - 1
        idx = np.clip(idx, 0, len(self.x_) - 1)
        return self.y_[idx]


def fit_and_calibrate(
    y_true: NDArray[np.float64],
    y_score: NDArray[np.float64],
    method: str = "platt" """Fit and calibrate.""",
) -> tuple[NDArray[np.float64], object]:
    """Fit and calibrate.

    Args:
        y_true: Y true.
        y_score: Y score.
        method: Method.

    Returns:
        Operation result.

    Raises:
        ValueError: When operation fails.
    """
    method = (method or "").lower()
    if method == "platt":
        cal = PlattCalibrator().fit(y_true, y_score)
    elif method == "isotonic":
        cal = IsotonicCalibrator().fit(y_true, y_score)
    elif method in ("none", "identity"):

        class Identity:
            """Identity implementation."""

            def predict_proba(self, v) -> None:
                """Predict proba.

                Args:
                    v: V.

                Returns:
                    Operation result.
                """
                return np.asarray(v, dtype=np.float64)

        cal = Identity()
    else:
        raise ValueError("unknown calibration method: %s" % method)
    return cal.predict_proba(y_score), cal
