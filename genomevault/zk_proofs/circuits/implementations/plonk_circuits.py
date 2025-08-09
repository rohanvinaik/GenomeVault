from __future__ import annotations

import numpy as np


def normalize_methylation(beta_values: np.ndarray) -> np.ndarray:
    """Z-score normalize beta-values along the last axis."""
    beta = np.asarray(beta_values, dtype=np.float32)
    mu = beta.mean(axis=-1, keepdims=True)
    sd = beta.std(axis=-1, keepdims=True) + 1e-8
    return (beta - mu) / sd
