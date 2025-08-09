from __future__ import annotations

import numpy as np

D = 10_000


def encode(X: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Encode feature matrix into hypervectors.

    Args:
        X: Input feature matrix (n_samples x n_features)
        seed: Random seed for reproducibility

    Returns:
        Hypervector matrix (n_samples x D)
    """
    if X.ndim != 2 or X.size == 0:
        raise ValueError("X must be a non-empty 2D array")

    n_samples, n_features = X.shape
    np.random.seed(seed)

    # Create random projection matrix (sparse for efficiency)
    sparsity = 0.1
    projection = np.random.randn(n_features, D) * np.sqrt(1 / (n_features * sparsity))
    mask = np.random.random((n_features, D)) < sparsity
    projection *= mask

    # Project and binarize
    V = X @ projection
    return np.sign(V + 1e-10)  # Avoid zero values


def bundle(vectors: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Bundle multiple hypervectors into one.

    Args:
        vectors: Matrix of hypervectors to bundle
        normalize: Whether to normalize the output

    Returns:
        Bundled hypervector
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")

    bundled = np.sum(vectors, axis=0)
    if normalize:
        bundled = np.sign(bundled + 1e-10)
    return bundled


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between hypervectors.

    Args:
        a: First hypervector
        b: Second hypervector

    Returns:
        Similarity score in [0, 1]
    """
    if a.shape != b.shape:
        raise ValueError("Hypervectors must have the same shape")

    # Cosine similarity for binary vectors
    dot = np.dot(a.flatten(), b.flatten())
    norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
    sim = dot / norm
    return (sim + 1) / 2  # Map from [-1, 1] to [0, 1]
