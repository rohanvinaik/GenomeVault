"""Common module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def validate_features(X: np.ndarray) -> None:
    """Validate feature matrix.

    Args:
        X: Feature matrix to validate

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")


def process(dataset: pd.DataFrame | Path, config: dict) -> np.ndarray:
    """Process dataset into normalized feature matrix.

    Args:
        dataset: Input data as DataFrame or path to file
        config: Processing configuration (unused in MVP)

    Returns:
        Normalized numeric feature matrix
    """
    # Load data if path provided
    if isinstance(dataset, Path):
        dataset = pd.read_csv(dataset)

    # Select numeric columns
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("no numeric columns")

    X = dataset[numeric_cols].values

    # Standardize
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    Xz = (X - mu) / sd
    validate_features(Xz)
    return Xz
