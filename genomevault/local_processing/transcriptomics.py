from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd


def process(dataset: Union[pd.DataFrame, Path], config: Dict[str, Any]) -> np.ndarray:
    """Process transcriptomic data and return normalized expression matrix.

    Args:
        dataset: Input data as DataFrame or path to file
        config: Processing configuration

    Returns:
        Normalized expression matrix (n_samples x n_genes)
    """
    # Load data if path provided
    if isinstance(dataset, Path):
        dataset = pd.read_csv(dataset)

    # Select numeric columns (gene expression values)
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataset")

    # Extract expression data
    X = dataset[numeric_cols].values

    # TPM normalization (Transcripts Per Million)
    row_sums = np.sum(X, axis=1, keepdims=True) + 1e-9
    X_tpm = (X / row_sums) * 1e6

    # Log transform
    X_log = np.log2(X_tpm + 1)

    # Standardize features
    mean = np.mean(X_log, axis=0)
    std = np.std(X_log, axis=0) + 1e-9
    X_normalized = (X_log - mean) / std

    # Ensure finite values
    X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return X_normalized


def validate_features(X: np.ndarray) -> bool:
    """Validate expression matrix."""
    return X.ndim == 2 and np.all(np.isfinite(X))
