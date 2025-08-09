from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomevault.local_processing.common import process, validate_features


class TestLocalProcessing:
    def test_process_dataframe(self):
        """Test processing of DataFrame."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [2.5, 3.5, 4.5, 5.5, 6.5],
                "c": ["x", "y", "z", "w", "v"],  # Non-numeric
            }
        )
        X = process(df, {})
        assert X.shape == (5, 2)  # Only numeric columns
        assert np.allclose(X.mean(axis=0), 0)  # Standardized
        assert np.allclose(X.std(axis=0), 1)  # Unit variance

    def test_validate_features(self):
        """Test feature validation."""
        X = np.random.randn(10, 5)
        validate_features(X)  # Should pass

        with pytest.raises(ValueError):
            validate_features(np.array([1, 2, 3]))  # 1D

        with pytest.raises(ValueError):
            X_bad = np.random.randn(5, 3)
            X_bad[0, 0] = np.inf
            validate_features(X_bad)  # Contains inf

    def test_no_numeric_columns(self):
        """Test error when no numeric columns."""
        df = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
        with pytest.raises(ValueError, match="no numeric columns"):
            process(df, {})
