"""Profile module."""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Profile dataframe.

    Args:
        df: Df.

    Returns:
        Operation result.
    """
    out: dict[str, Any] = {"row_count": int(df.shape[0]), "columns": {}}
    for c in df.columns:
        s = df[c]
        out["columns"][c] = {
            "dtype": str(s.dtype),
            "nulls": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
            "min": float(s.min()) if s.dtype.kind in "if" and s.size else None,
            "max": float(s.max()) if s.dtype.kind in "if" and s.size else None,
        }
    return out
