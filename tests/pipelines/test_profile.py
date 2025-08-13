import pandas as pd

from genomevault.pipelines.profile import profile_dataframe


def test_profile_dataframe_basic():
    """Test profile dataframe basic.
    Returns:
        Result of the operation."""
    df = pd.DataFrame({"a": [1, 2, 2], "b": ["x", None, "y"]})
    prof = profile_dataframe(df)
    assert prof["row_count"] == 3
    assert "a" in prof["columns"] and "b" in prof["columns"]
