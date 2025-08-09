from __future__ import annotations

import pytest

from genomevault.federated.aggregate import aggregate


class TestFederated:
    def test_aggregate_empty(self):
        """Test aggregation of empty list."""
        result = aggregate([])
        assert result == {"count": 0, "means": {}}

    def test_aggregate_numeric(self):
        """Test aggregation of numeric fields."""
        stats = [
            {"loss": 0.5, "accuracy": 0.8, "name": "client1"},
            {"loss": 0.3, "accuracy": 0.9, "name": "client2"},
            {"loss": 0.4, "accuracy": 0.85},  # Missing name
        ]
        result = aggregate(stats)
        assert result["count"] == 3
        assert result["means"]["loss"] == pytest.approx(0.4)
        assert result["means"]["accuracy"] == pytest.approx(0.85)
        assert "name" not in result["means"]  # Non-numeric

    def test_aggregate_mixed_types(self):
        """Test aggregation handles mixed types correctly."""
        stats = [
            {"value": 10, "flag": True},
            {"value": 20, "flag": False},
        ]
        result = aggregate(stats)
        assert result["means"]["value"] == 15
        # Booleans are treated as numeric (0/1)
        assert "flag" in result["means"]
