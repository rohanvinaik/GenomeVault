"""Diabetes-specific ZK proof circuits."""
from typing import Any, Dict

from ..base_circuits import BaseCircuit, ComparisonCircuit, RangeProofCircuit

class DiabetesRiskCircuit(BaseCircuit):
    """Circuit for privacy-preserving diabetes risk assessment."""

    def __init__(self):
        """Initialize instance."""
        self.glucose_level = 0
        self.hba1c = 0.0
        self.risk_score = 0.0
        self.glucose_comparison = ComparisonCircuit()
        self.hba1c_range = RangeProofCircuit(bit_width=16)

    def public_statement(self) -> Dict[str, Any]:
        """Public outputs of diabetes risk assessment."""
        return {
            "risk_category": self._get_risk_category(),
            "requires_intervention": self.risk_score > 0.7,
        }

    def witness(self) -> Dict[str, Any]:
        """Private inputs for diabetes risk assessment."""
        return {
            "glucose_level": self.glucose_level,
            "hba1c": self.hba1c,
            "risk_score": self.risk_score,
        }

    def assess_risk(self, glucose: int, hba1c: float) -> float:
        """Assess diabetes risk from glucose and HbA1c levels."""
        self.glucose_level = glucose
        self.hba1c = hba1c

        # Simple risk calculation
        glucose_risk = 0.0
        if glucose > 126:  # Diabetic range
            glucose_risk = 0.8
        elif glucose > 100:  # Pre-diabetic range
            glucose_risk = 0.5
        else:
            glucose_risk = 0.1

        hba1c_risk = 0.0
        if hba1c > 6.5:  # Diabetic range
            hba1c_risk = 0.8
        elif hba1c > 5.7:  # Pre-diabetic range
            hba1c_risk = 0.5
        else:
            hba1c_risk = 0.1

        # Combine risks
        self.risk_score = (glucose_risk + hba1c_risk) / 2
        return self.risk_score

    def _get_risk_category(self) -> str:
        """Get risk category from score."""
        if self.risk_score > 0.7:
            return "HIGH"
        elif self.risk_score > 0.4:
            return "MODERATE"
        else:
            return "LOW"


class GlucoseMonitoringCircuit(BaseCircuit):
    """Circuit for privacy-preserving glucose monitoring."""

    def __init__(self):
        """Initialize instance."""
        self.readings = []
        self.average = 0.0
        self.in_range_count = 0

    def public_statement(self) -> Dict[str, Any]:
        """Public outputs showing compliance without revealing values."""
        return {
            "average_in_range": 70 <= self.average <= 180,
            "compliance_rate": self.in_range_count / max(len(self.readings), 1),
        }

    def witness(self) -> Dict[str, Any]:
        """Private glucose readings."""
        return {"readings": self.readings, "average": self.average}

    def add_reading(self, glucose: int) -> None:
        """Add a glucose reading."""
        self.readings.append(glucose)
        if 70 <= glucose <= 180:
            self.in_range_count += 1

        # Update average
        if self.readings:
            self.average = sum(self.readings) / len(self.readings)


# Export public API
__all__ = [
    "DiabetesRiskCircuit",
    "GlucoseMonitoringCircuit",
]
