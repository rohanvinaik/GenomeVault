"""Training Proof module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .base_circuits import BaseCircuit


@dataclass
class TrainingProofCircuit(BaseCircuit):
    """Data container for trainingproofcircuit information."""

    gradients_sum: float
    threshold: float

    def public_statement(self) -> Dict[str, Any]:
        """Public statement.

        Returns:
            Operation result.
        """
        return {
            "predicate": "sum(gradients) > threshold",
            "threshold": float(self.threshold),
        }

    def witness(self) -> Dict[str, Any]:
        """Witness.

        Returns:
            Operation result.
        """
        return {"sum_gradients": float(self.gradients_sum)}

    def holds(self) -> bool:
        """Holds.

        Returns:
            Boolean result.
        """
        return self.gradients_sum > self.threshold
