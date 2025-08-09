from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from .base_circuits import BaseCircuit


@dataclass
class TrainingProofCircuit(BaseCircuit):
    gradients_sum: float
    threshold: float

    def public_statement(self) -> Dict[str, Any]:
        return {
            "predicate": "sum(gradients) > threshold",
            "threshold": float(self.threshold),
        }

    def witness(self) -> Dict[str, Any]:
        return {"sum_gradients": float(self.gradients_sum)}

    def holds(self) -> bool:
        return self.gradients_sum > self.threshold
