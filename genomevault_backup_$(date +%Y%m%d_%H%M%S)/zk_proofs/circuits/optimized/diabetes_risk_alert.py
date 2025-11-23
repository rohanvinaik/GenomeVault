"""Optimized diabetes risk alert circuit with batch constraint generation."""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import hashlib


@dataclass
class ConstraintBatch:
    """Batch of constraints for efficient generation."""

    constraints: List[Dict[str, Any]]

    def apply_to_circuit(self, circuit):
        """Apply all constraints in batch."""
        # Pre-allocate memory for all constraints
        circuit.reserve_constraints(len(self.constraints))

        # Add constraints in single operation
        for c in self.constraints:
            circuit.add_constraint_direct(c)


class OptimizedDiabetesRiskCircuit:
    """Optimized diabetes risk circuit with batch processing."""

    def __init__(self):
        self.constraint_cache = {}
        self.batch_size = 100

    def generate_constraint_batch(
        self, num_constraints: int, risk_factors: List[float]
    ) -> ConstraintBatch:
        """Generate constraints in batches for efficiency."""

        # Check cache first
        cache_key = hashlib.md5(
            f"{num_constraints}:{hash(tuple(risk_factors))}".encode()
        ).hexdigest()

        if cache_key in self.constraint_cache:
            return self.constraint_cache[cache_key]

        constraints = []

        # Batch generation - vectorized operations
        for i in range(0, num_constraints, self.batch_size):
            batch_end = min(i + self.batch_size, num_constraints)
            batch_factors = risk_factors[i:batch_end] if i < len(risk_factors) else []

            # Vectorized constraint generation
            batch_constraints = self._generate_batch_constraints(batch_factors)
            constraints.extend(batch_constraints)

        result = ConstraintBatch(constraints)
        self.constraint_cache[cache_key] = result
        return result

    def _generate_batch_constraints(self, factors: List[float]) -> List[Dict]:
        """Generate constraints for a batch of factors."""
        if not factors:
            return []

        # Use NumPy for vectorized operations
        factors_array = np.array(factors)

        # Vectorized threshold checks
        high_risk = factors_array > 0.7
        medium_risk = (factors_array > 0.3) & (factors_array <= 0.7)

        constraints = []
        for idx, (hr, mr, factor) in enumerate(zip(high_risk, medium_risk, factors)):
            if hr:
                constraint = {"type": "high_risk", "factor": factor, "threshold": 0.7, "index": idx}
            elif mr:
                constraint = {
                    "type": "medium_risk",
                    "factor": factor,
                    "threshold": 0.3,
                    "index": idx,
                }
            else:
                constraint = {"type": "low_risk", "factor": factor, "threshold": 0.3, "index": idx}
            constraints.append(constraint)

        return constraints

    def generate_witness(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate witness with optimized constraint generation."""
        risk_factors = inputs.get("risk_factors", [])
        num_constraints = len(risk_factors) * 3  # 3 constraints per factor

        # Batch constraint generation
        constraint_batch = self.generate_constraint_batch(num_constraints, risk_factors)

        # Generate witness
        witness = {
            "risk_score": self._calculate_risk_score(risk_factors),
            "constraints": constraint_batch.constraints,
            "num_factors": len(risk_factors),
        }

        return witness

    def _calculate_risk_score(self, factors: List[float]) -> float:
        """Calculate overall risk score."""
        if not factors:
            return 0.0

        # Weighted average with emphasis on high values
        factors_array = np.array(factors)
        weights = np.exp(factors_array * 2)  # Exponential weighting
        weighted_score = np.average(factors_array, weights=weights)

        return float(weighted_score)


def register_optimized_circuits():
    """Register optimized circuit implementations."""
    from genomevault.zk_proofs.circuit_registry import CircuitRegistry

    registry = CircuitRegistry.get_instance()
    registry.register("diabetes_risk_alert_optimized", OptimizedDiabetesRiskCircuit)
