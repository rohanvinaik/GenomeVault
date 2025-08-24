"""Optimized ZK proof circuits with performance improvements."""

from .diabetes_risk_alert import (
    OptimizedDiabetesRiskCircuit,
    ConstraintBatch,
    register_optimized_circuits
)

__all__ = [
    'OptimizedDiabetesRiskCircuit',
    'ConstraintBatch',
    'register_optimized_circuits'
]