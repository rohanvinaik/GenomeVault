"""
DEPRECATED: Clinical circuits module - use clinical_validation.circuits instead.

This file provides backward compatibility for existing code.
All new code should import from clinical_validation.circuits directly.
"""

import warnings

# Import from the new refactored location
from clinical_validation.circuits import (BaseCircuit,
                                          ClinicalBiomarkerCircuit,
                                          DiabetesRiskCircuit, create_circuit)
from clinical_validation.proofs import CircuitType, ProofData

# Issue deprecation warning
warnings.warn(
    "genomevault.zk_proofs.circuits.clinical_circuits is deprecated. "
    "Use clinical_validation.circuits instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward compatibility
__all__ = [
    "BaseCircuit",
    "CircuitType",
    "ClinicalBiomarkerCircuit",
    "DiabetesRiskCircuit",
    "ProofData",
    "create_circuit",
]
