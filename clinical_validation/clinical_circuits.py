"""
DEPRECATED: Use the new modular circuits structure.

This file has been refactored into:
- clinical_validation.circuits.diabetes
- clinical_validation.circuits.biomarkers
- clinical_validation.circuits.base
- clinical_validation.proofs.models

Import from clinical_validation.circuits instead.
"""

import warnings

# Import from the new structure
from .circuits import (
    BaseCircuit,
    ClinicalBiomarkerCircuit,
    DiabetesRiskCircuit,
    create_circuit,
)
from .proofs import CircuitType, ProofData

# Issue deprecation warning
warnings.warn(
    "clinical_validation.clinical_circuits is deprecated. "
    "Import directly from clinical_validation.circuits instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export for backward compatibility
__all__ = [
    "DiabetesRiskCircuit",
    "ClinicalBiomarkerCircuit",
    "BaseCircuit",
    "ProofData",
    "create_circuit",
    "CircuitType",
]
