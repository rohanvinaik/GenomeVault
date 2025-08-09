"""Clinical validation circuits module."""

# Import from the refactored clinical validation module
from genomevault.clinical_validation.circuits import (
    ClinicalBiomarkerCircuit,
    DiabetesRiskCircuit,
    ProofData,
)

# Maintain backward compatibility
__all__ = ["ClinicalBiomarkerCircuit", "DiabetesRiskCircuit", "ProofData"]
