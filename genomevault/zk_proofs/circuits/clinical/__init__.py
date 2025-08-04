"""
Clinical circuits integration for the ZK proofs module.
Provides backward compatibility links to the refactored clinical validation circuits.
"""

# Import from the refactored clinical validation module
from clinical_validation.circuits import (ClinicalBiomarkerCircuit,
                                          DiabetesRiskCircuit)
from clinical_validation.proofs import ProofData

# Maintain backward compatibility
__all__ = ["ClinicalBiomarkerCircuit", "DiabetesRiskCircuit", "ProofData"]
