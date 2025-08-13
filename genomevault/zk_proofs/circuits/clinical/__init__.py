"""Clinical validation circuits module."""

from genomevault.zk_proofs.circuits.clinical_circuits import (
    ClinicalBiomarkerCircuit,
    DiabetesRiskCircuit,
    ProofData,
)

# Maintain backward compatibility
__all__ = ["ClinicalBiomarkerCircuit", "DiabetesRiskCircuit", "ProofData"]
