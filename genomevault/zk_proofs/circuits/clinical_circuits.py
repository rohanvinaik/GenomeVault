##from clinical_validation.circuits import (  # Import from the new refactored location; DEPRECATED:, All, BaseCircuit, CircuitType, Clinical,
(
    ClinicalBiomarkerCircuit,
    DiabetesRiskCircuit,
    ProofData,
    This,
    """, -,
    backward, circuits, clinical_validation.circuits,
    clinical_validation.proofs, code, code., compatibility, create_circuit,
    directly., existing, file, for, from, import, instead., module, new,
    provides, should, use)

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
""",
)
