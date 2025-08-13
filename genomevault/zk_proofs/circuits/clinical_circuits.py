"""
Clinical circuits module - placeholder for backward compatibility.
Actual implementation moved to clinical subdirectory.

"""
import warnings
class ClinicalBiomarkerCircuit:
    """Placeholder - use genomevault.zk_proofs.circuits.clinical instead"""

    def __init__(self):
        """Initialize instance."""
        warnings.warn(
            "ClinicalBiomarkerCircuit moved to genomevault.zk_proofs.circuits.clinical",
            DeprecationWarning,
            stacklevel=2,
        )


class DiabetesRiskCircuit:
    """Placeholder - use genomevault.zk_proofs.circuits.clinical instead"""

    def __init__(self):
        """Initialize instance."""
        warnings.warn(
            "DiabetesRiskCircuit moved to genomevault.zk_proofs.circuits.clinical",
            DeprecationWarning,
            stacklevel=2,
        )


class ProofData:
    """Placeholder - use genomevault.zk_proofs.circuits.clinical instead"""

    def __init__(self):
        """Initialize instance."""
        warnings.warn(
            "ProofData moved to genomevault.zk_proofs.circuits.clinical",
            DeprecationWarning,
            stacklevel=2,
        )


class ClinicalCircuit:
    """Placeholder - use genomevault.zk_proofs.circuits.clinical instead"""

    def __init__(self):
        """Initialize instance."""
        warnings.warn(
            "ClinicalCircuit moved to genomevault.zk_proofs.circuits.clinical",
            DeprecationWarning,
            stacklevel=2,
        )


def create_circuit(*args, **kwargs):
    """Placeholder - use genomevault.zk_proofs.circuits.clinical instead"""
    warnings.warn(
        "genomevault.zk_proofs.circuits.clinical_circuits is deprecated. "
        "Use genomevault.zk_proofs.circuits.clinical instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return None


# Issue deprecation warning on import
warnings.warn(
    "genomevault.zk_proofs.circuits.clinical_circuits is deprecated. "
    "Use genomevault.zk_proofs.circuits.clinical instead.",
    DeprecationWarning,
    stacklevel=2,
)
