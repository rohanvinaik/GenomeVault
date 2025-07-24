"""
Clinical validation module for GenomeVault
Refactored and consolidated implementation
"""

from .circuits import ClinicalBiomarkerCircuit, DiabetesRiskCircuit, create_circuit
from .core import ClinicalValidator
from .data_sources import NHANESDataSource, PimaDataSource
from .proofs import CircuitType, ProofData, verify_proof

# Legacy imports for backward compatibility
from .zk_wrapper import ZKProver

__all__ = [
    "ClinicalValidator",
    "DiabetesRiskCircuit",
    "ClinicalBiomarkerCircuit",
    "create_circuit",
    "ProofData",
    "CircuitType",
    "verify_proof",
    "NHANESDataSource",
    "PimaDataSource",
    "ZKProver",  # Legacy
]
