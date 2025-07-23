"""
Clinical validation module for GenomeVault
Refactored and consolidated implementation
"""

from .core import ClinicalValidator
from .circuits import DiabetesRiskCircuit, ClinicalBiomarkerCircuit, create_circuit
from .proofs import ProofData, CircuitType, verify_proof
from .data_sources import NHANESDataSource, PimaDataSource

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
