"""
ZK Proof Circuits for GenomeVault
"""
from .base_circuits import BaseCircuit
from .clinical_circuits import DiabetesRiskCircuit, ClinicalBiomarkerCircuit

# Make commonly used circuits easily accessible
__all__ = [
    'BaseCircuit',
    'DiabetesRiskCircuit', 
    'ClinicalBiomarkerCircuit'
]
