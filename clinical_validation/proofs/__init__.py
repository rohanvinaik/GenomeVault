"""
Proof models and verification logic.
"""

from .models import ProofData, CircuitConfig, CircuitType, ComparisonType
from .verifier import ProofVerifier, verify_proof

__all__ = [
    'ProofData',
    'CircuitConfig', 
    'CircuitType',
    'ComparisonType',
    'ProofVerifier',
    'verify_proof'
]
