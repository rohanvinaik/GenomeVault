"""
Proof models and verification logic.
"""

from .models import CircuitConfig, CircuitType, ComparisonType, ProofData
from .verifier import ProofVerifier, verify_proof

__all__ = [
    "ProofData",
    "CircuitConfig",
    "CircuitType",
    "ComparisonType",
    "ProofVerifier",
    "verify_proof",
]
