"""
Experimental Zero-Knowledge proof circuits and advanced protocols.

EXPERIMENTAL: These ZK implementations are research prototypes.
Security properties have not been formally verified.
DO NOT use in production without thorough security audit.
"""

from .stark_prover import (
    FiatShamirTranscript,
    STARKProof,
    STARKProver,
    PostQuantumVerifier,
)
from .catalytic_proof import CatalyticProof, CatalyticSpace, CatalyticProofEngine
from .post_quantum import PQEngine, prove, verify
from .recursive_snark import RecursiveProof, RecursiveSNARKProver

__all__ = [
    "CatalyticProof",
    "CatalyticProofEngine",
    "CatalyticSpace",
    "FiatShamirTranscript",
    "PQEngine",
    "PostQuantumVerifier",
    "RecursiveProof",
    "RecursiveSNARKProver",
    "STARKProof",
    "STARKProver",
    "prove",
    "verify",
]
