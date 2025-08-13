"""
Experimental Zero-Knowledge proof circuits and advanced protocols.

EXPERIMENTAL: These ZK implementations are research prototypes.
Security properties have not been formally verified.
DO NOT use in production without thorough security audit.
"""

import warnings

warnings.warn(
    "Experimental ZK circuits are not audited for production use. "
    "These are research implementations that may have security vulnerabilities. "
    "DO NOT use for sensitive data without formal verification.",
    FutureWarning,
    stacklevel=2,
)

# Import with safety checks
from .catalytic_proof import CatalyticProofEngine
from .recursive_snark import RecursiveProof, RecursiveSNARKProver
from .stark_prover import PostQuantumVerifier, STARKProver

# Post-quantum is also experimental
try:
    from .post_quantum import PostQuantumZK
except ImportError:
    PostQuantumZK = None

__all__ = [
    "CatalyticProofEngine",
    "PostQuantumVerifier",
    "RecursiveProof",
    "RecursiveSNARKProver",
    "STARKProver",
    "PostQuantumZK",
]
