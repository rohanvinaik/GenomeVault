"""
Zero-knowledge proof implementations for zk.

DEPRECATION NOTICE: This module is being phased out.
For new code, use genomevault.zk_proofs.Prover instead.
"""
import warnings
warnings.warn(
    "genomevault.zk is deprecated. "
    "For ZK proofs, use genomevault.zk_proofs.Prover instead.",
    DeprecationWarning,
    stacklevel=2
)

from .models import ProofCreateRequest, ProofVerifyRequest
from .engine import Proof, ZKProofEngine
from .real_engine import RealProof, RealZKEngine
from .proof import ProofResult, ProofGenerator

__all__ = [
    "Proof",
    "ProofCreateRequest",
    "ProofGenerator",
    "ProofResult",
    "ProofVerifyRequest",
    "RealProof",
    "RealZKEngine",
    "ZKProofEngine",
]
