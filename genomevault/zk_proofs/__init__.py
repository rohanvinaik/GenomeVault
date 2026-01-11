"""
Zero-Knowledge Proofs Package for GenomeVault

Primary Components:
- Prover: Generate Groth16 ZK proofs via Circom backend
- Verifier: Verify ZK proofs
- PQEngine: Post-quantum proof engine

Usage:
    from genomevault.zk_proofs import Prover

    prover = Prover()
    proof = prover.prove_variant(public_input, private_input)
"""

# Primary production components
from .prover import Prover, Circuit, Proof, CircuitLibrary
from .verifier import Verifier

# Post-quantum proofs
from .post_quantum import PQEngine, prove, verify

__all__ = [
    # Primary
    "Prover",
    "Verifier",
    "Circuit",
    "Proof",
    "CircuitLibrary",
    # Post-quantum
    "PQEngine",
    "prove",
    "verify",
]
