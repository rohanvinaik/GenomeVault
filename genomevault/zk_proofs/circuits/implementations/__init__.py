"""Package initialization for implementations."""
from .commitment_circuit import PedersenCommitmentCircuit
from .diabetes_risk_circuit import DiabetesRiskCircuit
from .merkle_tree_circuit import MerkleInclusionCircuit
from .prs_circuit import PolygeneticRiskScoreCircuit
from .range_proof_circuit import RangeProofCircuit
from .variant_proof_circuit import VariantProofCircuit

"""
Actual ZK Circuit Implementations

This module provides working implementations of zero-knowledge proof circuits
for genomic privacy, building on the theoretical framework but providing
concrete, executable circuits using arkworks-style constraints.
"""


__all__ = [
    "DiabetesRiskCircuit",
    "MerkleInclusionCircuit",
    "PedersenCommitmentCircuit",
    "PolygeneticRiskScoreCircuit",
    "RangeProofCircuit",
    "VariantProofCircuit",
]
