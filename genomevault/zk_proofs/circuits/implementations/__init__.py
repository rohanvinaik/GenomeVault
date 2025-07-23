"""
Actual ZK Circuit Implementations

This module provides working implementations of zero-knowledge proof circuits
for genomic privacy, building on the theoretical framework but providing
concrete, executable circuits using arkworks-style constraints.
"""

from .variant_proof_circuit import VariantProofCircuit
from .diabetes_risk_circuit import DiabetesRiskCircuit
from .prs_circuit import PolygeneticRiskScoreCircuit
from .merkle_tree_circuit import MerkleInclusionCircuit
from .range_proof_circuit import RangeProofCircuit
from .commitment_circuit import PedersenCommitmentCircuit

__all__ = [
    "VariantProofCircuit",
    "DiabetesRiskCircuit",
    "PolygeneticRiskScoreCircuit",
    "MerkleInclusionCircuit",
    "RangeProofCircuit",
    "PedersenCommitmentCircuit",
]
