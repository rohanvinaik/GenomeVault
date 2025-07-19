"""
Zero-knowledge proof system for genomic privacy.

This package implements PLONK-based zero-knowledge proofs specialized
for genomic data, with support for variant verification, PRS calculation,
clinical assessments, and multi-omics integration.
"""

from .circuit_manager import CircuitManager, CircuitMetadata

# Base circuits
from .circuits.base_circuits import (
    AggregatorCircuit,
    BaseCircuit,
    ComparisonCircuit,
    FieldElement,
    HashPreimageCircuit,
    MerkleTreeCircuit,
    RangeProofCircuit,
)

# Multi-omics circuits
from .circuits.biological.multi_omics import (
    ClinicalTrialEligibilityCircuit,
    GenotypePhenotypeAssociationCircuit,
    MultiOmicsCorrelationCircuit,
    RareVariantBurdenCircuit,
    create_multi_omics_proof_suite,
)

# Biological circuits
from .circuits.biological.variant import (
    DiabetesRiskCircuit,
    PathwayEnrichmentCircuit,
    PharmacogenomicCircuit,
    PolygenenicRiskScoreCircuit,
    VariantPresenceCircuit,
    create_hypervector_proof,
)

# Post-quantum support
from .post_quantum import (
    LatticeProver,
    PostQuantumParameters,
    PostQuantumProver,
    PostQuantumTransition,
    STARKProver,
    benchmark_pq_performance,
    estimate_pq_proof_size,
)

# Core components
from .prover import Circuit, CircuitLibrary, Proof, Prover
from .verifier import VerificationResult, Verifier

__all__ = [
    # Core
    'Prover',
    'Proof',
    'Circuit',
    'CircuitLibrary',
    'Verifier',
    'VerificationResult',
    'CircuitManager',
    'CircuitMetadata',
    
    # Base circuits
    'BaseCircuit',
    'FieldElement',
    'MerkleTreeCircuit',
    'RangeProofCircuit',
    'ComparisonCircuit',
    'HashPreimageCircuit',
    'AggregatorCircuit',
    
    # Biological circuits
    'VariantPresenceCircuit',
    'PolygenenicRiskScoreCircuit',
    'DiabetesRiskCircuit',
    'PharmacogenomicCircuit',
    'PathwayEnrichmentCircuit',
    'create_hypervector_proof',
    
    # Multi-omics
    'MultiOmicsCorrelationCircuit',
    'GenotypePhenotypeAssociationCircuit',
    'ClinicalTrialEligibilityCircuit',
    'RareVariantBurdenCircuit',
    'create_multi_omics_proof_suite',
    
    # Post-quantum
    'PostQuantumProver',
    'STARKProver',
    'LatticeProver',
    'PostQuantumTransition',
    'PostQuantumParameters',
    'estimate_pq_proof_size',
    'benchmark_pq_performance'
]

# Version info
__version__ = '1.0.0'
__author__ = 'GenomeVault Team'
