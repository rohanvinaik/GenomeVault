"""
Zero-knowledge proof system for genomic privacy.

This package implements PLONK-based zero-knowledge proofs specialized
for genomic data, with support for variant verification, PRS calculation,
clinical assessments, and multi-omics integration.
"""

# Core components
from .prover import Prover, Proof, Circuit, CircuitLibrary
from .verifier import Verifier, VerificationResult
from .circuit_manager import CircuitManager, CircuitMetadata

# Base circuits
from .circuits.base_circuits import (
    BaseCircuit,
    FieldElement,
    MerkleTreeCircuit,
    RangeProofCircuit,
    ComparisonCircuit,
    HashPreimageCircuit,
    AggregatorCircuit
)

# Biological circuits
from .circuits.biological.variant import (
    VariantPresenceCircuit,
    PolygenenicRiskScoreCircuit,
    DiabetesRiskCircuit,
    PharmacogenomicCircuit,
    PathwayEnrichmentCircuit,
    create_hypervector_proof
)

# Multi-omics circuits
from .circuits.biological.multi_omics import (
    MultiOmicsCorrelationCircuit,
    GenotypePhenotypeAssociationCircuit,
    ClinicalTrialEligibilityCircuit,
    RareVariantBurdenCircuit,
    create_multi_omics_proof_suite
)

# Post-quantum support
from .post_quantum import (
    PostQuantumProver,
    STARKProver,
    LatticeProver,
    PostQuantumTransition,
    PostQuantumParameters,
    estimate_pq_proof_size,
    benchmark_pq_performance
)

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
