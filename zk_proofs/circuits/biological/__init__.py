"""
Biological zero-knowledge proof circuits.

Specialized circuits for genomic privacy including variant verification,
risk scores, clinical assessments, and multi-omics integration.
"""

from .variant import (
    VariantPresenceCircuit,
    PolygenenicRiskScoreCircuit,
    DiabetesRiskCircuit,
    PharmacogenomicCircuit,
    PathwayEnrichmentCircuit,
    create_hypervector_proof
)

from .multi_omics import (
    MultiOmicsCorrelationCircuit,
    GenotypePhenotypeAssociationCircuit,
    ClinicalTrialEligibilityCircuit,
    RareVariantBurdenCircuit,
    create_multi_omics_proof_suite
)

__all__ = [
    # Variant circuits
    'VariantPresenceCircuit',
    'PolygenenicRiskScoreCircuit',
    'DiabetesRiskCircuit',
    'PharmacogenomicCircuit',
    'PathwayEnrichmentCircuit',
    'create_hypervector_proof',
    
    # Multi-omics circuits
    'MultiOmicsCorrelationCircuit',
    'GenotypePhenotypeAssociationCircuit',
    'ClinicalTrialEligibilityCircuit',
    'RareVariantBurdenCircuit',
    'create_multi_omics_proof_suite'
]
