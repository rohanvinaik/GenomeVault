"""
Biological zero-knowledge proof circuits.

Specialized circuits for genomic privacy including variant verification,
risk scores, clinical assessments, and multi-omics integration.
"""

from .multi_omics import (
    ClinicalTrialEligibilityCircuit,
    GenotypePhenotypeAssociationCircuit,
    MultiOmicsCorrelationCircuit,
    RareVariantBurdenCircuit,
    create_multi_omics_proof_suite,
)
from .variant import (
    DiabetesRiskCircuit,
    PathwayEnrichmentCircuit,
    PharmacogenomicCircuit,
    PolygenenicRiskScoreCircuit,
    VariantPresenceCircuit,
    create_hypervector_proof,
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
