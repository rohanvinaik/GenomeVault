"""Package initialization for circuits."""

from .multi_modal_training_proof import (
    ModalityMetrics,
    CrossModalAlignment,
    MultiModalTrainingProof,
)
from .clinical_circuits import (
    ClinicalBiomarkerCircuit,
    DiabetesRiskCircuit,
    ProofData,
    ClinicalCircuit,
    create_circuit,
)
from .base_circuits import (
    FieldElement,
    BaseCircuit,
    MerkleTreeCircuit,
    RangeProofCircuit,
    ComparisonCircuit,
)
from .base_genomic import (
    BaseGenomicCircuit,
    AncestryCompositionCircuit,
    DiabetesRiskCircuit,
    PathwayEnrichmentCircuit,
    PharmacogenomicCircuit,
)
from .training_proof import TrainingProofCircuit

__all__ = [
    "AncestryCompositionCircuit",
    "BaseCircuit",
    "BaseGenomicCircuit",
    "ClinicalBiomarkerCircuit",
    "ClinicalCircuit",
    "ComparisonCircuit",
    "CrossModalAlignment",
    "DiabetesRiskCircuit",
    "DiabetesRiskCircuit",
    "FieldElement",
    "MerkleTreeCircuit",
    "ModalityMetrics",
    "MultiModalTrainingProof",
    "PathwayEnrichmentCircuit",
    "PharmacogenomicCircuit",
    "ProofData",
    "RangeProofCircuit",
    "TrainingProofCircuit",
    "create_circuit",
]
