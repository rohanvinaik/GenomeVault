"""
Lightweight package init.

Avoid importing heavy submodules (e.g., sequencing with pysam) at import time.
Import them lazily inside the functions that need them.

"""

from .epigenetics import normalize_methylation
from .proteomics import (
    QuantificationMethod,
    ModificationType,
    Peptide,
    ProteinMeasurement,
    ProteomicsProfile,
    ProteomicsProcessor,
)
from .phenotypes import (
    PhenotypeCategory,
    ClinicalMeasurement,
    Diagnosis,
    Medication,
    FamilyHistory,
    PhenotypeProfile,
    PhenotypeProcessor,
)
from .model_snapshot import ModelSnapshot
from .compression import CompressionProfile, CompressedData, CompressionEngine
from .drift_detection import (
    DriftType,
    DriftSeverity,
    DriftEvent,
    ModelMonitoringState,
    RealTimeModelMonitor,
    CovariateShiftDetector,
    PredictionDriftDetector,
    PerformanceDriftDetector,
    SemanticDriftDetector,
)
from .transcriptomics import process, validate_features
from .common import validate_features, process
from .pipeline import MultiOmicsPipeline
from .sequencing import (
    QualityMetrics,
    Variant,
    GenomicProfile,
    SequencingProcessor,
    DifferentialStorage,
)
from .differential_privacy_audit import (
    PrivacyMechanism,
    PrivacyEvent,
    PrivacyBudget,
    DifferentialPrivacyAuditor,
    PrivacyAccountant,
)

__all__ = [
    "ClinicalMeasurement",
    "CompressedData",
    "CompressionEngine",
    "CompressionProfile",
    "CovariateShiftDetector",
    "Diagnosis",
    "DifferentialPrivacyAuditor",
    "DifferentialStorage",
    "DriftEvent",
    "DriftSeverity",
    "DriftType",
    "FamilyHistory",
    "GenomicProfile",
    "Medication",
    "ModelMonitoringState",
    "ModelSnapshot",
    "ModificationType",
    "MultiOmicsPipeline",
    "Peptide",
    "PerformanceDriftDetector",
    "PhenotypeCategory",
    "PhenotypeProcessor",
    "PhenotypeProfile",
    "PredictionDriftDetector",
    "PrivacyAccountant",
    "PrivacyBudget",
    "PrivacyEvent",
    "PrivacyMechanism",
    "ProteinMeasurement",
    "ProteomicsProcessor",
    "ProteomicsProfile",
    "QualityMetrics",
    "QuantificationMethod",
    "RealTimeModelMonitor",
    "SemanticDriftDetector",
    "SequencingProcessor",
    "Variant",
    "normalize_methylation",
    "process",
    "process",
    "validate_features",
    "validate_features",
]
