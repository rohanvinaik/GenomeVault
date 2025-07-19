"""
GenomeVault Local Processing Package

Provides local multi-omics data processing capabilities including:
- Genomic sequencing data (WGS/WES)
- Transcriptomics (RNA-seq)
- Epigenetics (methylation, chromatin accessibility)
- Proteomics (mass spectrometry)
- Clinical phenotypes (EHR, FHIR)
"""

from .epigenetics import (
    ChromatinAccessibilityProcessor,
    ChromatinPeak,
    EpigeneticProfile,
    MethylationProcessor,
    MethylationSite,
    create_epigenetic_processor,
)
from .phenotypes import (
    ClinicalMeasurement,
    Diagnosis,
    FamilyHistory,
    Medication,
    PhenotypeCategory,
    PhenotypeProcessor,
    PhenotypeProfile,
)
from .proteomics import Peptide, ProteinMeasurement, ProteomicsProcessor, ProteomicsProfile
from .sequencing import (
    DifferentialStorage,
    GenomicProfile,
    QualityMetrics,
    SequencingProcessor,
    Variant,
)
from .transcriptomics import (
    BatchEffectResult,
    ExpressionProfile,
    TranscriptExpression,
    TranscriptomicsProcessor,
)

__all__ = [
    # Sequencing
    'SequencingProcessor',
    'DifferentialStorage',
    'GenomicProfile',
    'Variant',
    'QualityMetrics',
    
    # Transcriptomics
    'TranscriptomicsProcessor',
    'ExpressionProfile',
    'TranscriptExpression',
    'BatchEffectResult',
    
    # Epigenetics
    'MethylationProcessor',
    'ChromatinAccessibilityProcessor',
    'EpigeneticProfile',
    'MethylationSite',
    'ChromatinPeak',
    'create_epigenetic_processor',
    
    # Proteomics
    'ProteomicsProcessor',
    'ProteomicsProfile',
    'ProteinMeasurement',
    'Peptide',
    
    # Phenotypes
    'PhenotypeProcessor',
    'PhenotypeProfile',
    'ClinicalMeasurement',
    'Diagnosis',
    'Medication',
    'FamilyHistory',
    'PhenotypeCategory'
]

# Version info
__version__ = '1.0.0'
__author__ = 'GenomeVault Team'
