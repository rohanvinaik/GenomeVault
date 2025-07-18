"""
GenomeVault Local Processing Package

Provides local multi-omics data processing capabilities including:
- Genomic sequencing data (WGS/WES)
- Transcriptomics (RNA-seq)
- Epigenetics (methylation, chromatin accessibility)
- Proteomics (mass spectrometry)
- Clinical phenotypes (EHR, FHIR)
"""

from .sequencing import (
    SequencingProcessor,
    DifferentialStorage,
    GenomicProfile,
    Variant,
    QualityMetrics
)

from .transcriptomics import (
    TranscriptomicsProcessor,
    ExpressionProfile,
    TranscriptExpression,
    BatchEffectResult
)

from .epigenetics import (
    MethylationProcessor,
    ChromatinAccessibilityProcessor,
    EpigeneticProfile,
    MethylationSite,
    ChromatinPeak,
    create_epigenetic_processor
)

from .proteomics import (
    ProteomicsProcessor,
    ProteomicsProfile,
    ProteinMeasurement,
    Peptide
)

from .phenotypes import (
    PhenotypeProcessor,
    PhenotypeProfile,
    ClinicalMeasurement,
    Diagnosis,
    Medication,
    FamilyHistory,
    PhenotypeCategory
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
