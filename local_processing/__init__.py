"""
GenomeVault Local Processing Package
"""

from .sequencing import (
    SequencingProcessor,
    DifferentialStorage,
    GenomicProfile,
    Variant,
    QualityMetrics,
)

try:
    from .transcriptomics import (
        TranscriptomicsProcessor,
        ExpressionProfile,
        GeneExpression,
    )
except ImportError:
    TranscriptomicsProcessor = None
    ExpressionProfile = None
    GeneExpression = None

try:
    from .epigenetics import (
        EpigeneticsProcessor,
        MethylationProfile,
        MethylationSite,
    )
except ImportError:
    EpigeneticsProcessor = None
    MethylationProfile = None
    MethylationSite = None

__all__ = [
    'SequencingProcessor',
    'DifferentialStorage',
    'GenomicProfile',
    'Variant',
    'QualityMetrics',
]

if TranscriptomicsProcessor:
    __all__.extend(['TranscriptomicsProcessor', 'ExpressionProfile', 'GeneExpression'])
    
if EpigeneticsProcessor:
    __all__.extend(['EpigeneticsProcessor', 'MethylationProfile', 'MethylationSite'])
