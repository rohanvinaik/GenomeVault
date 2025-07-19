"""
GenomeVault Local Processing Package
"""

from .sequencing import (
    DifferentialStorage,
    GenomicProfile,
    QualityMetrics,
    SequencingProcessor,
    Variant,
)

try:
    from .transcriptomics import (
        ExpressionProfile,
        GeneExpression,
        TranscriptomicsProcessor,
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
    "SequencingProcessor",
    "DifferentialStorage",
    "GenomicProfile",
    "Variant",
    "QualityMetrics",
]

if TranscriptomicsProcessor:
    __all__.extend(["TranscriptomicsProcessor", "ExpressionProfile", "GeneExpression"])

if EpigeneticsProcessor:
    __all__.extend(["EpigeneticsProcessor", "MethylationProfile", "MethylationSite"])
