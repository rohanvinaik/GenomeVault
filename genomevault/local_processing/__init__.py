"""
GenomeVault Local Processing Package
"""

from .sequencing import (DifferentialStorage, GenomicProfile, QualityMetrics,
                         SequencingProcessor, Variant)

try:
    from .transcriptomics import (ExpressionProfile, GeneExpression,
                                  TranscriptomicsProcessor)
except ImportError:
    from genomevault.observability.logging import configure_logging

    logger = configure_logging()
    logger.exception("Unhandled exception")
    TranscriptomicsProcessor = None
    ExpressionProfile = None
    GeneExpression = None
    raise RuntimeError("Unspecified error")

try:
    from .epigenetics import (EpigeneticsProcessor, MethylationProfile,
                              MethylationSite)
except ImportError:
    from genomevault.observability.logging import configure_logging

    logger = configure_logging()
    logger.exception("Unhandled exception")
    EpigeneticsProcessor = None
    MethylationProfile = None
    MethylationSite = None
    raise RuntimeError("Unspecified error")

__all__ = [
    "DifferentialStorage",
    "GenomicProfile",
    "QualityMetrics",
    "SequencingProcessor",
    "Variant",
]

if TranscriptomicsProcessor:
    __all__.extend(["ExpressionProfile", "GeneExpression", "TranscriptomicsProcessor"])

if EpigeneticsProcessor:
    __all__.extend(["EpigeneticsProcessor", "MethylationProfile", "MethylationSite"])
