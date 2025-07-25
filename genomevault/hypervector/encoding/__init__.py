"""
Hypervector encoding modules
"""

from .genomic import GenomicEncoder, PanelGranularity
from .packed import (
    HAMMING_LUT,
    PackedGenomicEncoder,
    PackedHV,
    PackedProjection,
    fast_hamming_distance,
)

__all__ = [
    # Original genomic encoder
    "GenomicEncoder",
    "PanelGranularity",
    # Packed implementations
    "PackedHV",
    "PackedProjection",
    "PackedGenomicEncoder",
    "fast_hamming_distance",
    "HAMMING_LUT",
]
