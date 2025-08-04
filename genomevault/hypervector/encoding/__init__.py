"""
Hypervector encoding modules
"""

from .genomic import GenomicEncoder, PanelGranularity
from .orthogonal_projection import OrthogonalProjection
from .packed import (HAMMING_LUT, PackedGenomicEncoder, PackedHV,
                     PackedProjection, fast_hamming_distance)
from .sparse_projection import SparseRandomProjection
from .unified_encoder import UnifiedHypervectorEncoder, create_encoder

# Alias for backward compatibility
HypervectorEncoder = UnifiedHypervectorEncoder

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
    # Unified encoder with sparse/orthogonal projections
    "UnifiedHypervectorEncoder",
    "HypervectorEncoder",  # Alias
    "create_encoder",
    "SparseRandomProjection",
    "OrthogonalProjection",
]
