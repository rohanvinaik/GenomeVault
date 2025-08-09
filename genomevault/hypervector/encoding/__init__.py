"""
Hypervector encoding modules
"""

from .genomic import GenomicEncoder, PanelGranularity
from .orthogonal_projection import OrthogonalProjection
from .packed import (
    pack_bits,
    unpack_bits,
)
from .sparse_projection import SparseRandomProjection, sparse_random_matrix
from .unified_encoder import UnifiedHypervectorEncoder, create_encoder

# Alias for backward compatibility
HypervectorEncoder = UnifiedHypervectorEncoder

__all__ = [
    # Original genomic encoder
    "GenomicEncoder",
    "PanelGranularity",
    # Packed implementations
    "pack_bits",
    "unpack_bits",
    # Unified encoder with sparse/orthogonal projections
    "UnifiedHypervectorEncoder",
    "HypervectorEncoder",  # Alias
    "create_encoder",
    "sparse_random_matrix",
    "SparseRandomProjection",
    "OrthogonalProjection",
]
