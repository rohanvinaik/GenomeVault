"""
Hypervector encoding modules

This module is maintained for backward compatibility.
Please migrate to genomevault.hypervector_transform.encoding

"""
import warnings

from .genomic import GenomicEncoder, PanelGranularity  # noqa: E402
from .orthogonal_projection import OrthogonalProjection  # noqa: E402
from .packed import (  # noqa: E402

    pack_bits,
    unpack_bits,
)
from .sparse_projection import SparseRandomProjection, sparse_random_matrix  # noqa: E402
from .unified_encoder import UnifiedHypervectorEncoder, create_encoder  # noqa: E402

warnings.warn(
    "genomevault.hypervector.encoding is deprecated. "
    "Use genomevault.hypervector_transform.encoding instead.",
    DeprecationWarning,
    stacklevel=2,
)

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
