"""
Hypervector encoding modules

This module provides encoding functionality for hypervectors.
Note: HypervectorEncoder is available from genomevault.hypervector.encoding module (encoding.py file)
"""

# Import from submodules
from .orthogonal_projection import OrthogonalProjection
from .packed import pack_bits, unpack_bits
from .sparse_projection import SparseRandomProjection, sparse_random_matrix

__all__ = [
    "OrthogonalProjection",
    "SparseRandomProjection",
    "pack_bits",
    "sparse_random_matrix",
    "unpack_bits",
]
