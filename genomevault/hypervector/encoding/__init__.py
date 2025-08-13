"""
Hypervector encoding modules

This module is maintained for backward compatibility.
Please migrate to genomevault.hypervector_transform.encoding

"""

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
