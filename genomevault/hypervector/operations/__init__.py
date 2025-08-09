"""
Hypervector operations module

This module provides optimized operations for hyperdimensional computing,
including binding operations and Hamming distance calculations.
"""

from .binding import (
    circular_convolution,
    element_wise_multiply,
    permutation_binding,
    bundle,
    unbundle,
)
from .hamming_lut import (
    HammingLUT,
    export_platform_implementations,
    generate_popcount_lut,
)

__all__ = [
    "circular_convolution",
    "element_wise_multiply",
    "permutation_binding",
    "bundle",
    "unbundle",
    "HammingLUT",
    "export_platform_implementations",
    "generate_popcount_lut",
]
