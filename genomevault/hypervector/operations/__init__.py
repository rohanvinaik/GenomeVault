"""
Hypervector operations module

This module provides optimized operations for hyperdimensional computing,
including binding operations and Hamming distance calculations.
"""

from .binding import BindingOperation, HypervectorBinder, MultiModalBinder
from .hamming_lut import HammingLUT, export_platform_implementations, generate_popcount_lut

__all__ = [
    "BindingOperation",
    "HammingLUT",
    "HypervectorBinder",
    "MultiModalBinder",
    "export_platform_implementations",
    "generate_popcount_lut",
]
