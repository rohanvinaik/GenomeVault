"""
Hypervector operations module

This module provides optimized operations for hyperdimensional computing,
including binding operations and Hamming distance calculations.

This module is maintained for backward compatibility.
Please migrate to genomevault.hypervector_transform.binding_operations

"""
import warnings

from .binding import (  # noqa: E402

    bundle,
    circular_convolution,
    element_wise_multiply,
    permutation_binding,
    unbundle,
)
from .hamming_lut import (  # noqa: E402
    HammingLUT,
    export_platform_implementations,
    generate_popcount_lut,
)

warnings.warn(
    "genomevault.hypervector.operations is deprecated. "
    "Use genomevault.hypervector_transform.binding_operations instead.",
    DeprecationWarning,
    stacklevel=2,
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
