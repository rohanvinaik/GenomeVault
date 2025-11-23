"""
Hypervector module for genomic data encoding and operations
"""

# Import from hypervector_transform module as per encoder.py compatibility shim
# NOTE: Commented out to avoid circular import when metal_engine is imported
# from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
from .encoding.genomic import GenomicEncoder
from .error_handling import (
    AdaptiveHDCEncoder,
    ECCEncoderMixin,
    ErrorBudget,
    ErrorBudgetAllocator,
)

__all__ = [
    "AdaptiveHDCEncoder",
    "ECCEncoderMixin",
    "ErrorBudget",
    "ErrorBudgetAllocator",
    "GenomicEncoder",
    # "HypervectorEncoder",  # Removed from __all__ due to circular import
    # "HypervectorConfig",   # Removed from __all__ due to circular import
]
