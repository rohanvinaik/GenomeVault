"""
Hypervector module for genomic data encoding and operations
"""

# Import from hypervector_transform module as per encoder.py compatibility shim
from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
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
    "HypervectorEncoder",
    "HypervectorConfig",
]
