"""Compression module for genomic data."""

from .tiered_compression import (
    TieredCompressor,
    TieredCompressor as TieredCompression,  # Alias for compatibility
    CompressionTier,
    CompressionMetrics,
    VariantPriority,
)

__all__ = [
    "TieredCompression",
    "TieredCompressor",
    "CompressionTier",
    "CompressionMetrics",
    "VariantPriority",
]
