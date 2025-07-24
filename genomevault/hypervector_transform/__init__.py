"""
Hypervector Transform Module for GenomeVault

This module implements Hierarchical Hyperdimensional Computing (HDC) for
privacy-preserving genomic data encoding.

Key components:
- HypervectorEncoder: Main encoder for transforming genomic data
- HypervectorBinder: Binding operations for combining hypervectors
- HypervectorRegistry: Version management and reproducibility
- HDC API: RESTful endpoints for encoding services
"""

from .binding_operations import BindingOperations  # Legacy compatibility
from .binding_operations import (
    BindingType,
    HypervectorBinder,
    circular_bind,
    fourier_bind,
    protect_vector,
)
from .hdc_encoder import (
    CompressionTier,
    HypervectorConfig,
    HypervectorEncoder,
    OmicsType,
    ProjectionType,
    create_encoder,
    encode_genomic_data,
)
from .registry import (
    HypervectorRegistry,
    VersionMigrator,
)

__version__ = "1.0.0"

__all__ = [
    # Main encoder
    "HypervectorEncoder",
    "HypervectorConfig",
    "create_encoder",
    "encode_genomic_data",
    # Enums
    "CompressionTier",
    "ProjectionType",
    "OmicsType",
    "BindingType",
    # Binding operations
    "HypervectorBinder",
    "BindingOperations",
    "circular_bind",
    "fourier_bind",
    "protect_vector",
    # Registry
    "HypervectorRegistry",
    "VersionMigrator",
]
