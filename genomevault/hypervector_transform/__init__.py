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

from .holographic import (
    HolographicStructure,
    HolographicEncoder,
    encode_variant,
    query_hologram,
)
from .registry import HypervectorRegistry, VersionMigrator
from .hierarchical import (
    ProjectionDomain,
    HierarchicalHypervector,
    HolographicRepresentation,
    HierarchicalEncoder,
    create_hierarchical_encoder,
    encode_genomic_hierarchical,
)
from .binding_operations import (
    BindingOperation,
    BindingType,
    HypervectorBinder,
    BindingOperations,
    bind,
    superpose,
    circular_bind,
    fourier_bind,
    protect_vector,
)
from .hdc_api import (
    EncodingRequest,
    EncodingResponse,
    MultiModalEncodingRequest,
    SimilarityRequest,
    DecodeRequest,
    VersionInfo,
    PerformanceMetrics,
    get_encoder,
    include_routes,
)
from .binding import (
    BindingType,
    HypervectorBinder,
    PositionalBinder,
    CrossModalBinder,
    circular_bind,
    protect_vector,
)
from .mapping import (
    MappingConfig,
    SimilarityPreservingMapper,
    BiologicalSimilarityMapper,
    ManifoldPreservingMapper,
    create_biological_mapper,
    preserve_similarities,
)

__all__ = [
    "BindingOperation",
    "BindingOperations",
    "BindingType",
    "BindingType",
    "BiologicalSimilarityMapper",
    "CrossModalBinder",
    "DecodeRequest",
    "EncodingRequest",
    "EncodingResponse",
    "HierarchicalEncoder",
    "HierarchicalHypervector",
    "HolographicEncoder",
    "HolographicRepresentation",
    "HolographicStructure",
    "HypervectorBinder",
    "HypervectorBinder",
    "HypervectorRegistry",
    "ManifoldPreservingMapper",
    "MappingConfig",
    "MultiModalEncodingRequest",
    "PerformanceMetrics",
    "PositionalBinder",
    "ProjectionDomain",
    "SimilarityPreservingMapper",
    "SimilarityRequest",
    "VersionInfo",
    "VersionMigrator",
    "bind",
    "circular_bind",
    "circular_bind",
    "create_biological_mapper",
    "create_hierarchical_encoder",
    "encode_genomic_hierarchical",
    "encode_variant",
    "fourier_bind",
    "get_encoder",
    "include_routes",
    "preserve_similarities",
    "protect_vector",
    "protect_vector",
    "query_hologram",
    "superpose",
]
