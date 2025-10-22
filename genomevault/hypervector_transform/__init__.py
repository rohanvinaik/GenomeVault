"""
Hypervector Transform Module for GenomeVault

This module implements Hierarchical Hyperdimensional Computing (HDC) for
privacy-preserving genomic data encoding.

Key components:
- UnifiedGenomicEncoder: Unified interface supporting differential and legacy encoding
- HypervectorEncoder: Legacy direct encoding (backward compatible)
- DifferentialGenomicEncoder: New cryptographic differential encoding
- HypervectorBinder: Binding operations for combining hypervectors
- HypervectorRegistry: Version management and reproducibility
- HDC API: RESTful endpoints for encoding services

Migration Guide:
    # Legacy code (still works):
    from genomevault.hypervector_transform import HypervectorEncoder
    encoder = HypervectorEncoder(config)

    # New unified interface (recommended):
    from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode
    encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL)
    encoded = encoder.encode_genome(genome, AnalysisType.SLIDING_WINDOW)
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
from .hdc_encoder import (
    UnifiedGenomicEncoder,
    EncodingMode,
    EncodingFeatureFlags,
    # Legacy exports for backward compatibility
    HypervectorEncoder,
    HypervectorConfig,
    ProjectionType,
    # Hardware-accelerated backend system
    BackendOptimizedEncoder,
    BackendEncoderConfig,
    create_backend_encoder,
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
    # Unified encoding interface
    "UnifiedGenomicEncoder",
    "EncodingMode",
    "EncodingFeatureFlags",
    # Legacy encoding components
    "HypervectorEncoder",
    "HypervectorConfig",
    "ProjectionType",
    # Hardware-accelerated backend system
    "BackendOptimizedEncoder",
    "BackendEncoderConfig",
    "create_backend_encoder",
]
