"""
GenomeVault Hypervector Transform Package

Implements hierarchical hyperdimensional computing for genomic data:
- Multi-resolution encoding (10k, 15k, 20k dimensions)
- Domain-specific projections
- Holographic representations
- Privacy-preserving binding operations
"""

from .encoding import (
    HypervectorEncoder,
    DomainProjection,
    EncodingConfig,
    create_random_projection,
    encode_features
)

from .binding import (
    bind_vectors,
    unbind_vectors,
    bundle_vectors,
    permute_vector,
    circular_convolution,
    cross_modal_binding,
    HypervectorBinder,
    BindingType
)

from .holographic import (
    HolographicEncoder,
    create_holographic_memory,
    store_pattern,
    retrieve_pattern,
    calculate_capacity
)

from .mapping import (
    SimilarityMapper,
    DistanceMetric,
    map_to_similarity_space,
    preserve_distances,
    create_isometric_mapping
)

# Import hierarchical components if available
try:
    from .hierarchical import (
        HierarchicalEncoder,
        CompressionProfile,
        ResolutionLevel,
        encode_hierarchical,
        compress_hypervector
    )
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False

__all__ = [
    # Core encoding
    'HypervectorEncoder',
    'DomainProjection',
    'EncodingConfig',
    'create_random_projection',
    'encode_features',
    
    # Binding operations
    'bind_vectors',
    'unbind_vectors',
    'bundle_vectors',
    'permute_vector',
    'circular_convolution',
    'cross_modal_binding',
    'HypervectorBinder',
    'BindingType',
    
    # Holographic representation
    'HolographicEncoder',
    'create_holographic_memory',
    'store_pattern',
    'retrieve_pattern',
    'calculate_capacity',
    
    # Similarity mapping
    'SimilarityMapper',
    'DistanceMetric',
    'map_to_similarity_space',
    'preserve_distances',
    'create_isometric_mapping'
]

# Add hierarchical components if available
if HIERARCHICAL_AVAILABLE:
    __all__.extend([
        'HierarchicalEncoder',
        'CompressionProfile',
        'ResolutionLevel',
        'encode_hierarchical',
        'compress_hypervector'
    ])

# Version info
__version__ = '1.0.0'
__author__ = 'GenomeVault Team'

# Constants
DEFAULT_DIMENSIONS = 10000
MID_LEVEL_DIMENSIONS = 15000
HIGH_LEVEL_DIMENSIONS = 20000
