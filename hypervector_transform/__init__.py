"""
GenomeVault Hypervector Transform Package
"""

from .binding import (
    BindingType,
    HypervectorBinder,
    PositionalBinder,
    CrossModalBinder,
    circular_bind,
    protect_vector,
)

from .encoding import (
    ProjectionType,
    HypervectorConfig,
    HypervectorEncoder,
    create_encoder,
    encode_genomic_data,
)

__all__ = [
    # Binding
    'BindingType',
    'HypervectorBinder',
    'PositionalBinder', 
    'CrossModalBinder',
    'circular_bind',
    'protect_vector',
    # Encoding
    'ProjectionType',
    'HypervectorConfig',
    'HypervectorEncoder',
    'create_encoder',
    'encode_genomic_data',
]
