"""
HDC Encoder Module

Provides unified interface for genomic encoding supporting both:
- Legacy direct variant encoding (backward compatible)
- New differential encoding with cryptographic security
- Hardware-accelerated backend system (CPU/Metal/CUDA)

Use UnifiedGenomicEncoder for new code to access both modes.
Use HypervectorEncoder directly for legacy compatibility.
Use BackendOptimizedEncoder for hardware-accelerated encoding.
"""

from __future__ import annotations

from genomevault.core.constants import OmicsType
from genomevault.utils.config import CompressionTier
from .encoding import HypervectorConfig, HypervectorEncoder, ProjectionType
from .unified_encoder import (
    UnifiedGenomicEncoder,
    EncodingMode,
    EncodingFeatureFlags,
)
from .backend_adapter import (
    BackendOptimizedEncoder,
    BackendEncoderConfig,
    create_backend_encoder,
)

__all__ = [
    # Legacy exports (backward compatible)
    "HypervectorEncoder",
    "HypervectorConfig",
    "ProjectionType",
    "CompressionTier",
    "OmicsType",
    # New unified interface
    "UnifiedGenomicEncoder",
    "EncodingMode",
    "EncodingFeatureFlags",
    # Hardware-accelerated backend system
    "BackendOptimizedEncoder",
    "BackendEncoderConfig",
    "create_backend_encoder",
]
