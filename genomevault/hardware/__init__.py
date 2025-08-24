"""
Hardware acceleration infrastructure for GenomeVault.

Provides unified access to various hardware acceleration backends:
- CPU optimizations (SIMD, vectorization)
- GPU acceleration (CUDA, Metal, ROCm, oneAPI)
- Cloud acceleration (AWS, GCP, Azure)
- Specialized hardware (TPUs, FPGAs)

This module serves as a central hub for all hardware acceleration,
allowing multiple pipelines to share optimized implementations.
"""

from genomevault.hardware.backend import (
    HardwareBackend,
    AcceleratorType,
    get_best_accelerator,
    list_available_accelerators
)

from genomevault.hardware.unified_engine import (
    UnifiedAccelerationEngine,
    AccelerationConfig
)

__all__ = [
    'HardwareBackend',
    'AcceleratorType',
    'get_best_accelerator',
    'list_available_accelerators',
    'UnifiedAccelerationEngine',
    'AccelerationConfig'
]