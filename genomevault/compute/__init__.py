"""
GenomeVault Compute Backend

Hardware abstraction layer for CPU/GPU acceleration

Usage:
    >>> from genomevault.compute import get_accelerator, initialize_backend, ComputeBackend
    >>>
    >>> # Auto-detect best backend
    >>> accelerator = get_accelerator()
    >>> result = accelerator.encode_single(variants)
    >>>
    >>> # Or explicitly choose backend
    >>> initialize_backend(ComputeBackend.METAL)
    >>> accelerator = get_accelerator()

Available Backends:
    - CPU: Always available, production default
    - Metal: Apple Silicon (M1/M2/M3)
    - CUDA: NVIDIA GPUs

Default Behavior:
    AUTO detection priority: Metal > CUDA > CPU
"""

from genomevault.compute.backend import (
    ComputeBackend,
    HardwareAccelerator,
    ComputeBackendManager,
    get_accelerator,
    get_backend,
    initialize_backend,
)

__all__ = [
    'ComputeBackend',
    'HardwareAccelerator',
    'ComputeBackendManager',
    'get_accelerator',
    'get_backend',
    'initialize_backend',
]