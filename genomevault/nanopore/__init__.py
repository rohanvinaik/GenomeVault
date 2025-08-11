"""
Nanopore streaming and biological signal detection module.

This module implements real-time nanopore data processing with:
- Streaming hypervector encoding
- Catalytic memory management
- Biological signal detection (methylation, SVs, etc.)
- Privacy-preserving anomaly detection

from .biological_signals import BiologicalSignalDetector
from .gpu_kernels import GPUBindingKernel
from .streaming import NanoporeStreamProcessor

__all__ = [
    "BiologicalSignalDetector",
    "GPUBindingKernel",
    "NanoporeStreamProcessor",
]
