"""
Nanopore streaming and biological signal detection module.

This module implements real-time nanopore data processing with:
- Streaming hypervector encoding
- Catalytic memory management
- Biological signal detection (methylation, SVs, etc.)
- Privacy-preserving anomaly detection
"""

from .biological_signals import (
    BiologicalSignalType,
    BiologicalSignal,
    ModificationProfile,
    BiologicalSignalDetector,
    example_signal_detection,
)
from .cli import count_fastq_reads, main

__all__ = [
    "BiologicalSignal",
    "BiologicalSignalDetector",
    "BiologicalSignalType",
    "ModificationProfile",
    "count_fastq_reads",
    "example_signal_detection",
    "main",
]
