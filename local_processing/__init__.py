"""
Local Processing Module

Secure containerized processing of multi-omics data.
All processing happens locally without raw data leaving the device.
"""

from .pipeline import MultiOmicsPipeline
from .compression import TieredCompressor
from .validators import (
    validate_genomic_data,
    validate_transcriptomic_data,
    validate_epigenetic_data,
    validate_proteomic_data
)

__all__ = [
    "MultiOmicsPipeline",
    "TieredCompressor",
    "validate_genomic_data",
    "validate_transcriptomic_data",
    "validate_epigenetic_data",
    "validate_proteomic_data"
]
