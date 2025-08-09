# genomevault/core/constants.py
from __future__ import annotations

"""Constants module."""
"""Constants module."""
from enum import Enum

# Hypervector defaults
HYPERVECTOR_DIMENSIONS = 10_000
DEFAULT_SPARSITY = 0.1

# Security defaults
DEFAULT_SECURITY_LEVEL = 128
MAX_VARIANTS = 1000
VERIFICATION_TIME_MAX = 30.0

# ZK Proof defaults
DEFAULT_CIRCUIT_SIZE = 1024
MAX_PROOF_SIZE = 1024 * 1024  # 1MB


class OmicsType(Enum):
    """OmicsType implementation."""
    GENOMIC = "genomic"
    TRANSCRIPTOMIC = "transcriptomic"
    PROTEOMIC = "proteomic"
    METABOLOMIC = "metabolomic"
    EPIGENOMIC = "epigenomic"
