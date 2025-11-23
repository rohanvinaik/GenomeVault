"""Constants module."""

from __future__ import annotations

from enum import Enum

HYPERVECTOR_DIMENSIONS: int = 10_000
DEFAULT_SPARSITY: float = 0.1

# Security defaults
DEFAULT_SECURITY_LEVEL: int = 128
MAX_VARIANTS: int = 1000
VERIFICATION_TIME_MAX: float = 30.0

# ZK Proof defaults
DEFAULT_CIRCUIT_SIZE: int = 1024
MAX_PROOF_SIZE: int = 1024 * 1024  # 1MB


class OmicsType(Enum):
    """OmicsType implementation."""

    GENOMIC = "genomic"
    TRANSCRIPTOMIC = "transcriptomic"
    PROTEOMIC = "proteomic"
    METABOLOMIC = "metabolomic"
    EPIGENOMIC = "epigenomic"


class CompressionTier(str, Enum):
    """Compression tier levels."""

    MINI = "mini"  # ~25KB - 5,000 most-studied SNPs
    CLINICAL = "clinical"  # ~300KB - ACMG + PharmGKB variants
    FULL = "full"  # 100-200KB per modality - Full HDC vectors
