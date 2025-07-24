"""HIPAA compliance and verification package."""

# Import modules directly for clean namespace
from . import integration, verifier
from .models import (
    EXPIRED,
    FAILED,
    INDIVIDUAL,
    ORGANIZATION,
    PENDING,
    REVOKED,
    VERIFIED,
    HIPAACredentials,
    NPIRecord,
    NPIType,
    VerificationRecord,
    VerificationStatus,
    is_active,
    to_chain_data,
)

__all__ = [
    "integration",
    "verifier",
    "EXPIRED",
    "FAILED",
    "HIPAACredentials",
    "INDIVIDUAL",
    "NPIRecord",
    "NPIType",
    "ORGANIZATION",
    "PENDING",
    "REVOKED",
    "VERIFIED",
    "VerificationRecord",
    "VerificationStatus",
    "is_active",
    "to_chain_data",
]
