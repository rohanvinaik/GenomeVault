"""
HIPAA Package - Phase 2 Integration

Provides NPI verification, trusted signatory registry, and institutional onboarding.
"""

from .models import (
    VerificationStatus,
    NPIType,
    HIPAACredentials,
    VerificationRecord,
    NPIRecord,
)
# Skip integration.py import to avoid circular dependencies
# from .integration import HIPAANodeIntegration, HIPAAGovernanceIntegration
from .verifier import verify_access, REQUIRED_FIELDS

__all__ = [
    "HIPAACredentials",
    # "HIPAAGovernanceIntegration",
    # "HIPAANodeIntegration",
    "NPIRecord",
    "NPIType",
    "REQUIRED_FIELDS",
    "VerificationRecord",
    "VerificationStatus",
    "verify_access",
]
