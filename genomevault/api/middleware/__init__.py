"""
Middleware components for GenomeVault API.

Provides authentication, rate limiting, and audit logging middleware
for HIPAA-compliant API access control.
"""

from .authentication import (
    AuthenticationMiddleware,
    APIKey,
    APIKeyStatus,
    MFAMethod,
    MFAVerifier,
    generate_api_key,
    rotate_api_key,
)

from .rate_limiter import (
    RateLimitMiddleware,
    RateLimiter,
    RateLimitTier,
    RateLimitConfig,
    TIER_CONFIGS,
)

__all__ = [
    # Authentication
    "AuthenticationMiddleware",
    "APIKey",
    "APIKeyStatus",
    "MFAMethod",
    "MFAVerifier",
    "generate_api_key",
    "rotate_api_key",
    # Rate Limiting
    "RateLimitMiddleware",
    "RateLimiter",
    "RateLimitTier",
    "RateLimitConfig",
    "TIER_CONFIGS",
]
