"""
Comprehensive Security Middleware Suite for GenomeVault API.

Provides authentication, rate limiting, audit logging, input sanitization,
CORS security, and security headers middleware for HIPAA-compliant
API access control with PHI data protection.
"""

# Legacy imports (for backward compatibility)
try:
    from .authentication import (
        AuthenticationMiddleware,
        APIKey,
        APIKeyStatus,
        MFAMethod,
        MFAVerifier,
        generate_api_key,
        rotate_api_key,
    )
    LEGACY_AUTH_AVAILABLE = True
except ImportError:
    LEGACY_AUTH_AVAILABLE = False

try:
    from .rate_limiter import (
        RateLimiter,
        RateLimitTier,
        RateLimitConfig,
        TIER_CONFIGS,
    )
    LEGACY_RATE_LIMITING_AVAILABLE = True
except ImportError:
    LEGACY_RATE_LIMITING_AVAILABLE = False

# New enhanced security middleware
from .rate_limiting import (
    RateLimitMiddleware,
    create_rate_limit_middleware,
    get_rate_limit_stats
)

from .input_sanitization import (
    InputSanitizationMiddleware,
    sanitize_clinical_id,
    sanitize_genomic_sequence,
    sanitize_genomic_variant
)

from .audit_logging import (
    AuditMiddleware,
    get_audit_logger,
    audit_authentication,
    audit_phi_access,
    audit_genomic_analysis
)

from .cors_security import (
    CORSSecurityMiddleware,
    create_cors_middleware,
    validate_origin_security,
    get_cors_policy_info
)

from .security_headers import (
    SecurityHeadersMiddleware,
    create_security_headers_middleware,
    handle_csp_report,
    get_security_profile_info,
    validate_security_headers
)

# Build __all__ list dynamically
__all__ = [
    # Enhanced security middleware
    "RateLimitMiddleware",
    "create_rate_limit_middleware", 
    "get_rate_limit_stats",
    "InputSanitizationMiddleware",
    "sanitize_clinical_id",
    "sanitize_genomic_sequence",
    "sanitize_genomic_variant",
    "AuditMiddleware",
    "get_audit_logger",
    "audit_authentication",
    "audit_phi_access", 
    "audit_genomic_analysis",
    "CORSSecurityMiddleware",
    "create_cors_middleware",
    "validate_origin_security",
    "get_cors_policy_info",
    "SecurityHeadersMiddleware",
    "create_security_headers_middleware",
    "handle_csp_report",
    "get_security_profile_info",
    "validate_security_headers"
]

# Add legacy imports if available
if LEGACY_AUTH_AVAILABLE:
    __all__.extend([
        "AuthenticationMiddleware",
        "APIKey",
        "APIKeyStatus", 
        "MFAMethod",
        "MFAVerifier",
        "generate_api_key",
        "rotate_api_key"
    ])

if LEGACY_RATE_LIMITING_AVAILABLE:
    __all__.extend([
        "RateLimiter",
        "RateLimitTier",
        "RateLimitConfig",
        "TIER_CONFIGS"
    ])
