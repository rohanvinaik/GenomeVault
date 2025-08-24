"""
Authentication module for GenomeVault API.

Provides OAuth2, OIDC, and API key authentication with
HIPAA-compliant audit logging and MFA support.
"""

# Legacy OAuth2 and OIDC imports
try:
    from .oauth2 import (
        Token,
        TokenData,
        User,
        UserRole,
        create_access_token,
        create_refresh_token,
        authenticate_user,
        get_current_user,
        get_current_active_user,
        require_scopes,
        require_mfa,
        revoke_token,
        revoke_all_user_tokens,
        refresh_access_token,
        oauth2_scheme,
    )

    OAUTH2_AVAILABLE = True
except ImportError:
    OAUTH2_AVAILABLE = False

try:
    from .oidc_provider import (
        OIDCProvider,
        OIDCConfig,
        OIDCUserInfo,
        OIDCManager,
        oidc_manager,
    )

    OIDC_AVAILABLE = True
except ImportError:
    OIDC_AVAILABLE = False

# Enhanced API Key Authentication
from .api_key import (
    APIKeyScope,
    APIKeyType,
    APIKeyInfo,
    APIKeyManager,
    get_api_key_manager,
    authenticate_api_key,
    require_scope,
    require_clinical_access,
    require_admin_access,
)

# Build __all__ dynamically
__all__ = [
    # Enhanced API Key Authentication
    "APIKeyScope",
    "APIKeyType",
    "APIKeyInfo",
    "APIKeyManager",
    "get_api_key_manager",
    "authenticate_api_key",
    "require_scope",
    "require_clinical_access",
    "require_admin_access",
]

# Add OAuth2 imports if available
if OAUTH2_AVAILABLE:
    __all__.extend(
        [
            "Token",
            "TokenData",
            "User",
            "UserRole",
            "create_access_token",
            "create_refresh_token",
            "authenticate_user",
            "get_current_user",
            "get_current_active_user",
            "require_scopes",
            "require_mfa",
            "revoke_token",
            "revoke_all_user_tokens",
            "refresh_access_token",
            "oauth2_scheme",
        ]
    )

# Add OIDC imports if available
if OIDC_AVAILABLE:
    __all__.extend(
        [
            "OIDCProvider",
            "OIDCConfig",
            "OIDCUserInfo",
            "OIDCManager",
            "oidc_manager",
        ]
    )
