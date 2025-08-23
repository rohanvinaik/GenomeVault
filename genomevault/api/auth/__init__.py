"""
Authentication module for GenomeVault API.

Provides OAuth2, OIDC, and API key authentication with
HIPAA-compliant audit logging and MFA support.
"""

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

from .oidc_provider import (
    OIDCProvider,
    OIDCConfig,
    OIDCUserInfo,
    OIDCManager,
    oidc_manager,
)

__all__ = [
    # OAuth2
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
    # OIDC
    "OIDCProvider",
    "OIDCConfig",
    "OIDCUserInfo",
    "OIDCManager",
    "oidc_manager",
]