"""
OIDC (OpenID Connect) provider integration for enterprise SSO.

Supports Okta, Auth0, Azure AD, Google, and generic OIDC providers
for HIPAA-compliant authentication with enterprise identity providers.
"""

import os
import json
import logging
import httpx
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from enum import Enum
from urllib.parse import urlencode

from fastapi import HTTPException, status, Request
from jose import jwt, JWTError
from pydantic import BaseModel, HttpUrl
import redis

from genomevault.api.auth.oauth2 import (
    User,
    UserRole,
    Token,
    TokenData,
    create_access_token,
    create_refresh_token,
    ROLE_SCOPES,
)

logger = logging.getLogger(__name__)

# Redis client for caching
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


class OIDCProvider(str, Enum):
    """Supported OIDC providers."""

    OKTA = "okta"
    AUTH0 = "auth0"
    AZURE_AD = "azure_ad"
    GOOGLE = "google"
    GENERIC = "generic"


class OIDCConfig(BaseModel):
    """OIDC provider configuration."""

    provider: OIDCProvider
    client_id: str
    client_secret: str
    issuer: HttpUrl
    authorization_endpoint: HttpUrl
    token_endpoint: HttpUrl
    userinfo_endpoint: HttpUrl
    jwks_uri: HttpUrl
    redirect_uri: HttpUrl
    scopes: List[str] = ["openid", "profile", "email"]
    # Provider-specific settings
    tenant_id: Optional[str] = None  # Azure AD
    domain: Optional[str] = None  # Auth0/Okta
    audience: Optional[str] = None  # Auth0
    # Attribute mapping
    username_claim: str = "preferred_username"
    email_claim: str = "email"
    name_claim: str = "name"
    groups_claim: str = "groups"
    # HIPAA-specific claims
    npi_claim: Optional[str] = "custom:npi_number"
    organization_claim: Optional[str] = "custom:organization_id"
    baa_claim: Optional[str] = "custom:baa_signed"


class OIDCUserInfo(BaseModel):
    """OIDC user information from provider."""

    sub: str  # Subject identifier
    preferred_username: Optional[str] = None
    email: Optional[str] = None
    email_verified: bool = False
    name: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    groups: List[str] = []
    # Custom claims for GenomeVault
    npi_number: Optional[str] = None
    organization_id: Optional[str] = None
    baa_signed: bool = False
    roles: List[str] = []
    department: Optional[str] = None
    job_title: Optional[str] = None


class OIDCManager:
    """Manager for OIDC provider integrations."""

    def __init__(self):
        """Initialize OIDC manager with provider configurations."""
        self.providers: Dict[OIDCProvider, OIDCConfig] = {}
        self._load_provider_configs()
        self.http_client = httpx.AsyncClient(timeout=30.0)

    def _load_provider_configs(self):
        """Load OIDC provider configurations from environment."""
        # Okta configuration
        if os.getenv("OKTA_DOMAIN"):
            self.providers[OIDCProvider.OKTA] = OIDCConfig(
                provider=OIDCProvider.OKTA,
                client_id=os.getenv("OKTA_CLIENT_ID", ""),
                client_secret=os.getenv("OKTA_CLIENT_SECRET", ""),
                issuer=f"https://{os.getenv('OKTA_DOMAIN')}",
                authorization_endpoint=f"https://{os.getenv('OKTA_DOMAIN')}/oauth2/v1/authorize",
                token_endpoint=f"https://{os.getenv('OKTA_DOMAIN')}/oauth2/v1/token",
                userinfo_endpoint=f"https://{os.getenv('OKTA_DOMAIN')}/oauth2/v1/userinfo",
                jwks_uri=f"https://{os.getenv('OKTA_DOMAIN')}/oauth2/v1/keys",
                redirect_uri=os.getenv(
                    "OKTA_REDIRECT_URI", "https://api.genomevault.io/auth/callback/okta"
                ),
                domain=os.getenv("OKTA_DOMAIN"),
                scopes=["openid", "profile", "email", "groups"],
                groups_claim="groups",
                npi_claim="npi_number",
                organization_claim="organization_id",
                baa_claim="baa_signed",
            )

        # Auth0 configuration
        if os.getenv("AUTH0_DOMAIN"):
            self.providers[OIDCProvider.AUTH0] = OIDCConfig(
                provider=OIDCProvider.AUTH0,
                client_id=os.getenv("AUTH0_CLIENT_ID", ""),
                client_secret=os.getenv("AUTH0_CLIENT_SECRET", ""),
                issuer=f"https://{os.getenv('AUTH0_DOMAIN')}/",
                authorization_endpoint=f"https://{os.getenv('AUTH0_DOMAIN')}/authorize",
                token_endpoint=f"https://{os.getenv('AUTH0_DOMAIN')}/oauth/token",
                userinfo_endpoint=f"https://{os.getenv('AUTH0_DOMAIN')}/userinfo",
                jwks_uri=f"https://{os.getenv('AUTH0_DOMAIN')}/.well-known/jwks.json",
                redirect_uri=os.getenv(
                    "AUTH0_REDIRECT_URI", "https://api.genomevault.io/auth/callback/auth0"
                ),
                domain=os.getenv("AUTH0_DOMAIN"),
                audience=os.getenv("AUTH0_AUDIENCE"),
                scopes=["openid", "profile", "email"],
                username_claim="nickname",
                groups_claim="https://genomevault.io/groups",
                npi_claim="https://genomevault.io/npi_number",
                organization_claim="https://genomevault.io/organization_id",
                baa_claim="https://genomevault.io/baa_signed",
            )

        # Azure AD configuration
        if os.getenv("AZURE_TENANT_ID"):
            tenant_id = os.getenv("AZURE_TENANT_ID")
            self.providers[OIDCProvider.AZURE_AD] = OIDCConfig(
                provider=OIDCProvider.AZURE_AD,
                client_id=os.getenv("AZURE_CLIENT_ID", ""),
                client_secret=os.getenv("AZURE_CLIENT_SECRET", ""),
                issuer=f"https://login.microsoftonline.com/{tenant_id}/v2.0",
                authorization_endpoint=f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize",
                token_endpoint=f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
                userinfo_endpoint="https://graph.microsoft.com/v1.0/me",
                jwks_uri=f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys",
                redirect_uri=os.getenv(
                    "AZURE_REDIRECT_URI", "https://api.genomevault.io/auth/callback/azure"
                ),
                tenant_id=tenant_id,
                scopes=["openid", "profile", "email", "User.Read"],
                username_claim="preferred_username",
                email_claim="mail",
                groups_claim="groups",
                npi_claim="extension_npi_number",
                organization_claim="companyName",
                baa_claim="extension_baa_signed",
            )

        # Google configuration
        if os.getenv("GOOGLE_CLIENT_ID"):
            self.providers[OIDCProvider.GOOGLE] = OIDCConfig(
                provider=OIDCProvider.GOOGLE,
                client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
                client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
                issuer="https://accounts.google.com",
                authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
                token_endpoint="https://oauth2.googleapis.com/token",
                userinfo_endpoint="https://openidconnect.googleapis.com/v1/userinfo",
                jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
                redirect_uri=os.getenv(
                    "GOOGLE_REDIRECT_URI", "https://api.genomevault.io/auth/callback/google"
                ),
                scopes=["openid", "profile", "email"],
                username_claim="email",
                email_claim="email",
                name_claim="name",
            )

    def get_authorization_url(
        self, provider: OIDCProvider, state: str, nonce: str, code_challenge: Optional[str] = None
    ) -> str:
        """Generate authorization URL for OIDC provider."""
        config = self.providers.get(provider)
        if not config:
            raise ValueError(f"Provider {provider} not configured")

        params = {
            "client_id": config.client_id,
            "response_type": "code",
            "redirect_uri": str(config.redirect_uri),
            "scope": " ".join(config.scopes),
            "state": state,
            "nonce": nonce,
        }

        # Add PKCE if provided
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        # Provider-specific parameters
        if provider == OIDCProvider.AUTH0 and config.audience:
            params["audience"] = config.audience

        if provider == OIDCProvider.AZURE_AD:
            params["response_mode"] = "query"
            params["prompt"] = "select_account"

        if provider == OIDCProvider.GOOGLE:
            params["access_type"] = "offline"
            params["prompt"] = "consent"

        return f"{config.authorization_endpoint}?{urlencode(params)}"

    async def exchange_code_for_tokens(
        self, provider: OIDCProvider, code: str, code_verifier: Optional[str] = None
    ) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        config = self.providers.get(provider)
        if not config:
            raise ValueError(f"Provider {provider} not configured")

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "redirect_uri": str(config.redirect_uri),
        }

        # Add PKCE verifier if provided
        if code_verifier:
            data["code_verifier"] = code_verifier

        try:
            response = await self.http_client.post(
                str(config.token_endpoint),
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()

            tokens = response.json()

            # Log token exchange
            logger.info(
                f"Token exchange successful for provider {provider}",
                extra={
                    "event": "oidc_token_exchange",
                    "provider": provider.value,
                },
            )

            return tokens

        except httpx.HTTPError as e:
            logger.error(
                f"Token exchange failed for provider {provider}: {str(e)}",
                extra={
                    "event": "oidc_token_exchange_failed",
                    "provider": provider.value,
                    "error": str(e),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Token exchange failed: {str(e)}"
            )

    async def get_user_info(self, provider: OIDCProvider, access_token: str) -> OIDCUserInfo:
        """Get user information from OIDC provider."""
        config = self.providers.get(provider)
        if not config:
            raise ValueError(f"Provider {provider} not configured")

        try:
            # Special handling for Azure AD (uses Microsoft Graph)
            if provider == OIDCProvider.AZURE_AD:
                headers = {"Authorization": f"Bearer {access_token}"}

                # Get basic user info
                response = await self.http_client.get(
                    str(config.userinfo_endpoint), headers=headers
                )
                response.raise_for_status()
                user_data = response.json()

                # Get user groups
                groups_response = await self.http_client.get(
                    "https://graph.microsoft.com/v1.0/me/memberOf", headers=headers
                )
                if groups_response.status_code == 200:
                    groups_data = groups_response.json()
                    groups = [g["displayName"] for g in groups_data.get("value", [])]
                else:
                    groups = []

                # Map Azure AD data to OIDCUserInfo
                return OIDCUserInfo(
                    sub=user_data.get("id"),
                    preferred_username=user_data.get("userPrincipalName"),
                    email=user_data.get("mail") or user_data.get("userPrincipalName"),
                    email_verified=True,  # Azure AD emails are verified
                    name=user_data.get("displayName"),
                    given_name=user_data.get("givenName"),
                    family_name=user_data.get("surname"),
                    groups=groups,
                    organization_id=user_data.get("companyName"),
                    department=user_data.get("department"),
                    job_title=user_data.get("jobTitle"),
                    # Custom extensions for HIPAA
                    npi_number=user_data.get("extension_npi_number"),
                    baa_signed=user_data.get("extension_baa_signed", False),
                )

            else:
                # Standard OIDC userinfo endpoint
                response = await self.http_client.get(
                    str(config.userinfo_endpoint),
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                response.raise_for_status()
                user_data = response.json()

                # Map provider-specific claims
                return OIDCUserInfo(
                    sub=user_data.get("sub"),
                    preferred_username=user_data.get(config.username_claim),
                    email=user_data.get(config.email_claim),
                    email_verified=user_data.get("email_verified", False),
                    name=user_data.get(config.name_claim),
                    given_name=user_data.get("given_name"),
                    family_name=user_data.get("family_name"),
                    groups=user_data.get(config.groups_claim, []),
                    npi_number=user_data.get(config.npi_claim) if config.npi_claim else None,
                    organization_id=(
                        user_data.get(config.organization_claim)
                        if config.organization_claim
                        else None
                    ),
                    baa_signed=(
                        user_data.get(config.baa_claim, False) if config.baa_claim else False
                    ),
                    roles=user_data.get("roles", []),
                    department=user_data.get("department"),
                    job_title=user_data.get("job_title"),
                )

        except httpx.HTTPError as e:
            logger.error(
                f"Failed to get user info from {provider}: {str(e)}",
                extra={
                    "event": "oidc_userinfo_failed",
                    "provider": provider.value,
                    "error": str(e),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to get user info: {str(e)}"
            )

    async def validate_id_token(
        self, provider: OIDCProvider, id_token: str, nonce: str
    ) -> Dict[str, Any]:
        """Validate and decode OIDC ID token."""
        config = self.providers.get(provider)
        if not config:
            raise ValueError(f"Provider {provider} not configured")

        try:
            # Get JWKS from cache or fetch
            jwks_cache_key = f"jwks:{provider.value}"
            jwks_data = redis_client.get(jwks_cache_key)

            if not jwks_data:
                response = await self.http_client.get(str(config.jwks_uri))
                response.raise_for_status()
                jwks_data = response.text
                # Cache for 1 hour
                redis_client.setex(jwks_cache_key, 3600, jwks_data)

            jwks = json.loads(jwks_data)

            # Decode and validate ID token
            unverified_header = jwt.get_unverified_header(id_token)

            # Find the key
            rsa_key = None
            for key in jwks["keys"]:
                if key["kid"] == unverified_header["kid"]:
                    rsa_key = key
                    break

            if not rsa_key:
                raise ValueError("Unable to find appropriate key")

            # Validate token
            payload = jwt.decode(
                id_token,
                rsa_key,
                algorithms=["RS256"],
                audience=config.client_id,
                issuer=str(config.issuer),
            )

            # Validate nonce
            if payload.get("nonce") != nonce:
                raise ValueError("Invalid nonce")

            # Check expiration
            if "exp" in payload:
                exp_time = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
                if datetime.now(timezone.utc) > exp_time:
                    raise ValueError("Token expired")

            logger.info(
                f"ID token validated for provider {provider}",
                extra={
                    "event": "oidc_token_validated",
                    "provider": provider.value,
                    "sub": payload.get("sub"),
                },
            )

            return payload

        except (JWTError, ValueError) as e:
            logger.error(
                f"ID token validation failed for {provider}: {str(e)}",
                extra={
                    "event": "oidc_token_validation_failed",
                    "provider": provider.value,
                    "error": str(e),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid ID token: {str(e)}"
            )

    def map_groups_to_roles(self, groups: List[str], provider: OIDCProvider) -> List[UserRole]:
        """Map OIDC groups to GenomeVault roles."""
        role_mapping = {
            # Okta groups
            "genomevault-clinicians": UserRole.CLINICIAN,
            "genomevault-researchers": UserRole.RESEARCHER,
            "genomevault-admins": UserRole.ADMIN,
            "genomevault-patients": UserRole.PATIENT,
            # Azure AD groups
            "Clinicians": UserRole.CLINICIAN,
            "Researchers": UserRole.RESEARCHER,
            "Administrators": UserRole.ADMIN,
            "Patients": UserRole.PATIENT,
            # Auth0 roles
            "clinician": UserRole.CLINICIAN,
            "researcher": UserRole.RESEARCHER,
            "admin": UserRole.ADMIN,
            "patient": UserRole.PATIENT,
            # Generic medical staff groups
            "medical-staff": UserRole.CLINICIAN,
            "physicians": UserRole.CLINICIAN,
            "nurses": UserRole.CLINICIAN,
            "research-staff": UserRole.RESEARCHER,
            "it-admins": UserRole.ADMIN,
        }

        roles = []
        for group in groups:
            group_lower = group.lower()
            for pattern, role in role_mapping.items():
                if pattern.lower() in group_lower:
                    if role not in roles:
                        roles.append(role)

        # Default to patient if no roles found
        if not roles:
            roles = [UserRole.PATIENT]

        logger.info(
            f"Mapped {len(groups)} groups to {len(roles)} roles",
            extra={
                "event": "group_role_mapping",
                "provider": provider.value,
                "groups": groups,
                "roles": [r.value for r in roles],
            },
        )

        return roles

    async def create_genomevault_user(
        self, provider: OIDCProvider, user_info: OIDCUserInfo
    ) -> User:
        """Create or update GenomeVault user from OIDC user info."""
        # Map groups to roles
        roles = self.map_groups_to_roles(user_info.groups, provider)

        # Aggregate scopes from all roles
        scopes = set()
        for role in roles:
            scopes.update(ROLE_SCOPES.get(role, set()))

        # Create user object
        user = User(
            username=user_info.preferred_username or user_info.email or user_info.sub,
            email=user_info.email,
            full_name=user_info.name,
            disabled=False,
            roles=roles,
            scopes=list(scopes),
            mfa_enabled=provider
            in [OIDCProvider.OKTA, OIDCProvider.AZURE_AD],  # Enterprise providers usually have MFA
            npi_number=user_info.npi_number,
            organization_id=user_info.organization_id,
            baa_signed=user_info.baa_signed,
        )

        # Store user in cache (in production, would sync with database)
        user_cache_key = f"user:{user.username}"
        redis_client.setex(user_cache_key, timedelta(hours=1), user.json())

        logger.info(
            f"Created/updated user from {provider}",
            extra={
                "event": "oidc_user_created",
                "provider": provider.value,
                "username": user.username,
                "roles": [r.value for r in roles],
                "has_npi": bool(user.npi_number),
                "baa_signed": user.baa_signed,
            },
        )

        return user

    async def authenticate_oidc_user(
        self,
        provider: OIDCProvider,
        code: str,
        state: str,
        nonce: str,
        code_verifier: Optional[str] = None,
        request: Optional[Request] = None,
    ) -> Token:
        """Complete OIDC authentication flow and return GenomeVault tokens."""
        # Exchange code for tokens
        oidc_tokens = await self.exchange_code_for_tokens(provider, code, code_verifier)

        # Validate ID token if present
        if "id_token" in oidc_tokens:
            await self.validate_id_token(provider, oidc_tokens["id_token"], nonce)

        # Get user info
        user_info = await self.get_user_info(provider, oidc_tokens["access_token"])

        # Create or update GenomeVault user
        user = await self.create_genomevault_user(provider, user_info)

        # Generate session ID
        import secrets

        session_id = secrets.token_urlsafe(32)

        # Create GenomeVault tokens
        token_data = TokenData(
            username=user.username,
            user_id=user.username,
            scopes=user.scopes,
            roles=[role.value for role in user.roles],
            session_id=session_id,
            mfa_verified=user.mfa_enabled,  # Trust enterprise MFA
            npi_number=user.npi_number,
            organization_id=user.organization_id,
            baa_signed=user.baa_signed,
        )

        access_token = create_access_token(token_data)

        # Get client info for refresh token
        ip_address = None
        user_agent = None
        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")

        refresh_token = create_refresh_token(
            user, session_id=session_id, ip_address=ip_address, user_agent=user_agent
        )

        # Store OIDC tokens for later use (e.g., API calls to provider)
        oidc_cache_key = f"oidc_tokens:{user.username}:{provider.value}"
        redis_client.setex(
            oidc_cache_key,
            timedelta(hours=1),
            json.dumps(
                {
                    "access_token": oidc_tokens.get("access_token"),
                    "refresh_token": oidc_tokens.get("refresh_token"),
                    "expires_in": oidc_tokens.get("expires_in", 3600),
                }
            ),
        )

        logger.info(
            f"OIDC authentication successful for {user.username} via {provider}",
            extra={
                "event": "oidc_authentication_success",
                "provider": provider.value,
                "username": user.username,
                "session_id": session_id,
                "ip_address": ip_address,
            },
        )

        from genomevault.api.auth.oauth2 import ACCESS_TOKEN_EXPIRE_MINUTES

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=refresh_token,
            scope=" ".join(user.scopes),
        )

    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


# Global OIDC manager instance
oidc_manager = OIDCManager()
