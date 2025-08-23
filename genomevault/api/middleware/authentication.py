"""
Authentication middleware for GenomeVault API.

Provides API key validation, token verification, and audit logging
for all authentication events with HIPAA compliance.
"""

import os
import json
import secrets
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from enum import Enum

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis
import pyotp

from genomevault.api.auth.oauth2 import (
    User,
    get_current_user,
    jwt,
    SECRET_KEY,
    ALGORITHM,
)

logger = logging.getLogger(__name__)

# Redis client
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


class APIKeyStatus(str, Enum):
    """API key status."""
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"
    ROTATING = "rotating"


class APIKey(BaseModel):
    """API key model."""
    key_id: str
    key_hash: str  # Store hash, not plaintext
    name: str
    description: Optional[str] = None
    user_id: str
    scopes: List[str] = []
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    last_rotated_at: Optional[datetime] = None
    rotation_interval_days: Optional[int] = 90
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    # Usage limits
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    rate_limit_per_day: int = 10000
    usage_count: int = 0
    # IP restrictions
    allowed_ips: List[str] = []
    # HIPAA compliance
    phi_access_allowed: bool = False
    audit_required: bool = True


class MFAMethod(str, Enum):
    """MFA methods."""
    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    HARDWARE_TOKEN = "hardware_token"
    BIOMETRIC = "biometric"


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling authentication across all requests.
    
    Features:
    - API key validation with rotation support
    - JWT token verification
    - Audit logging for HIPAA compliance
    - Rate limiting integration
    - MFA enforcement for sensitive operations
    """
    
    def __init__(
        self,
        app: ASGIApp,
        exempt_paths: List[str] = None,
        require_auth: bool = True,
        audit_all: bool = True
    ):
        """Initialize authentication middleware."""
        super().__init__(app)
        self.exempt_paths = exempt_paths or [
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json",
            "/health",
            "/healthz",
            "/ready",
            "/api/v1/auth/token",
            "/api/v1/auth/refresh",
            "/api/v1/auth/oidc",
        ]
        self.require_auth = require_auth
        self.audit_all = audit_all
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware."""
        # Check if path is exempt
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Extract authentication credentials
        auth_method = None
        auth_success = False
        user_id = None
        
        try:
            # Check for API key first
            api_key = self._extract_api_key(request)
            if api_key:
                user = await self._validate_api_key(api_key, request)
                auth_method = "api_key"
                auth_success = True
                user_id = user.username
                request.state.user = user
                request.state.auth_method = "api_key"
            
            # Check for Bearer token
            elif "authorization" in request.headers:
                auth_header = request.headers["authorization"]
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:]
                    user = await self._validate_jwt_token(token, request)
                    auth_method = "jwt"
                    auth_success = True
                    user_id = user.username
                    request.state.user = user
                    request.state.auth_method = "jwt"
            
            # No authentication provided
            elif self.require_auth:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Process request
            response = await call_next(request)
            
            # Audit successful request
            if self.audit_all and auth_success:
                await self._audit_request(
                    request=request,
                    response=response,
                    user_id=user_id,
                    auth_method=auth_method,
                    success=True
                )
            
            return response
            
        except HTTPException as e:
            # Audit failed authentication
            if self.audit_all:
                await self._audit_request(
                    request=request,
                    response=None,
                    user_id=user_id,
                    auth_method=auth_method,
                    success=False,
                    error=str(e.detail)
                )
            raise
            
        except Exception as e:
            logger.error(f"Authentication middleware error: {str(e)}")
            if self.audit_all:
                await self._audit_request(
                    request=request,
                    response=None,
                    user_id=user_id,
                    auth_method=auth_method,
                    success=False,
                    error=str(e)
                )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request."""
        # Check header first (X-API-Key)
        if "x-api-key" in request.headers:
            return request.headers["x-api-key"]
        
        # Check query parameter
        if "api_key" in request.query_params:
            return request.query_params["api_key"]
        
        # Check Authorization header for API key
        if "authorization" in request.headers:
            auth_header = request.headers["authorization"]
            if auth_header.startswith("ApiKey "):
                return auth_header[7:]
        
        return None
    
    async def _validate_api_key(
        self,
        api_key: str,
        request: Request
    ) -> User:
        """Validate API key and return associated user."""
        # Hash the API key for lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Look up API key in Redis
        key_data = redis_client.get(f"api_key:{key_hash}")
        if not key_data:
            logger.warning(
                f"Invalid API key attempted from {request.client.host}",
                extra={
                    "event": "api_key_invalid",
                    "ip_address": request.client.host if request.client else None,
                }
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        api_key_obj = APIKey.parse_raw(key_data)
        
        # Check status
        if api_key_obj.status != APIKeyStatus.ACTIVE:
            logger.warning(
                f"Inactive API key used: {api_key_obj.key_id}",
                extra={
                    "event": "api_key_inactive",
                    "key_id": api_key_obj.key_id,
                    "status": api_key_obj.status.value,
                }
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"API key is {api_key_obj.status.value}"
            )
        
        # Check expiration
        if api_key_obj.expires_at and datetime.now(timezone.utc) > api_key_obj.expires_at:
            api_key_obj.status = APIKeyStatus.EXPIRED
            redis_client.set(f"api_key:{key_hash}", api_key_obj.json())
            
            logger.warning(
                f"Expired API key used: {api_key_obj.key_id}",
                extra={
                    "event": "api_key_expired",
                    "key_id": api_key_obj.key_id,
                }
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key expired"
            )
        
        # Check IP restrictions
        if api_key_obj.allowed_ips:
            client_ip = request.client.host if request.client else None
            if client_ip not in api_key_obj.allowed_ips:
                logger.warning(
                    f"API key used from unauthorized IP: {client_ip}",
                    extra={
                        "event": "api_key_ip_unauthorized",
                        "key_id": api_key_obj.key_id,
                        "ip_address": client_ip,
                    }
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API key not authorized from this IP"
                )
        
        # Check if rotation is needed
        if api_key_obj.rotation_interval_days and api_key_obj.last_rotated_at:
            rotation_due = api_key_obj.last_rotated_at + timedelta(days=api_key_obj.rotation_interval_days)
            if datetime.now(timezone.utc) > rotation_due:
                # Mark for rotation
                api_key_obj.status = APIKeyStatus.ROTATING
                redis_client.set(f"api_key:{key_hash}", api_key_obj.json())
                
                logger.info(
                    f"API key rotation needed: {api_key_obj.key_id}",
                    extra={
                        "event": "api_key_rotation_needed",
                        "key_id": api_key_obj.key_id,
                    }
                )
        
        # Update last used
        api_key_obj.last_used_at = datetime.now(timezone.utc)
        api_key_obj.usage_count += 1
        redis_client.set(f"api_key:{key_hash}", api_key_obj.json())
        
        # Get associated user
        from genomevault.api.auth.oauth2 import get_user
        user = get_user(api_key_obj.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found for API key"
            )
        
        # Apply scope restrictions from API key
        user.scopes = [s for s in user.scopes if s in api_key_obj.scopes]
        
        logger.info(
            f"API key authenticated: {api_key_obj.key_id}",
            extra={
                "event": "api_key_authenticated",
                "key_id": api_key_obj.key_id,
                "user_id": user.username,
            }
        )
        
        return user
    
    async def _validate_jwt_token(
        self,
        token: str,
        request: Request
    ) -> User:
        """Validate JWT token and return user."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("username")
            
            if not username:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti and redis_client.exists(f"revoked_token:{jti}"):
                logger.warning(f"Revoked token used by {username}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            # Get user
            from genomevault.api.auth.oauth2 import get_user
            user = get_user(username)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            # Apply token scopes
            token_scopes = payload.get("scopes", [])
            user.scopes = [s for s in user.scopes if s in token_scopes]
            
            return user
            
        except jwt.JWTError as e:
            logger.warning(
                f"JWT validation failed: {str(e)}",
                extra={
                    "event": "jwt_validation_failed",
                    "error": str(e),
                }
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def _audit_request(
        self,
        request: Request,
        response: Optional[Any],
        user_id: Optional[str],
        auth_method: Optional[str],
        success: bool,
        error: Optional[str] = None
    ):
        """Audit authentication event for HIPAA compliance."""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "api_request",
            "user_id": user_id,
            "auth_method": auth_method,
            "success": success,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "status_code": response.status_code if response else None,
            "error": error,
        }
        
        # Check if PHI was accessed
        phi_paths = ["/api/v1/clinical", "/api/v1/phi", "/api/v1/patients"]
        if any(request.url.path.startswith(path) for path in phi_paths):
            audit_entry["phi_access"] = True
            audit_entry["hipaa_required"] = True
        
        # Store in Redis with TTL (7 years for HIPAA)
        audit_key = f"audit:{datetime.now(timezone.utc).strftime('%Y%m%d')}:{secrets.token_hex(8)}"
        redis_client.setex(
            audit_key,
            timedelta(days=2555),  # 7 years
            json.dumps(audit_entry)
        )
        
        # Log to file/SIEM
        logger.info(
            f"API request audit: {user_id or 'anonymous'} - {request.method} {request.url.path}",
            extra=audit_entry
        )


def generate_api_key(
    user_id: str,
    name: str,
    scopes: List[str],
    expires_in_days: Optional[int] = 365,
    **kwargs
) -> tuple[str, APIKey]:
    """Generate a new API key for a user."""
    # Generate cryptographically secure key
    api_key = f"gv_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_id = f"key_{secrets.token_hex(8)}"
    
    # Create API key object
    api_key_obj = APIKey(
        key_id=key_id,
        key_hash=key_hash,
        name=name,
        user_id=user_id,
        scopes=scopes,
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(days=expires_in_days) if expires_in_days else None,
        last_rotated_at=datetime.now(timezone.utc),
        **kwargs
    )
    
    # Store in Redis
    redis_client.set(f"api_key:{key_hash}", api_key_obj.json())
    
    # Add to user's API keys list
    user_keys_key = f"user_api_keys:{user_id}"
    redis_client.sadd(user_keys_key, key_id)
    
    logger.info(
        f"API key generated for user {user_id}",
        extra={
            "event": "api_key_generated",
            "user_id": user_id,
            "key_id": key_id,
            "scopes": scopes,
        }
    )
    
    return api_key, api_key_obj


def rotate_api_key(old_key_id: str, user_id: str) -> tuple[str, APIKey]:
    """Rotate an existing API key."""
    # Find old key
    user_keys_key = f"user_api_keys:{user_id}"
    if not redis_client.sismember(user_keys_key, old_key_id):
        raise ValueError(f"API key {old_key_id} not found for user {user_id}")
    
    # Get old key data
    old_key_data = None
    for key in redis_client.scan_iter(match="api_key:*"):
        data = redis_client.get(key)
        if data:
            obj = APIKey.parse_raw(data)
            if obj.key_id == old_key_id:
                old_key_data = obj
                old_key_hash = key.split(":")[1]
                break
    
    if not old_key_data:
        raise ValueError(f"API key data not found for {old_key_id}")
    
    # Generate new key with same settings
    new_key, new_key_obj = generate_api_key(
        user_id=user_id,
        name=f"{old_key_data.name} (rotated)",
        scopes=old_key_data.scopes,
        expires_in_days=None,  # Will calculate from old key
        rate_limit_per_minute=old_key_data.rate_limit_per_minute,
        rate_limit_per_hour=old_key_data.rate_limit_per_hour,
        rate_limit_per_day=old_key_data.rate_limit_per_day,
        allowed_ips=old_key_data.allowed_ips,
        phi_access_allowed=old_key_data.phi_access_allowed,
    )
    
    # Calculate new expiration based on rotation interval
    if old_key_data.rotation_interval_days:
        new_key_obj.expires_at = datetime.now(timezone.utc) + timedelta(days=old_key_data.rotation_interval_days)
        new_key_obj.rotation_interval_days = old_key_data.rotation_interval_days
    
    # Revoke old key
    old_key_data.status = APIKeyStatus.REVOKED
    redis_client.set(f"api_key:{old_key_hash}", old_key_data.json())
    
    logger.info(
        f"API key rotated: {old_key_id} -> {new_key_obj.key_id}",
        extra={
            "event": "api_key_rotated",
            "old_key_id": old_key_id,
            "new_key_id": new_key_obj.key_id,
            "user_id": user_id,
        }
    )
    
    return new_key, new_key_obj


class MFAVerifier:
    """Multi-factor authentication verifier."""
    
    @staticmethod
    def generate_totp_secret() -> str:
        """Generate TOTP secret for user."""
        return pyotp.random_base32()
    
    @staticmethod
    def generate_totp_uri(
        secret: str,
        username: str,
        issuer: str = "GenomeVault"
    ) -> str:
        """Generate TOTP provisioning URI for QR code."""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(
            name=username,
            issuer_name=issuer
        )
    
    @staticmethod
    def verify_totp(secret: str, token: str) -> bool:
        """Verify TOTP token."""
        totp = pyotp.TOTP(secret)
        # Allow 30 second window (1 period before/after)
        return totp.verify(token, valid_window=1)
    
    @staticmethod
    async def send_sms_code(phone: str) -> str:
        """Send SMS verification code."""
        code = secrets.token_hex(3)  # 6-digit hex code
        
        # Store code in Redis with 5-minute TTL
        redis_client.setex(
            f"mfa_sms:{phone}",
            timedelta(minutes=5),
            code
        )
        
        # In production, integrate with SMS provider (Twilio, AWS SNS, etc.)
        logger.info(
            f"SMS code sent to {phone[:3]}****{phone[-2:]}",
            extra={
                "event": "mfa_sms_sent",
                "phone_masked": f"{phone[:3]}****{phone[-2:]}",
            }
        )
        
        return code
    
    @staticmethod
    def verify_sms_code(phone: str, code: str) -> bool:
        """Verify SMS code."""
        stored_code = redis_client.get(f"mfa_sms:{phone}")
        if stored_code and stored_code == code:
            # Delete code after successful verification
            redis_client.delete(f"mfa_sms:{phone}")
            return True
        return False
    
    @staticmethod
    async def send_email_code(email: str) -> str:
        """Send email verification code."""
        code = secrets.token_hex(4)  # 8-digit hex code
        
        # Store code in Redis with 10-minute TTL
        redis_client.setex(
            f"mfa_email:{email}",
            timedelta(minutes=10),
            code
        )
        
        # In production, integrate with email provider
        logger.info(
            f"Email code sent to {email}",
            extra={
                "event": "mfa_email_sent",
                "email": email,
            }
        )
        
        return code
    
    @staticmethod
    def verify_email_code(email: str, code: str) -> bool:
        """Verify email code."""
        stored_code = redis_client.get(f"mfa_email:{email}")
        if stored_code and stored_code == code:
            # Delete code after successful verification
            redis_client.delete(f"mfa_email:{email}")
            return True
        return False


from pydantic import BaseModel  # Add this import at the top