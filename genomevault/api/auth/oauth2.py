"""
OAuth2 implementation with JWT tokens for GenomeVault API.

Provides secure authentication with JWT tokens, refresh tokens, and
comprehensive scope-based authorization for HIPAA-compliant access control.
"""

import os
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Set
from enum import Enum

from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, ValidationError
import redis
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Configuration from environment
SECRET_KEY = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
MFA_REQUIRED_SCOPES = os.getenv("MFA_REQUIRED_SCOPES", "write:clinical,admin:all").split(",")

# Redis configuration for token storage
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    scopes={
        "read:genomic": "Read genomic data",
        "write:genomic": "Write genomic data",
        "read:clinical": "Read clinical data",
        "write:clinical": "Write clinical data",
        "read:phi": "Read PHI data (requires MFA)",
        "write:phi": "Write PHI data (requires MFA)",
        "admin:users": "Manage users",
        "admin:system": "System administration",
        "admin:all": "Full administrative access",
        "pir:query": "Execute PIR queries",
        "zk:prove": "Generate zero-knowledge proofs",
        "zk:verify": "Verify zero-knowledge proofs",
        "federated:participate": "Participate in federated learning",
        "blockchain:write": "Write to blockchain",
    }
)


class TokenType(str, Enum):
    """Token types in the system."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    SERVICE = "service"


class UserRole(str, Enum):
    """User roles for RBAC."""
    PATIENT = "patient"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    ADMIN = "admin"
    SERVICE = "service"


class Token(BaseModel):
    """OAuth2 token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: str = ""


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: List[str] = []
    roles: List[str] = []
    session_id: Optional[str] = None
    mfa_verified: bool = False
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    token_type: TokenType = TokenType.ACCESS
    # HIPAA compliance fields
    npi_number: Optional[str] = None
    organization_id: Optional[str] = None
    baa_signed: bool = False


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[UserRole] = [UserRole.PATIENT]
    scopes: List[str] = []
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    # HIPAA fields
    npi_number: Optional[str] = None
    organization_id: Optional[str] = None
    baa_signed: bool = False
    last_login: Optional[datetime] = None
    password_changed_at: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None


class UserInDB(User):
    """User model with password."""
    hashed_password: str


class RefreshTokenData(BaseModel):
    """Refresh token storage model."""
    user_id: str
    username: str
    token_id: str
    issued_at: datetime
    expires_at: datetime
    used: bool = False
    revoked: bool = False
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


# Role to scope mapping
ROLE_SCOPES: Dict[UserRole, Set[str]] = {
    UserRole.PATIENT: {
        "read:genomic",
        "read:clinical",
        "pir:query",
    },
    UserRole.CLINICIAN: {
        "read:genomic",
        "read:clinical",
        "write:clinical",
        "read:phi",
        "write:phi",
        "pir:query",
        "zk:verify",
    },
    UserRole.RESEARCHER: {
        "read:genomic",
        "read:clinical",
        "pir:query",
        "zk:prove",
        "zk:verify",
        "federated:participate",
    },
    UserRole.ADMIN: {
        "admin:users",
        "admin:system",
        "admin:all",
        "blockchain:write",
    },
    UserRole.SERVICE: {
        "read:genomic",
        "write:genomic",
        "pir:query",
        "zk:prove",
        "zk:verify",
        "federated:participate",
        "blockchain:write",
    },
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(
    data: TokenData,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    to_encode = data.dict(exclude_none=True)
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "token_type": TokenType.ACCESS.value,
        "jti": secrets.token_urlsafe(16),  # JWT ID for tracking
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    # Store token metadata in Redis for tracking
    token_key = f"token:access:{to_encode['jti']}"
    redis_client.setex(
        token_key,
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        encoded_jwt
    )
    
    # Log token creation for audit
    logger.info(
        f"Access token created for user {data.username}",
        extra={
            "event": "token_created",
            "token_type": "access",
            "username": data.username,
            "scopes": data.scopes,
            "session_id": data.session_id,
        }
    )
    
    return encoded_jwt


def create_refresh_token(
    user: User,
    session_id: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> str:
    """Create a refresh token."""
    token_id = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    refresh_data = RefreshTokenData(
        user_id=user.username,
        username=user.username,
        token_id=token_id,
        issued_at=datetime.now(timezone.utc),
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    # Store refresh token in Redis
    redis_key = f"refresh_token:{token_id}"
    redis_client.setex(
        redis_key,
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        refresh_data.json()
    )
    
    # Store user's refresh tokens list
    user_tokens_key = f"user_refresh_tokens:{user.username}"
    redis_client.sadd(user_tokens_key, token_id)
    redis_client.expire(user_tokens_key, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    
    # Create JWT refresh token
    token_data = {
        "sub": user.username,
        "token_id": token_id,
        "token_type": TokenType.REFRESH.value,
        "exp": expires_at,
        "iat": datetime.now(timezone.utc),
        "session_id": session_id,
    }
    
    encoded_jwt = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    
    logger.info(
        f"Refresh token created for user {user.username}",
        extra={
            "event": "token_created",
            "token_type": "refresh",
            "username": user.username,
            "session_id": session_id,
            "ip_address": ip_address,
        }
    )
    
    return encoded_jwt


async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme)
) -> User:
    """Get current authenticated user from token."""
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("username")
        token_type: str = payload.get("token_type")
        jti: str = payload.get("jti")
        
        if username is None or token_type != TokenType.ACCESS.value:
            raise credentials_exception
        
        # Check if token is revoked
        if jti and redis_client.exists(f"revoked_token:{jti}"):
            logger.warning(f"Revoked token used by {username}")
            raise credentials_exception
        
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(
            username=username,
            scopes=token_scopes,
            mfa_verified=payload.get("mfa_verified", False),
            npi_number=payload.get("npi_number"),
            organization_id=payload.get("organization_id"),
            baa_signed=payload.get("baa_signed", False),
        )
        
    except (JWTError, ValidationError):
        raise credentials_exception
    
    # Get user from database (mock implementation)
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    
    # Check required scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            # Check if MFA is required for this scope
            if scope in MFA_REQUIRED_SCOPES and not token_data.mfa_verified:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="MFA verification required for this scope",
                    headers={"WWW-Authenticate": authenticate_value},
                )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )
    
    # HIPAA compliance checks
    if any(scope.startswith("write:phi") for scope in security_scopes.scopes):
        if not token_data.baa_signed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="BAA agreement required for PHI access",
            )
        
        if not token_data.npi_number:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="NPI number required for PHI access",
            )
    
    # Log successful authentication
    logger.info(
        f"User {username} authenticated successfully",
        extra={
            "event": "authentication_success",
            "username": username,
            "scopes": security_scopes.scopes,
            "mfa_verified": token_data.mfa_verified,
        }
    )
    
    return user


async def get_current_active_user(
    current_user: User = Security(get_current_user, scopes=[])
) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    # Check if account is locked
    if current_user.account_locked_until:
        if datetime.now(timezone.utc) < current_user.account_locked_until:
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is locked due to multiple failed login attempts",
            )
        else:
            # Unlock account
            current_user.account_locked_until = None
            current_user.failed_login_attempts = 0
    
    return current_user


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user with username and password."""
    user = get_user_db(username)
    if not user:
        # Log failed attempt
        logger.warning(
            f"Authentication failed - user not found: {username}",
            extra={"event": "authentication_failed", "reason": "user_not_found"}
        )
        return None
    
    if not verify_password(password, user.hashed_password):
        # Increment failed login attempts
        user.failed_login_attempts += 1
        
        # Lock account after 5 failed attempts
        if user.failed_login_attempts >= 5:
            user.account_locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
            logger.warning(
                f"Account locked due to failed attempts: {username}",
                extra={"event": "account_locked", "username": username}
            )
        
        logger.warning(
            f"Authentication failed - invalid password: {username}",
            extra={
                "event": "authentication_failed",
                "reason": "invalid_password",
                "failed_attempts": user.failed_login_attempts
            }
        )
        return None
    
    # Reset failed attempts on successful login
    user.failed_login_attempts = 0
    user.last_login = datetime.now(timezone.utc)
    
    # Check password age (90 days for HIPAA compliance)
    if user.password_changed_at:
        password_age = datetime.now(timezone.utc) - user.password_changed_at
        if password_age > timedelta(days=90):
            logger.warning(
                f"Password expired for user: {username}",
                extra={"event": "password_expired", "username": username}
            )
            # In production, would force password change
    
    logger.info(
        f"User authenticated successfully: {username}",
        extra={"event": "authentication_success", "username": username}
    )
    
    return user


def get_user(username: str) -> Optional[User]:
    """Get user from database (mock implementation)."""
    # In production, this would query a real database
    fake_users_db = {
        "clinician": {
            "username": "clinician",
            "email": "clinician@genomevault.io",
            "full_name": "Dr. Jane Smith",
            "disabled": False,
            "roles": [UserRole.CLINICIAN],
            "scopes": list(ROLE_SCOPES[UserRole.CLINICIAN]),
            "mfa_enabled": True,
            "npi_number": "1234567890",
            "organization_id": "org_123",
            "baa_signed": True,
        },
        "researcher": {
            "username": "researcher",
            "email": "researcher@genomevault.io",
            "full_name": "Dr. John Doe",
            "disabled": False,
            "roles": [UserRole.RESEARCHER],
            "scopes": list(ROLE_SCOPES[UserRole.RESEARCHER]),
            "mfa_enabled": False,
        },
        "admin": {
            "username": "admin",
            "email": "admin@genomevault.io",
            "full_name": "Admin User",
            "disabled": False,
            "roles": [UserRole.ADMIN],
            "scopes": list(ROLE_SCOPES[UserRole.ADMIN]),
            "mfa_enabled": True,
        },
    }
    
    if username in fake_users_db:
        return User(**fake_users_db[username])
    return None


def get_user_db(username: str) -> Optional[UserInDB]:
    """Get user with password from database (mock implementation)."""
    user = get_user(username)
    if user:
        # In production, this would be retrieved from database
        return UserInDB(
            **user.dict(),
            hashed_password=get_password_hash("genomevault123")
        )
    return None


def revoke_token(token_jti: str, reason: str = "manual_revocation"):
    """Revoke a token by its JTI."""
    redis_key = f"revoked_token:{token_jti}"
    redis_client.setex(
        redis_key,
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS + 1),  # Keep longer than refresh token
        reason
    )
    
    logger.info(
        f"Token revoked: {token_jti}",
        extra={"event": "token_revoked", "jti": token_jti, "reason": reason}
    )


def revoke_all_user_tokens(username: str, reason: str = "security"):
    """Revoke all tokens for a user."""
    # Get all user's refresh tokens
    user_tokens_key = f"user_refresh_tokens:{username}"
    token_ids = redis_client.smembers(user_tokens_key)
    
    for token_id in token_ids:
        redis_key = f"refresh_token:{token_id}"
        token_data = redis_client.get(redis_key)
        if token_data:
            redis_client.delete(redis_key)
    
    # Clear user's token list
    redis_client.delete(user_tokens_key)
    
    logger.info(
        f"All tokens revoked for user: {username}",
        extra={
            "event": "all_tokens_revoked",
            "username": username,
            "reason": reason,
            "token_count": len(token_ids)
        }
    )


async def refresh_access_token(refresh_token: str) -> Token:
    """Refresh an access token using a refresh token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_id: str = payload.get("token_id")
        token_type: str = payload.get("token_type")
        
        if not username or not token_id or token_type != TokenType.REFRESH.value:
            raise credentials_exception
        
        # Check if refresh token exists and is valid
        redis_key = f"refresh_token:{token_id}"
        stored_token = redis_client.get(redis_key)
        
        if not stored_token:
            logger.warning(f"Invalid refresh token used by {username}")
            raise credentials_exception
        
        refresh_data = RefreshTokenData.parse_raw(stored_token)
        
        if refresh_data.used:
            # Token reuse detected - potential security breach
            logger.error(
                f"Refresh token reuse detected for user {username}",
                extra={
                    "event": "token_reuse_detected",
                    "username": username,
                    "token_id": token_id
                }
            )
            # Revoke all user tokens as security measure
            revoke_all_user_tokens(username, "token_reuse_detected")
            raise credentials_exception
        
        if refresh_data.revoked:
            logger.warning(f"Revoked refresh token used by {username}")
            raise credentials_exception
        
        # Mark token as used
        refresh_data.used = True
        redis_client.setex(
            redis_key,
            timedelta(minutes=5),  # Keep for audit trail
            refresh_data.json()
        )
        
        # Get user and create new tokens
        user = get_user(username)
        if not user:
            raise credentials_exception
        
        # Create new access token
        token_data = TokenData(
            username=user.username,
            user_id=user.username,
            scopes=user.scopes,
            roles=[role.value for role in user.roles],
            session_id=payload.get("session_id"),
            npi_number=user.npi_number,
            organization_id=user.organization_id,
            baa_signed=user.baa_signed,
        )
        
        access_token = create_access_token(token_data)
        
        # Create new refresh token
        new_refresh_token = create_refresh_token(
            user,
            session_id=payload.get("session_id"),
            ip_address=refresh_data.ip_address,
            user_agent=refresh_data.user_agent
        )
        
        logger.info(
            f"Token refreshed for user {username}",
            extra={
                "event": "token_refreshed",
                "username": username,
                "old_token_id": token_id
            }
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=new_refresh_token,
            scope=" ".join(user.scopes)
        )
        
    except (JWTError, ValidationError) as e:
        logger.error(f"Token refresh failed: {str(e)}")
        raise credentials_exception


# Dependency for requiring specific scopes
def require_scopes(*required_scopes: str):
    """Create a dependency that requires specific scopes."""
    def scope_checker(
        current_user: User = Security(get_current_user, scopes=list(required_scopes))
    ):
        return current_user
    return scope_checker


# Dependency for requiring MFA
async def require_mfa(
    current_user: User = Depends(get_current_active_user),
    token: str = Depends(oauth2_scheme)
) -> User:
    """Require MFA verification for sensitive operations."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        mfa_verified = payload.get("mfa_verified", False)
        
        if not mfa_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="MFA verification required",
            )
        
        return current_user
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )