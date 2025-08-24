"""
API Key Authentication for GenomeVault.

Provides secure API key-based authentication with role-based access control,
rate limiting, and PHI-safe audit logging. Supports both header and query
parameter authentication methods.
"""

import os
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set
from enum import Enum
from dataclasses import dataclass

from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.api_key import APIKeyQuery, APIKeyHeader, APIKeyCookie
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class APIKeyScope(str, Enum):
    """API key access scopes for different operations."""
    
    # Read-only access
    READ = "read"
    READ_HDC = "read:hdc"
    READ_METRICS = "read:metrics"
    READ_HEALTH = "read:health"
    
    # Write access
    WRITE = "write"
    WRITE_HDC = "write:hdc"
    WRITE_ZK = "write:zk"
    WRITE_PIR = "write:pir"
    
    # Admin access
    ADMIN = "admin"
    ADMIN_USERS = "admin:users"
    ADMIN_KEYS = "admin:keys"
    
    # Clinical access (high sensitivity)
    CLINICAL_READ = "clinical:read"
    CLINICAL_WRITE = "clinical:write"
    CLINICAL_ADMIN = "clinical:admin"


class APIKeyType(str, Enum):
    """Types of API keys."""
    
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    RESEARCH = "research"
    CLINICAL = "clinical"
    SERVICE = "service"


@dataclass
class APIKeyInfo:
    """Information about an API key."""
    
    key_id: str
    key_hash: str
    name: str
    description: str
    key_type: APIKeyType
    scopes: Set[APIKeyScope]
    rate_limit_per_hour: int
    rate_limit_per_minute: int
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    is_active: bool
    client_ip_whitelist: Optional[List[str]]
    user_agent_pattern: Optional[str]
    
    def has_scope(self, scope: APIKeyScope) -> bool:
        """Check if API key has required scope."""
        return scope in self.scopes or APIKeyScope.ADMIN in self.scopes
    
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_ip_allowed(self, client_ip: str) -> bool:
        """Check if client IP is allowed."""
        if not self.client_ip_whitelist:
            return True
        return client_ip in self.client_ip_whitelist


class APIKeyManager:
    """Manages API keys and authentication."""
    
    def __init__(self, redis_client=None):
        """Initialize API key manager.
        
        Args:
            redis_client: Redis client for caching and rate limiting
        """
        self.redis_client = redis_client
        self._keys_cache: Dict[str, APIKeyInfo] = {}
        self._load_keys()
    
    def _load_keys(self):
        """Load API keys from environment and configuration."""
        # Load keys from environment variables
        # In production, these should come from a secure key management system
        
        # Example development key
        dev_key = os.getenv("GENOMEVAULT_DEV_API_KEY", "gv_dev_" + secrets.token_urlsafe(32))
        dev_key_hash = self._hash_key(dev_key)
        
        self._keys_cache[dev_key_hash] = APIKeyInfo(
            key_id="dev_key_001",
            key_hash=dev_key_hash,
            name="Development Key",
            description="Development and testing API key",
            key_type=APIKeyType.DEVELOPMENT,
            scopes={
                APIKeyScope.READ, APIKeyScope.READ_HDC, APIKeyScope.READ_METRICS,
                APIKeyScope.READ_HEALTH, APIKeyScope.WRITE_HDC
            },
            rate_limit_per_hour=1000,
            rate_limit_per_minute=50,
            created_at=datetime.utcnow(),
            expires_at=None,
            last_used_at=None,
            is_active=True,
            client_ip_whitelist=None,
            user_agent_pattern=None
        )
        
        # Example production key
        prod_key = os.getenv("GENOMEVAULT_PROD_API_KEY")
        if prod_key:
            prod_key_hash = self._hash_key(prod_key)
            self._keys_cache[prod_key_hash] = APIKeyInfo(
                key_id="prod_key_001",
                key_hash=prod_key_hash,
                name="Production Key",
                description="Production API key",
                key_type=APIKeyType.PRODUCTION,
                scopes={
                    APIKeyScope.READ, APIKeyScope.WRITE, APIKeyScope.READ_HDC,
                    APIKeyScope.WRITE_HDC, APIKeyScope.WRITE_ZK, APIKeyScope.WRITE_PIR,
                    APIKeyScope.READ_METRICS
                },
                rate_limit_per_hour=10000,
                rate_limit_per_minute=500,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=365),
                last_used_at=None,
                is_active=True,
                client_ip_whitelist=None,
                user_agent_pattern=None
            )
        
        # Example clinical key (highest security)
        clinical_key = os.getenv("GENOMEVAULT_CLINICAL_API_KEY")
        if clinical_key:
            clinical_key_hash = self._hash_key(clinical_key)
            self._keys_cache[clinical_key_hash] = APIKeyInfo(
                key_id="clinical_key_001",
                key_hash=clinical_key_hash,
                name="Clinical Research Key",
                description="Clinical research API key with PHI access",
                key_type=APIKeyType.CLINICAL,
                scopes={
                    APIKeyScope.READ, APIKeyScope.CLINICAL_READ, APIKeyScope.CLINICAL_WRITE,
                    APIKeyScope.READ_HDC, APIKeyScope.WRITE_HDC
                },
                rate_limit_per_hour=5000,
                rate_limit_per_minute=200,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=90),
                last_used_at=None,
                is_active=True,
                client_ip_whitelist=["127.0.0.1", "::1"],  # Localhost only for clinical
                user_agent_pattern="GenomeVault-Clinical/*"
            )
    
    def _hash_key(self, api_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(f"genomevault:{api_key}".encode()).hexdigest()
    
    def validate_api_key(self, api_key: str, client_ip: str = None, user_agent: str = None) -> APIKeyInfo:
        """Validate an API key and return key info.
        
        Args:
            api_key: The API key to validate
            client_ip: Client IP address
            user_agent: Client user agent
            
        Returns:
            APIKeyInfo object if valid
            
        Raises:
            HTTPException: If key is invalid, expired, or access denied
        """
        key_hash = self._hash_key(api_key)
        key_info = self._keys_cache.get(key_hash)
        
        if not key_info:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        if not key_info.is_active:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="API key is disabled"
            )
        
        if key_info.is_expired():
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="API key has expired"
            )
        
        # Check IP whitelist
        if client_ip and not key_info.is_ip_allowed(client_ip):
            # Log security event without exposing client IP
            print(f"API key access denied for key_id={key_info.key_id}: IP not whitelisted")
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Access denied from this IP address"
            )
        
        # Check user agent pattern
        if (key_info.user_agent_pattern and user_agent and 
            not self._matches_pattern(user_agent, key_info.user_agent_pattern)):
            print(f"API key access denied for key_id={key_info.key_id}: User agent mismatch")
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Access denied: invalid client"
            )
        
        # Update last used timestamp
        key_info.last_used_at = datetime.utcnow()
        
        return key_info
    
    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Simple pattern matching for user agents."""
        import fnmatch
        return fnmatch.fnmatch(value, pattern)
    
    def check_rate_limit(self, key_info: APIKeyInfo, client_ip: str) -> bool:
        """Check if API key has exceeded rate limits.
        
        Args:
            key_info: API key information
            client_ip: Client IP (for additional rate limiting)
            
        Returns:
            True if within rate limit, False if exceeded
        """
        if not self.redis_client:
            # Without Redis, we can't track rate limits effectively
            return True
        
        current_time = int(time.time())
        
        # Check minute rate limit
        minute_key = f"rate_limit:minute:{key_info.key_id}:{current_time // 60}"
        minute_count = self.redis_client.get(minute_key)
        
        if minute_count and int(minute_count) >= key_info.rate_limit_per_minute:
            return False
        
        # Check hour rate limit
        hour_key = f"rate_limit:hour:{key_info.key_id}:{current_time // 3600}"
        hour_count = self.redis_client.get(hour_key)
        
        if hour_count and int(hour_count) >= key_info.rate_limit_per_hour:
            return False
        
        # Increment counters
        pipe = self.redis_client.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 120)  # 2 minutes TTL
        pipe.incr(hour_key)
        pipe.expire(hour_key, 7200)  # 2 hours TTL
        pipe.execute()
        
        return True
    
    def generate_api_key(self, prefix: str = "gv") -> str:
        """Generate a new API key.
        
        Args:
            prefix: Key prefix for identification
            
        Returns:
            New API key string
        """
        return f"{prefix}_{secrets.token_urlsafe(32)}"
    
    def create_api_key(self, 
                      name: str,
                      description: str,
                      key_type: APIKeyType,
                      scopes: Set[APIKeyScope],
                      rate_limit_per_hour: int = 1000,
                      rate_limit_per_minute: int = 50,
                      expires_in_days: Optional[int] = None,
                      client_ip_whitelist: Optional[List[str]] = None) -> tuple[str, APIKeyInfo]:
        """Create a new API key.
        
        Returns:
            Tuple of (api_key, key_info)
        """
        api_key = self.generate_api_key()
        key_hash = self._hash_key(api_key)
        key_id = f"{key_type.value}_{secrets.token_hex(8)}"
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        key_info = APIKeyInfo(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            description=description,
            key_type=key_type,
            scopes=scopes,
            rate_limit_per_hour=rate_limit_per_hour,
            rate_limit_per_minute=rate_limit_per_minute,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            last_used_at=None,
            is_active=True,
            client_ip_whitelist=client_ip_whitelist,
            user_agent_pattern=None
        )
        
        self._keys_cache[key_hash] = key_info
        return api_key, key_info


# Global API key manager instance
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        redis_client = None
        if REDIS_AVAILABLE:
            try:
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                redis_client = redis.from_url(redis_url, decode_responses=True)
                # Test connection
                redis_client.ping()
            except Exception as e:
                print(f"Warning: Could not connect to Redis: {e}")
                redis_client = None
        
        _api_key_manager = APIKeyManager(redis_client)
    
    return _api_key_manager


# FastAPI security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


async def get_api_key_from_request(request: Request,
                                  header_key: Optional[str] = Depends(api_key_header),
                                  query_key: Optional[str] = Depends(api_key_query),
                                  bearer_creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_auth)) -> Optional[str]:
    """Extract API key from request headers or query parameters."""
    
    # Try header first
    if header_key:
        return header_key
    
    # Try bearer token
    if bearer_creds and bearer_creds.scheme.lower() == "bearer":
        return bearer_creds.credentials
    
    # Try query parameter
    if query_key:
        return query_key
    
    return None


async def authenticate_api_key(request: Request,
                              api_key: Optional[str] = Depends(get_api_key_from_request)) -> APIKeyInfo:
    """Authenticate API key and return key information.
    
    Raises:
        HTTPException: If authentication fails
    """
    if not api_key:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    manager = get_api_key_manager()
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")
    
    # Validate API key
    key_info = manager.validate_api_key(api_key, client_ip, user_agent)
    
    # Check rate limits
    if not manager.check_rate_limit(key_info, client_ip):
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit-Hour": str(key_info.rate_limit_per_hour),
                "X-RateLimit-Limit-Minute": str(key_info.rate_limit_per_minute)
            }
        )
    
    # Store key info in request state for audit logging
    request.state.api_key_info = key_info
    
    return key_info


def require_scope(required_scope: APIKeyScope):
    """Dependency factory for requiring specific API key scopes."""
    
    async def scope_checker(key_info: APIKeyInfo = Depends(authenticate_api_key)) -> APIKeyInfo:
        if not key_info.has_scope(required_scope):
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scope: {required_scope.value}"
            )
        return key_info
    
    return scope_checker


def require_clinical_access():
    """Dependency for requiring clinical data access."""
    return require_scope(APIKeyScope.CLINICAL_READ)


def require_admin_access():
    """Dependency for requiring admin access."""
    return require_scope(APIKeyScope.ADMIN)