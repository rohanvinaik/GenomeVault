"""
Advanced Rate Limiting Middleware for GenomeVault.

Implements sophisticated rate limiting using Redis with multiple strategies:
- Token bucket algorithm for smooth rate limiting
- Sliding window for precise rate limiting
- Per-endpoint rate limiting
- IP-based rate limiting
- User-based rate limiting
- Adaptive rate limiting for clinical endpoints
"""

import time
import json
import hashlib
from typing import Dict, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""
    
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window" 
    FIXED_WINDOW = "fixed_window"


class EndpointSensitivity(str, Enum):
    """Endpoint sensitivity levels for rate limiting."""
    
    PUBLIC = "public"          # Health checks, docs
    STANDARD = "standard"      # Regular API endpoints
    COMPUTE = "compute"        # HDC encoding, ZK proofs
    CLINICAL = "clinical"      # Clinical data access
    ADMIN = "admin"           # Administrative functions


@dataclass
class RateLimitConfig:
    """Rate limiting configuration for an endpoint or user type."""
    
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_allowance: int  # Extra requests allowed in burst
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    
    def get_window_limits(self) -> Dict[int, int]:
        """Get limits for different time windows in seconds."""
        return {
            60: self.requests_per_minute,      # 1 minute
            3600: self.requests_per_hour,      # 1 hour  
            86400: self.requests_per_day       # 1 day
        }


class RateLimitManager:
    """Manages rate limiting using Redis."""
    
    # Default rate limit configurations
    DEFAULT_CONFIGS = {
        EndpointSensitivity.PUBLIC: RateLimitConfig(
            requests_per_minute=300,
            requests_per_hour=5000,
            requests_per_day=50000,
            burst_allowance=50
        ),
        EndpointSensitivity.STANDARD: RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=2000,
            requests_per_day=20000,
            burst_allowance=20
        ),
        EndpointSensitivity.COMPUTE: RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000,
            burst_allowance=10
        ),
        EndpointSensitivity.CLINICAL: RateLimitConfig(
            requests_per_minute=20,
            requests_per_hour=200,
            requests_per_day=1000,
            burst_allowance=5
        ),
        EndpointSensitivity.ADMIN: RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=500,
            burst_allowance=2
        )
    }
    
    def __init__(self, redis_client=None):
        """Initialize rate limit manager.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis_client = redis_client
        self._fallback_storage: Dict[str, Dict] = {}  # Fallback when Redis unavailable
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting."""
        # Try to get API key info from request state
        api_key_info = getattr(request.state, 'api_key_info', None)
        if api_key_info:
            return f"api_key:{api_key_info.key_id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        
        # Add user agent for better client identification
        user_agent = request.headers.get("user-agent", "")
        user_agent_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]
        
        return f"ip:{client_ip}:{user_agent_hash}"
    
    def _get_endpoint_sensitivity(self, path: str, method: str) -> EndpointSensitivity:
        """Determine endpoint sensitivity based on path and method."""
        path_lower = path.lower()
        
        # Admin endpoints
        if any(pattern in path_lower for pattern in ['/admin/', '/keys/', '/users/']):
            return EndpointSensitivity.ADMIN
            
        # Clinical endpoints
        if any(pattern in path_lower for pattern in ['/clinical/', '/patient/', '/phi/']):
            return EndpointSensitivity.CLINICAL
            
        # Compute-intensive endpoints  
        if any(pattern in path_lower for pattern in ['/hdc/', '/zk/', '/pir/', '/encode/', '/prove/']):
            return EndpointSensitivity.COMPUTE
            
        # Public endpoints
        if any(pattern in path_lower for pattern in ['/health', '/docs', '/metrics', '/ping']):
            return EndpointSensitivity.PUBLIC
            
        # Default to standard
        return EndpointSensitivity.STANDARD
    
    def _token_bucket_check(self, client_id: str, config: RateLimitConfig) -> Tuple[bool, Dict[str, int]]:
        """Implement token bucket rate limiting algorithm."""
        current_time = time.time()
        bucket_key = f"rate_limit:bucket:{client_id}"
        
        if self.redis_client:
            # Use Redis for distributed rate limiting
            pipe = self.redis_client.pipeline()
            pipe.hgetall(bucket_key)
            result = pipe.execute()[0]
            
            if result:
                last_refill = float(result.get('last_refill', current_time))
                tokens = float(result.get('tokens', config.requests_per_minute))
            else:
                last_refill = current_time
                tokens = config.requests_per_minute
        else:
            # Use in-memory fallback
            bucket_data = self._fallback_storage.get(bucket_key, {
                'last_refill': current_time,
                'tokens': config.requests_per_minute
            })
            last_refill = bucket_data['last_refill']
            tokens = bucket_data['tokens']
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = current_time - last_refill
        tokens_to_add = time_elapsed * (config.requests_per_minute / 60.0)  # tokens per second
        tokens = min(config.requests_per_minute + config.burst_allowance, tokens + tokens_to_add)
        
        # Check if request is allowed
        if tokens >= 1:
            tokens -= 1
            allowed = True
        else:
            allowed = False
        
        # Update bucket
        bucket_data = {
            'last_refill': current_time,
            'tokens': tokens
        }
        
        if self.redis_client:
            pipe = self.redis_client.pipeline()
            pipe.hset(bucket_key, mapping=bucket_data)
            pipe.expire(bucket_key, 3600)  # 1 hour TTL
            pipe.execute()
        else:
            self._fallback_storage[bucket_key] = bucket_data
        
        # Return rate limit headers info
        headers = {
            'X-RateLimit-Remaining': int(tokens),
            'X-RateLimit-Limit': config.requests_per_minute,
            'X-RateLimit-Reset': int(current_time + (60 - (current_time % 60)))
        }
        
        return allowed, headers
    
    def _sliding_window_check(self, client_id: str, config: RateLimitConfig) -> Tuple[bool, Dict[str, int]]:
        """Implement sliding window rate limiting algorithm."""
        current_time = int(time.time())
        
        headers = {}
        allowed = True
        
        # Check multiple time windows
        for window_seconds, limit in config.get_window_limits().items():
            window_key = f"rate_limit:window:{window_seconds}:{client_id}"
            
            if self.redis_client:
                # Remove old entries
                cutoff_time = current_time - window_seconds
                self.redis_client.zremrangebyscore(window_key, 0, cutoff_time)
                
                # Count current requests in window
                current_count = self.redis_client.zcard(window_key)
                
                if current_count >= limit:
                    allowed = False
                    headers[f'X-RateLimit-Limit-{window_seconds}s'] = limit
                    headers[f'X-RateLimit-Remaining-{window_seconds}s'] = 0
                    headers[f'X-RateLimit-Reset-{window_seconds}s'] = cutoff_time + window_seconds
                else:
                    # Add current request
                    self.redis_client.zadd(window_key, {str(current_time): current_time})
                    self.redis_client.expire(window_key, window_seconds + 10)
                    
                    headers[f'X-RateLimit-Limit-{window_seconds}s'] = limit
                    headers[f'X-RateLimit-Remaining-{window_seconds}s'] = limit - current_count - 1
            else:
                # Fallback implementation (less accurate)
                window_data = self._fallback_storage.get(window_key, [])
                # Remove old entries
                cutoff_time = current_time - window_seconds
                window_data = [t for t in window_data if t > cutoff_time]
                
                if len(window_data) >= limit:
                    allowed = False
                else:
                    window_data.append(current_time)
                
                self._fallback_storage[window_key] = window_data
                headers[f'X-RateLimit-Remaining-{window_seconds}s'] = max(0, limit - len(window_data))
        
        return allowed, headers
    
    def check_rate_limit(self, request: Request) -> Tuple[bool, Dict[str, int]]:
        """Check if request should be rate limited.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Tuple of (allowed: bool, headers: Dict[str, int])
        """
        client_id = self._get_client_identifier(request)
        sensitivity = self._get_endpoint_sensitivity(request.url.path, request.method)
        config = self.DEFAULT_CONFIGS[sensitivity]
        
        # Choose rate limiting strategy
        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._token_bucket_check(client_id, config)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._sliding_window_check(client_id, config)
        else:
            # Fixed window fallback
            return self._token_bucket_check(client_id, config)
    
    def get_client_stats(self, client_id: str) -> Dict[str, any]:
        """Get rate limiting statistics for a client."""
        if not self.redis_client:
            return {"error": "Redis not available"}
        
        stats = {}
        current_time = int(time.time())
        
        # Get stats for different windows
        for window_seconds in [60, 3600, 86400]:
            window_key = f"rate_limit:window:{window_seconds}:{client_id}"
            cutoff_time = current_time - window_seconds
            
            # Clean old entries
            self.redis_client.zremrangebyscore(window_key, 0, cutoff_time)
            
            # Get current count
            current_count = self.redis_client.zcard(window_key)
            stats[f'requests_last_{window_seconds}s'] = current_count
        
        return stats


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for applying rate limiting to requests."""
    
    def __init__(self, app, redis_client=None, enable_rate_limiting: bool = True):
        """Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            redis_client: Redis client for distributed rate limiting
            enable_rate_limiting: Whether to enable rate limiting
        """
        super().__init__(app)
        self.rate_limit_manager = RateLimitManager(redis_client)
        self.enable_rate_limiting = enable_rate_limiting
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply rate limiting to requests."""
        
        # Skip rate limiting if disabled or for certain paths
        if not self.enable_rate_limiting or self._should_skip_rate_limiting(request):
            return await call_next(request)
        
        # Check rate limits
        allowed, rate_limit_headers = self.rate_limit_manager.check_rate_limit(request)
        
        if not allowed:
            # Rate limit exceeded
            self._log_rate_limit_exceeded(request)
            
            return Response(
                content=json.dumps({
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please slow down and try again later.",
                    "type": "rate_limit_error"
                }),
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Content-Type": "application/json",
                    **{str(k): str(v) for k, v in rate_limit_headers.items()},
                    "Retry-After": "60"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for header, value in rate_limit_headers.items():
            response.headers[header] = str(value)
        
        return response
    
    def _should_skip_rate_limiting(self, request: Request) -> bool:
        """Check if rate limiting should be skipped for this request."""
        path = request.url.path.lower()
        
        # Skip for certain internal endpoints
        skip_paths = ['/health', '/ping', '/ready', '/metrics']
        return any(skip_path in path for skip_path in skip_paths)
    
    def _log_rate_limit_exceeded(self, request: Request):
        """Log rate limit exceeded event (without exposing sensitive data)."""
        client_id_hash = hashlib.sha256(
            self.rate_limit_manager._get_client_identifier(request).encode()
        ).hexdigest()[:16]
        
        # PHI-safe logging - no IP addresses or user data
        print(f"Rate limit exceeded: client_hash={client_id_hash}, "
              f"endpoint={request.url.path}, method={request.method}, "
              f"time={datetime.utcnow().isoformat()}")


def create_rate_limit_middleware(redis_url: Optional[str] = None, 
                               enable_rate_limiting: bool = True) -> RateLimitMiddleware:
    """Create rate limiting middleware with Redis connection.
    
    Args:
        redis_url: Redis connection URL
        enable_rate_limiting: Whether to enable rate limiting
        
    Returns:
        Configured RateLimitMiddleware
    """
    redis_client = None
    
    if REDIS_AVAILABLE and redis_url:
        try:
            redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            redis_client.ping()
            print(f"Rate limiting connected to Redis: {redis_url}")
        except Exception as e:
            print(f"Warning: Could not connect to Redis for rate limiting: {e}")
            print("Rate limiting will use in-memory fallback (not suitable for production)")
    
    return lambda app: RateLimitMiddleware(
        app, 
        redis_client=redis_client, 
        enable_rate_limiting=enable_rate_limiting
    )


# Dependency for getting rate limit statistics
async def get_rate_limit_stats(request: Request) -> Dict[str, any]:
    """Get rate limiting statistics for current client."""
    manager = RateLimitManager()
    client_id = manager._get_client_identifier(request)
    return manager.get_client_stats(client_id)