"""
Rate limiting middleware for GenomeVault API.

Implements token bucket algorithm with Redis backend for distributed rate limiting
across multiple API instances with support for different tiers and PHI access control.
"""

import os
import time
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from enum import Enum

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Redis client
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=False)  # Use bytes for Lua scripts


class RateLimitTier(str, Enum):
    """Rate limit tiers for different user types."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"


class RateLimitConfig(BaseModel):
    """Rate limit configuration."""
    tier: RateLimitTier
    # Requests per time period
    requests_per_second: Optional[int] = None
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    # Burst allowance
    burst_size: int = 10
    # PHI access limits (stricter for HIPAA compliance)
    phi_requests_per_minute: int = 10
    phi_requests_per_hour: int = 100
    phi_requests_per_day: int = 500
    # Compute-intensive operation limits
    compute_requests_per_minute: int = 5
    compute_requests_per_hour: int = 50
    # Concurrent request limits
    max_concurrent_requests: int = 10
    # Cost tracking
    cost_per_request: float = 0.001
    monthly_budget: Optional[float] = None


# Default configurations per tier
TIER_CONFIGS = {
    RateLimitTier.FREE: RateLimitConfig(
        tier=RateLimitTier.FREE,
        requests_per_minute=10,
        requests_per_hour=100,
        requests_per_day=500,
        burst_size=5,
        phi_requests_per_minute=0,  # No PHI access for free tier
        phi_requests_per_hour=0,
        phi_requests_per_day=0,
        compute_requests_per_minute=1,
        compute_requests_per_hour=5,
        max_concurrent_requests=2,
        cost_per_request=0,
        monthly_budget=0,
    ),
    RateLimitTier.BASIC: RateLimitConfig(
        tier=RateLimitTier.BASIC,
        requests_per_minute=60,
        requests_per_hour=1000,
        requests_per_day=10000,
        burst_size=10,
        phi_requests_per_minute=5,
        phi_requests_per_hour=50,
        phi_requests_per_day=250,
        compute_requests_per_minute=5,
        compute_requests_per_hour=50,
        max_concurrent_requests=5,
        cost_per_request=0.001,
        monthly_budget=100,
    ),
    RateLimitTier.PROFESSIONAL: RateLimitConfig(
        tier=RateLimitTier.PROFESSIONAL,
        requests_per_minute=300,
        requests_per_hour=5000,
        requests_per_day=50000,
        burst_size=50,
        phi_requests_per_minute=30,
        phi_requests_per_hour=500,
        phi_requests_per_day=5000,
        compute_requests_per_minute=20,
        compute_requests_per_hour=200,
        max_concurrent_requests=20,
        cost_per_request=0.0005,
        monthly_budget=500,
    ),
    RateLimitTier.ENTERPRISE: RateLimitConfig(
        tier=RateLimitTier.ENTERPRISE,
        requests_per_minute=1000,
        requests_per_hour=20000,
        requests_per_day=200000,
        burst_size=100,
        phi_requests_per_minute=100,
        phi_requests_per_hour=2000,
        phi_requests_per_day=20000,
        compute_requests_per_minute=50,
        compute_requests_per_hour=500,
        max_concurrent_requests=100,
        cost_per_request=0.0001,
        monthly_budget=None,  # Custom billing
    ),
    RateLimitTier.UNLIMITED: RateLimitConfig(
        tier=RateLimitTier.UNLIMITED,
        requests_per_second=None,
        requests_per_minute=999999,
        requests_per_hour=999999,
        requests_per_day=999999,
        burst_size=1000,
        phi_requests_per_minute=999999,
        phi_requests_per_hour=999999,
        phi_requests_per_day=999999,
        compute_requests_per_minute=999999,
        compute_requests_per_hour=999999,
        max_concurrent_requests=1000,
        cost_per_request=0,
        monthly_budget=None,
    ),
}


# Lua script for atomic rate limit check and increment
RATE_LIMIT_LUA_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local current_time = tonumber(ARGV[3])

-- Get current count
local current = redis.call('GET', key)
if current == false then
    current = 0
else
    current = tonumber(current)
end

-- Check if limit exceeded
if current >= limit then
    local ttl = redis.call('TTL', key)
    return {0, current, ttl}  -- Denied, current count, TTL
end

-- Increment and set expiry
local new_count = redis.call('INCR', key)
if new_count == 1 then
    redis.call('EXPIRE', key, window)
end

local ttl = redis.call('TTL', key)
return {1, new_count, ttl}  -- Allowed, new count, TTL
"""

# Lua script for token bucket algorithm
TOKEN_BUCKET_LUA_SCRIPT = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])  -- Tokens per second
local capacity = tonumber(ARGV[2])  -- Bucket capacity
local current_time = tonumber(ARGV[3])
local requested = tonumber(ARGV[4] or 1)  -- Tokens requested

local bucket = redis.call('HGETALL', key)
local tokens = capacity
local last_refill = current_time

if #bucket > 0 then
    for i = 1, #bucket, 2 do
        if bucket[i] == 'tokens' then
            tokens = tonumber(bucket[i + 1])
        elseif bucket[i] == 'last_refill' then
            last_refill = tonumber(bucket[i + 1])
        end
    end
    
    -- Refill tokens based on time elapsed
    local elapsed = current_time - last_refill
    local tokens_to_add = elapsed * rate
    tokens = math.min(capacity, tokens + tokens_to_add)
end

-- Check if enough tokens available
if tokens < requested then
    -- Not enough tokens
    local wait_time = (requested - tokens) / rate
    return {0, tokens, wait_time}  -- Denied, available tokens, wait time
end

-- Consume tokens
tokens = tokens - requested
redis.call('HSET', key, 'tokens', tokens, 'last_refill', current_time)
redis.call('EXPIRE', key, 3600)  -- Expire after 1 hour of inactivity

return {1, tokens, 0}  -- Allowed, remaining tokens, no wait
"""


class RateLimiter:
    """Rate limiter implementation with multiple algorithms."""
    
    def __init__(self):
        """Initialize rate limiter with Lua scripts."""
        # Register Lua scripts
        self.rate_limit_script = redis_client.register_script(RATE_LIMIT_LUA_SCRIPT)
        self.token_bucket_script = redis_client.register_script(TOKEN_BUCKET_LUA_SCRIPT)
    
    def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int,
        prefix: str = "rate_limit"
    ) -> Tuple[bool, int, int]:
        """
        Check rate limit using sliding window counter.
        
        Returns:
            Tuple of (allowed, current_count, ttl_seconds)
        """
        key = f"{prefix}:{identifier}:{window}"
        current_time = int(time.time())
        
        result = self.rate_limit_script(
            keys=[key.encode()],
            args=[limit, window, current_time]
        )
        
        allowed = bool(result[0])
        current_count = int(result[1])
        ttl = int(result[2]) if result[2] > 0 else window
        
        return allowed, current_count, ttl
    
    def check_token_bucket(
        self,
        identifier: str,
        rate: float,
        capacity: int,
        tokens_requested: int = 1,
        prefix: str = "token_bucket"
    ) -> Tuple[bool, float, float]:
        """
        Check rate limit using token bucket algorithm.
        
        Returns:
            Tuple of (allowed, remaining_tokens, wait_time_seconds)
        """
        key = f"{prefix}:{identifier}"
        current_time = time.time()
        
        result = self.token_bucket_script(
            keys=[key.encode()],
            args=[rate, capacity, current_time, tokens_requested]
        )
        
        allowed = bool(result[0])
        remaining_tokens = float(result[1])
        wait_time = float(result[2])
        
        return allowed, remaining_tokens, wait_time
    
    def check_concurrent_requests(
        self,
        identifier: str,
        max_concurrent: int
    ) -> Tuple[bool, int]:
        """
        Check concurrent request limit.
        
        Returns:
            Tuple of (allowed, current_concurrent)
        """
        key = f"concurrent:{identifier}"
        
        # Increment concurrent count
        current = redis_client.incr(key)
        
        # Set expiry if first request
        if current == 1:
            redis_client.expire(key, 300)  # 5 minute timeout
        
        if current > max_concurrent:
            # Decrement back if over limit
            redis_client.decr(key)
            return False, current - 1
        
        return True, current
    
    def release_concurrent_request(self, identifier: str):
        """Release a concurrent request slot."""
        key = f"concurrent:{identifier}"
        current = redis_client.decr(key)
        
        # Clean up if no more concurrent requests
        if current <= 0:
            redis_client.delete(key)
    
    def track_usage(
        self,
        identifier: str,
        cost: float,
        endpoint: str,
        response_time: float
    ):
        """Track API usage for billing and analytics."""
        # Daily usage tracking
        today = datetime.now().strftime("%Y%m%d")
        usage_key = f"usage:{identifier}:{today}"
        
        # Increment counters
        redis_client.hincrby(usage_key, "requests", 1)
        redis_client.hincrbyfloat(usage_key, "cost", cost)
        redis_client.hincrbyfloat(usage_key, "total_response_time", response_time)
        
        # Track per-endpoint usage
        endpoint_key = f"endpoint_usage:{identifier}:{today}"
        redis_client.hincrby(endpoint_key, endpoint, 1)
        
        # Set expiry for 30 days
        redis_client.expire(usage_key, 30 * 24 * 3600)
        redis_client.expire(endpoint_key, 30 * 24 * 3600)
    
    def get_usage_stats(self, identifier: str, days: int = 7) -> Dict[str, Any]:
        """Get usage statistics for an identifier."""
        stats = {
            "daily_usage": [],
            "total_requests": 0,
            "total_cost": 0,
            "average_response_time": 0,
            "endpoint_breakdown": {},
        }
        
        total_response_time = 0
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            usage_key = f"usage:{identifier}:{date}"
            endpoint_key = f"endpoint_usage:{identifier}:{date}"
            
            # Get daily stats
            daily_data = redis_client.hgetall(usage_key)
            if daily_data:
                requests = int(daily_data.get(b"requests", 0))
                cost = float(daily_data.get(b"cost", 0))
                response_time = float(daily_data.get(b"total_response_time", 0))
                
                stats["daily_usage"].append({
                    "date": date,
                    "requests": requests,
                    "cost": cost,
                    "average_response_time": response_time / requests if requests > 0 else 0,
                })
                
                stats["total_requests"] += requests
                stats["total_cost"] += cost
                total_response_time += response_time
            
            # Get endpoint breakdown
            endpoint_data = redis_client.hgetall(endpoint_key)
            for endpoint, count in endpoint_data.items():
                endpoint_str = endpoint.decode() if isinstance(endpoint, bytes) else endpoint
                stats["endpoint_breakdown"][endpoint_str] = stats["endpoint_breakdown"].get(endpoint_str, 0) + int(count)
        
        if stats["total_requests"] > 0:
            stats["average_response_time"] = total_response_time / stats["total_requests"]
        
        return stats


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for API requests.
    
    Features:
    - Multiple rate limit windows (second, minute, hour, day)
    - Different limits for PHI and compute-intensive endpoints
    - Token bucket for burst handling
    - Concurrent request limiting
    - Usage tracking for billing
    """
    
    def __init__(
        self,
        app: ASGIApp,
        default_tier: RateLimitTier = RateLimitTier.BASIC,
        exempt_paths: List[str] = None
    ):
        """Initialize rate limit middleware."""
        super().__init__(app)
        self.default_tier = default_tier
        self.exempt_paths = exempt_paths or [
            "/health",
            "/healthz",
            "/ready",
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json",
        ]
        self.rate_limiter = RateLimiter()
        
        # PHI endpoints that need stricter limits
        self.phi_endpoints = [
            "/api/v1/clinical",
            "/api/v1/phi",
            "/api/v1/patients",
        ]
        
        # Compute-intensive endpoints
        self.compute_endpoints = [
            "/api/v1/hypervector/encode",
            "/api/v1/zk/prove",
            "/api/v1/federated/train",
            "/api/v1/pir/query",
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiting."""
        # Check if path is exempt
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Get identifier (user ID, API key, or IP)
        identifier = self._get_identifier(request)
        
        # Get user's rate limit configuration
        config = self._get_rate_limit_config(request)
        
        # Determine endpoint type
        is_phi = any(request.url.path.startswith(path) for path in self.phi_endpoints)
        is_compute = any(request.url.path.startswith(path) for path in self.compute_endpoints)
        
        # Select appropriate limits
        if is_phi:
            limits = [
                ("minute", config.phi_requests_per_minute, 60),
                ("hour", config.phi_requests_per_hour, 3600),
                ("day", config.phi_requests_per_day, 86400),
            ]
            prefix = "phi_rate_limit"
        elif is_compute:
            limits = [
                ("minute", config.compute_requests_per_minute, 60),
                ("hour", config.compute_requests_per_hour, 3600),
            ]
            prefix = "compute_rate_limit"
        else:
            limits = [
                ("minute", config.requests_per_minute, 60),
                ("hour", config.requests_per_hour, 3600),
                ("day", config.requests_per_day, 86400),
            ]
            prefix = "rate_limit"
        
        # Check rate limits
        for window_name, limit, window_seconds in limits:
            allowed, current, ttl = self.rate_limiter.check_rate_limit(
                identifier=identifier,
                limit=limit,
                window=window_seconds,
                prefix=f"{prefix}:{window_name}"
            )
            
            if not allowed:
                logger.warning(
                    f"Rate limit exceeded for {identifier}",
                    extra={
                        "event": "rate_limit_exceeded",
                        "identifier": identifier,
                        "window": window_name,
                        "limit": limit,
                        "current": current,
                        "endpoint": request.url.path,
                        "is_phi": is_phi,
                    }
                )
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Rate limit exceeded. Limit: {limit} per {window_name}",
                        "limit": limit,
                        "window": window_name,
                        "retry_after": ttl,
                    },
                    headers={
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time()) + ttl),
                        "Retry-After": str(ttl),
                    }
                )
        
        # Check token bucket for burst handling
        if config.requests_per_second:
            allowed, remaining, wait_time = self.rate_limiter.check_token_bucket(
                identifier=identifier,
                rate=config.requests_per_second,
                capacity=config.burst_size,
                tokens_requested=1
            )
            
            if not allowed:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Rate limit exceeded. Please wait {wait_time:.1f} seconds",
                        "retry_after": wait_time,
                    },
                    headers={
                        "Retry-After": str(int(wait_time) + 1),
                    }
                )
        
        # Check concurrent requests
        allowed, current_concurrent = self.rate_limiter.check_concurrent_requests(
            identifier=identifier,
            max_concurrent=config.max_concurrent_requests
        )
        
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Too many concurrent requests. Limit: {config.max_concurrent_requests}",
                    "limit": config.max_concurrent_requests,
                    "current": current_concurrent,
                },
                headers={
                    "X-RateLimit-Concurrent-Limit": str(config.max_concurrent_requests),
                    "X-RateLimit-Concurrent-Current": str(current_concurrent),
                }
            )
        
        try:
            # Track request start time
            start_time = time.time()
            
            # Process request
            response = await call_next(request)
            
            # Track usage
            response_time = time.time() - start_time
            self.rate_limiter.track_usage(
                identifier=identifier,
                cost=config.cost_per_request,
                endpoint=request.url.path,
                response_time=response_time
            )
            
            # Add rate limit headers to response
            response.headers["X-RateLimit-Limit"] = str(config.requests_per_minute)
            response.headers["X-RateLimit-Tier"] = config.tier.value
            
            return response
            
        finally:
            # Release concurrent request slot
            self.rate_limiter.release_concurrent_request(identifier)
    
    def _get_identifier(self, request: Request) -> str:
        """Get identifier for rate limiting (user ID, API key, or IP)."""
        # Check for authenticated user
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.username}"
        
        # Check for API key
        if "x-api-key" in request.headers:
            api_key = request.headers["x-api-key"]
            # Hash API key for privacy
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            return f"api_key:{key_hash}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def _get_rate_limit_config(self, request: Request) -> RateLimitConfig:
        """Get rate limit configuration for request."""
        # Check for authenticated user with tier
        if hasattr(request.state, "user") and request.state.user:
            user = request.state.user
            # Get tier from user metadata (would be stored in database)
            tier_str = getattr(user, "rate_limit_tier", self.default_tier.value)
            try:
                tier = RateLimitTier(tier_str)
            except ValueError:
                tier = self.default_tier
            
            return TIER_CONFIGS.get(tier, TIER_CONFIGS[self.default_tier])
        
        # Check for API key tier
        if "x-api-key-tier" in request.headers:
            tier_str = request.headers["x-api-key-tier"]
            try:
                tier = RateLimitTier(tier_str)
                return TIER_CONFIGS.get(tier, TIER_CONFIGS[self.default_tier])
            except ValueError:
                pass
        
        # Default tier for unauthenticated requests
        return TIER_CONFIGS[RateLimitTier.FREE]


from typing import List  # Add this import at the top