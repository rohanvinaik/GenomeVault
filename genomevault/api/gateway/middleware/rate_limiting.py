"""
Rate limiting middleware for GenomeVault API Gateway.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from genomevault.observability.logging import get_logger

logger = get_logger(__name__)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.
    
    Provides different rate limits based on:
    - User authentication tier (standard, clinical, research)
    - API endpoint category
    - User-specific overrides
    """
    
    def __init__(self, app):
        """Initialize rate limiting middleware."""
        super().__init__(app)
        
        # Rate limit configurations by tier
        self.rate_limits = {
            "standard": {"requests": 1000, "window": 3600},  # 1000/hour
            "clinical": {"requests": 10000, "window": 3600},  # 10000/hour
            "research": {"requests": 50000, "window": 3600},  # 50000/hour
            "admin": {"requests": 100000, "window": 3600},    # 100000/hour
            "anonymous": {"requests": 100, "window": 3600},   # 100/hour for unauthenticated
        }
        
        # Endpoint-specific multipliers
        self.endpoint_multipliers = {
            "/vectors/encode": 2.0,     # More expensive operations
            "/proofs/generate": 5.0,    # Very expensive operations
            "/queries/pir": 3.0,        # PIR queries are resource intensive
            "/algorithms/execute": 4.0, # Algorithm execution
            "/health": 0.1,             # Health checks are cheap
        }
        
        # In-memory storage for rate limiting (use Redis in production)
        self.buckets: Dict[str, Dict[str, float]] = defaultdict(lambda: {"tokens": 0, "last_refill": time.time()})
        
        # Paths exempt from rate limiting
        self.exempt_paths = {"/health", "/health/liveness", "/health/readiness"}
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Apply rate limiting to incoming requests.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            HTTP response with rate limit headers
        """
        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)
        
        # Get client identifier and tier
        client_id, tier = self._get_client_context(request)
        
        # Get rate limit configuration
        rate_limit_config = self.rate_limits.get(tier, self.rate_limits["anonymous"])
        endpoint_multiplier = self._get_endpoint_multiplier(request.url.path)
        
        # Calculate effective rate limit
        effective_limit = max(1, int(rate_limit_config["requests"] / endpoint_multiplier))
        window_seconds = rate_limit_config["window"]
        
        # Check rate limit
        allowed, remaining, reset_time = await self._check_rate_limit(
            client_id, effective_limit, window_seconds
        )
        
        # Create response if rate limited
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for client {client_id}",
                extra={
                    "client_id": client_id,
                    "tier": tier,
                    "path": request.url.path,
                    "effective_limit": effective_limit,
                    "reset_time": reset_time
                }
            )
            
            response = Response(
                content='{"type":"RateLimitError","code":"GV_RATE_LIMITED","message":"Rate limit exceeded"}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Content-Type": "application/json",
                    "X-RateLimit-Limit": str(effective_limit),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(int(reset_time)),
                    "Retry-After": str(int(reset_time - time.time())),
                }
            )
            return response
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add rate limiting headers
        response.headers["X-RateLimit-Limit"] = str(effective_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_time))
        response.headers["X-RateLimit-Tier"] = tier
        
        # Log request for monitoring
        logger.info(
            f"Request processed: {request.method} {request.url.path}",
            extra={
                "client_id": client_id,
                "tier": tier,
                "remaining": remaining,
                "process_time": process_time,
                "status_code": response.status_code
            }
        )
        
        return response
    
    def _get_client_context(self, request: Request) -> Tuple[str, str]:
        """
        Get client identifier and rate limiting tier.
        
        Args:
            request: HTTP request
            
        Returns:
            Tuple of (client_id, tier)
        """
        # Use authenticated user context if available
        if hasattr(request.state, "user_id") and request.state.user_id:
            client_id = f"user:{request.state.user_id}"
            tier = getattr(request.state, "rate_limit_tier", "standard")
            return client_id, tier
        
        # Fall back to IP-based identification for unauthenticated requests
        client_ip = self._get_client_ip(request)
        client_id = f"ip:{client_ip}"
        tier = "anonymous"
        
        return client_id, tier
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address, considering proxy headers.
        
        Args:
            request: HTTP request
            
        Returns:
            Client IP address
        """
        # Check for forwarded IP headers (for load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_endpoint_multiplier(self, path: str) -> float:
        """
        Get rate limit multiplier for specific endpoint.
        
        Args:
            path: Request path
            
        Returns:
            Multiplier value (higher = more restrictive)
        """
        # Check for exact matches first
        if path in self.endpoint_multipliers:
            return self.endpoint_multipliers[path]
        
        # Check for prefix matches
        for endpoint_prefix, multiplier in self.endpoint_multipliers.items():
            if path.startswith(endpoint_prefix):
                return multiplier
        
        # Default multiplier
        return 1.0
    
    async def _check_rate_limit(self, client_id: str, limit: int, window_seconds: int) -> Tuple[bool, int, float]:
        """
        Check rate limit using token bucket algorithm.
        
        Args:
            client_id: Client identifier
            limit: Request limit for the window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (allowed, remaining_tokens, reset_time)
        """
        current_time = time.time()
        bucket = self.buckets[client_id]
        
        # Calculate time since last refill
        time_since_refill = current_time - bucket["last_refill"]
        
        # Refill tokens based on time passed
        if time_since_refill > 0:
            # Calculate refill rate (tokens per second)
            refill_rate = limit / window_seconds
            tokens_to_add = time_since_refill * refill_rate
            
            # Add tokens, but don't exceed limit
            bucket["tokens"] = min(limit, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = current_time
        
        # Check if we have tokens available
        if bucket["tokens"] >= 1:
            # Consume one token
            bucket["tokens"] -= 1
            allowed = True
            remaining = int(bucket["tokens"])
        else:
            # No tokens available
            allowed = False
            remaining = 0
        
        # Calculate reset time (when bucket will be full again)
        tokens_needed = limit - bucket["tokens"]
        refill_rate = limit / window_seconds
        time_to_full = tokens_needed / refill_rate if refill_rate > 0 else window_seconds
        reset_time = current_time + time_to_full
        
        return allowed, remaining, reset_time
    
    async def _get_user_rate_limit_tier(self, user_id: str) -> str:
        """
        Get rate limiting tier for specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Rate limiting tier
        """
        # TODO: Implement actual user tier lookup
        # This would query your user/subscription database
        
        # For demo purposes, determine tier based on user_id pattern
        if user_id.startswith("admin_"):
            return "admin"
        elif user_id.startswith("clinical_"):
            return "clinical"
        elif user_id.startswith("research_"):
            return "research"
        else:
            return "standard"
    
    def get_rate_limit_status(self, client_id: str, tier: str) -> dict:
        """
        Get current rate limit status for a client.
        
        Args:
            client_id: Client identifier
            tier: Rate limiting tier
            
        Returns:
            Rate limit status dictionary
        """
        bucket = self.buckets.get(client_id)
        rate_limit_config = self.rate_limits.get(tier, self.rate_limits["anonymous"])
        
        if not bucket:
            return {
                "limit": rate_limit_config["requests"],
                "remaining": rate_limit_config["requests"],
                "reset_time": time.time() + rate_limit_config["window"],
                "tier": tier
            }
        
        current_time = time.time()
        time_since_refill = current_time - bucket["last_refill"]
        
        # Calculate current token count
        if time_since_refill > 0:
            refill_rate = rate_limit_config["requests"] / rate_limit_config["window"]
            tokens_to_add = time_since_refill * refill_rate
            current_tokens = min(rate_limit_config["requests"], bucket["tokens"] + tokens_to_add)
        else:
            current_tokens = bucket["tokens"]
        
        # Calculate reset time
        tokens_needed = rate_limit_config["requests"] - current_tokens
        refill_rate = rate_limit_config["requests"] / rate_limit_config["window"]
        time_to_full = tokens_needed / refill_rate if refill_rate > 0 else rate_limit_config["window"]
        reset_time = current_time + time_to_full
        
        return {
            "limit": rate_limit_config["requests"],
            "remaining": int(current_tokens),
            "reset_time": reset_time,
            "tier": tier
        }