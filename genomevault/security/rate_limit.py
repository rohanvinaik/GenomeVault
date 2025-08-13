"""Rate Limit module."""

from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse
import os
import time


class TokenBucket:
    """TokenBucket implementation."""

    def __init__(self, rate: float, burst: int):
        """Initialize instance.

        Args:
            rate: Rate.
            burst: Burst.
        """
        self.rate = float(rate)
        self.capacity = int(burst)
        self.tokens = float(burst)
        self.updated = time.monotonic()

    def allow(self) -> bool:
        """Allow.

        Returns:
            Boolean result.
        """
        now = time.monotonic()
        delta = now - self.updated
        self.updated = now
        self.tokens = min(self.capacity, self.tokens + delta * self.rate)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """RateLimitMiddleware implementation."""

    def __init__(self, app, *, rate: float | None = None, burst: int | None = None):
        """Initialize instance.

        Args:
            app: App.
        """
        super().__init__(app)
        self.rate = float(rate or float(os.getenv("RATE_LIMIT_RPS", "5.0")))
        self.burst = int(burst or int(os.getenv("RATE_LIMIT_BURST", "10")))
        self.buckets: dict[str, TokenBucket] = {}

    def _key(self, request: Request) -> str:
        # Per-client-IP key
        client = request.client.host if request.client else "unknown"
        return client

    async def dispatch(self, request: Request, call_next):
        """Async operation to Dispatch.

        Args:
            request: Client request.
            call_next: Call next.

        Returns:
            Operation result.
        """
        key = self._key(request)
        tb = self.buckets.get(key)
        if tb is None:
            tb = self.buckets[key] = TokenBucket(self.rate, self.burst)
        if not tb.allow():
            return PlainTextResponse("Too Many Requests", status_code=429)
        return await call_next(request)
