"""Middleware module for GenomeVault observability."""

# Import enhanced middleware
from .enhanced import (
    EnhancedObservabilityMiddleware,
    PerformanceTimingMiddleware,
    add_enhanced_observability_middleware,
    add_performance_timing_middleware
)

# Import basic middleware from parent for compatibility  
import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Basic ObservabilityMiddleware implementation for compatibility."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Basic request dispatch with minimal observability."""
        t0 = time.perf_counter()
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = req_id
        
        try:
            resp = await call_next(request)
        except Exception as e:
            print(f"Request failed: {e}")
            raise
            
        dt = (time.perf_counter() - t0) * 1000.0
        resp.headers["X-Request-ID"] = req_id
        return resp

def add_observability_middleware(app):
    """Add basic observability middleware."""
    app.add_middleware(ObservabilityMiddleware)

__all__ = [
    "EnhancedObservabilityMiddleware",
    "PerformanceTimingMiddleware", 
    "add_enhanced_observability_middleware",
    "add_performance_timing_middleware",
    "ObservabilityMiddleware",
    "add_observability_middleware"
]