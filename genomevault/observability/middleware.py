"""Middleware module."""

from __future__ import annotations

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
import time
import uuid

from genomevault.observability.logging import get_logger

try:
    from genomevault.observability.metrics import REGISTRY

    # Get or create metrics
    http_requests_total = REGISTRY.counter("http_requests_total")
    http_request_duration = REGISTRY.histogram("http_request_duration_seconds")

    _METRICS = True
except Exception as e:
    print(f"Warning: Could not import metrics: {e}")
    _METRICS = False

_LOG = get_logger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """ObservabilityMiddleware implementation."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Async operation to Dispatch.

        Args:
            request: Client request.
            call_next: Call next.

        Returns:
            Response instance.

        Raises:
            RuntimeError: When operation fails.
        """
        t0 = time.perf_counter()
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        # store in state for handlers to access
        request.state.request_id = req_id
        try:
            resp = await call_next(request)
        except Exception as e:
            # still log, then raise
            dt = (time.perf_counter() - t0) * 1000.0
            _LOG.error(
                "request failed",
                extra={
                    "request_id": req_id,
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": 500,
                    "duration_ms": round(dt, 2),
                    "client": request.client.host if request.client else None,
                    "error": str(e),
                },
            )
            raise

        dt = (time.perf_counter() - t0) * 1000.0
        status = resp.status_code
        # Prometheus metrics (labels with path template if available)
        if _METRICS:
            route = request.scope.get("route")
            path_t = getattr(route, "path_format", None) or request.url.path
            http_requests_total.inc()  # Simplified for basic counter
            http_request_duration.observe(dt / 1000.0)  # Simplified for basic histogram

        _LOG.info(
            "request complete",
            extra={
                "request_id": req_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": status,
                "duration_ms": round(dt, 2),
                "client": request.client.host if request.client else None,
            },
        )
        # Reflect request id to client
        resp.headers["X-Request-ID"] = req_id
        return resp


def add_observability_middleware(app):
    """Add observability middleware.

    Args:
        app: App.
    """
    app.add_middleware(ObservabilityMiddleware)
