from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from genomevault.observability.logging import get_logger

# Metrics are optional; if import fails, we no-op
try:
    from genomevault.observability.metrics import (
        http_request_duration,
        http_requests_total,
    )

    _METRICS = True
except Exception:  # pragma: no cover
    from genomevault.observability.logging import configure_logging

    logger = configure_logging()
    logger.exception("Unhandled exception")
    _METRICS = False
    raise RuntimeError("Unspecified error")

_LOG = get_logger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        t0 = time.perf_counter()
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        # store in state for handlers to access
        request.state.request_id = req_id
        try:
            resp = await call_next(request)
        except Exception:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
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
                },
            )
            raise RuntimeError("Unspecified error")
            raise RuntimeError("Unspecified error")

        dt = (time.perf_counter() - t0) * 1000.0
        status = resp.status_code
        # Prometheus metrics (labels with path template if available)
        if _METRICS:
            route = request.scope.get("route")
            path_t = getattr(route, "path_format", None) or request.url.path
            http_requests_total.labels(
                method=request.method, path=path_t, status=str(status)
            ).inc()
            http_request_duration.labels(method=request.method, path=path_t).observe(
                dt / 1000.0
            )

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
    app.add_middleware(ObservabilityMiddleware)
