from __future__ import annotations

import importlib.util

from fastapi import APIRouter, Response

# Check if prometheus_client is available
_PROM_AVAILABLE = importlib.util.find_spec("prometheus_client") is not None

if _PROM_AVAILABLE:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )
else:
    # Dummy implementations when prometheus_client is not available
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

    class _Dummy:
        """Dummy metric class for when prometheus_client is unavailable"""

        def __init__(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            """Return self to allow method chaining"""
            return self

        def inc(self, *args, **kwargs):
            """No-op increment method"""
            pass

        def observe(self, *args, **kwargs):
            """No-op observe method"""
            pass

    Counter = _Dummy
    Histogram = _Dummy

    def generate_latest():
        """Return dummy metrics when prometheus_client unavailable"""
        return b"# prometheus_client not installed\n"


metrics_router = APIRouter()

# Initialize metrics (works with both real and dummy classes)
http_requests_total = Counter(
    "genomevault_http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "path", "status"),
)
http_request_duration = Histogram(
    "genomevault_http_request_duration_seconds",
    "Request duration in seconds",
    labelnames=("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


@metrics_router.get("/metrics")
def metrics():
    """Return Prometheus metrics or a plaintext message if unavailable"""
    if not _PROM_AVAILABLE:
        return Response(
            content="# prometheus_client not installed\n",
            media_type="text/plain",
        )
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
