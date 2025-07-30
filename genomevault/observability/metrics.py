from __future__ import annotations

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    _PROM = True
except Exception:  # pragma: no cover
    from genomevault.observability.logging import configure_logging

    logger = configure_logging()
    logger.exception("Unhandled exception")
    _PROM = False
    raise

from fastapi import APIRouter, Response

metrics_router = APIRouter()

if _PROM:
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
else:
    # Dummies to avoid crashes when prometheus_client is not installed
    class _Dummy:
        def labels(self, **_):
            return self

        def inc(self, *_a, **_k):
            pass

        def observe(self, *_a, **_k):
            pass

    http_requests_total = _Dummy()
    http_request_duration = _Dummy()


@metrics_router.get("/metrics")
def metrics():
    if not _PROM:
        return Response(
            content="# prometheus_client not installed\n", media_type="text/plain; version=0.0.4"
        )
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
