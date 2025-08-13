"""
Prometheus metrics endpoint for monitoring.

"""
from typing import Any

from fastapi import APIRouter, Response
from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Gauge,
    Info,
)

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["metrics"])

# Application-level metrics
app_info = Info("genomevault_app", "Application information")
app_info.info({"version": "1.0.0", "service": "genomevault-api"})

# Request metrics (shared across routers)
http_requests_total = Counter(
    "genomevault_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "genomevault_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

# Resource metrics
memory_usage_bytes = Gauge("genomevault_memory_usage_bytes", "Memory usage in bytes")

cpu_usage_percent = Gauge("genomevault_cpu_usage_percent", "CPU usage percentage")

# Genomic processing metrics
genomic_variants_processed_total = Counter(
    "genomevault_genomic_variants_processed_total", "Total genomic variants processed"
)

hypervector_dimensions = Histogram(
    "genomevault_hypervector_dimensions",
    "Distribution of hypervector dimensions used",
    buckets=[1000, 5000, 10000, 15000, 20000, 50000, 100000],
)

encoding_compression_ratio = Histogram(
    "genomevault_encoding_compression_ratio",
    "Compression ratio achieved during encoding",
    buckets=[1, 10, 50, 100, 500, 1000],
)

# Privacy metrics
privacy_requests_total = Counter(
    "genomevault_privacy_requests_total",
    "Total privacy-preserving operations",
    ["operation_type"],
)

zk_proofs_generated_total = Counter(
    "genomevault_zk_proofs_generated_total",
    "Total zero-knowledge proofs generated",
    ["circuit_type"],
)

pir_queries_total = Counter(
    "genomevault_pir_queries_total", "Total Private Information Retrieval queries"
)

# Error metrics
errors_total = Counter("genomevault_errors_total", "Total errors", ["error_type", "component"])

# Performance metrics
cache_hits_total = Counter("genomevault_cache_hits_total", "Total cache hits", ["cache_type"])

cache_misses_total = Counter("genomevault_cache_misses_total", "Total cache misses", ["cache_type"])

db_connections_active = Gauge("genomevault_db_connections_active", "Active database connections")

db_query_duration_seconds = Histogram(
    "genomevault_db_query_duration_seconds",
    "Database query duration in seconds",
    ["query_type"],
)


@router.get("/metrics", include_in_schema=False)
async def get_metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns current metrics in Prometheus text format.
    This endpoint should be scraped by Prometheus server.
    """
    try:
        # Update resource metrics
        try:
            import psutil

            process = psutil.Process()
            memory_usage_bytes.set(process.memory_info().rss)
            cpu_usage_percent.set(process.cpu_percent())
        except ImportError:
            logger.debug("psutil not available, skipping resource metrics")
        except Exception as e:
            logger.warning(f"Failed to collect resource metrics: {e}")

        # Generate metrics in Prometheus format
        metrics_output = generate_latest()

        return Response(content=metrics_output, media_type=CONTENT_TYPE_LATEST)

    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}", exc_info=True)
        errors_total.labels(error_type="metrics_generation", component="metrics").inc()

        # Return empty metrics on error
        return Response(
            content="# Error generating metrics\n",
            media_type="text/plain",
            status_code=500,
        )


@router.get("/metrics/health")
async def metrics_health() -> Any:
    """
    Health check for metrics endpoint.

    Verifies that metrics collection is working properly.
    """
    try:
        # Try to generate metrics
        _ = generate_latest()

        return {"status": "healthy", "message": "Metrics collection is working"}
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Metrics collection failed: {str(e)}",
        }
