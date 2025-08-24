"""Prometheus metrics endpoint for monitoring."""

from typing import Any

from fastapi import APIRouter, Response

from genomevault.observability.metrics.prometheus import (
    get_prometheus_metrics,
    get_metrics_content_type,
    get_metrics_collector,
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["metrics"])


@router.get("/metrics", include_in_schema=False)
async def get_metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns current metrics in Prometheus text format.
    This endpoint should be scraped by Prometheus server.
    """
    try:
        # Update system metrics
        collector = get_metrics_collector()
        
        try:
            import psutil
            process = psutil.Process()
            
            # Update database connections (example)
            collector.update_database_connections("sqlite", 1)
            
            # Update active users (example)
            collector.update_active_users(1)
            
        except ImportError:
            logger.debug("psutil not available, skipping system metrics")
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

        # Generate comprehensive metrics using our enhanced collector
        metrics_output = get_prometheus_metrics()
        content_type = get_metrics_content_type()

        return Response(content=metrics_output, media_type=content_type)

    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}", exc_info=True)
        collector = get_metrics_collector()
        collector.record_error("metrics_generation", "metrics", "error")

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
        # Try to generate metrics using our enhanced system
        _ = get_prometheus_metrics()

        return {"status": "healthy", "message": "Metrics collection is working"}
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Metrics collection failed: {str(e)}",
        }
