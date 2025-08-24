"""Prometheus metrics endpoint for monitoring."""

from typing import Any, Dict

from fastapi import APIRouter, Response, HTTPException

from genomevault.observability.metrics.prometheus import (
    get_prometheus_metrics,
    get_metrics_content_type,
    get_metrics_collector,
)
from genomevault.zk_proofs.prover import Prover
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


@router.get("/metrics/zk/dashboard")
async def get_zk_dashboard() -> Dict[str, Any]:
    """
    Get ZK proof performance dashboard data.
    
    Returns comprehensive performance metrics for ZK operations including:
    - Proof generation statistics
    - Memory usage patterns  
    - Device utilization
    - Cache hit rates
    - Error rates
    """
    try:
        prover = Prover()
        dashboard_data = prover.get_performance_dashboard()
        system_info = prover.get_system_info()
        
        return {
            "status": "success",
            "system_info": system_info,
            "performance_metrics": dashboard_data,
            "timestamp": dashboard_data.get("timestamp")
        }
    except Exception as e:
        logger.error(f"Failed to get ZK dashboard data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get ZK dashboard data: {str(e)}"
        )


@router.get("/metrics/zk/report")
async def get_zk_performance_report() -> Dict[str, Any]:
    """
    Get comprehensive ZK performance report.
    
    Returns detailed text report with:
    - Performance summary statistics
    - Operation breakdown by circuit type
    - Memory and timing analysis
    - Optimization recommendations
    """
    try:
        prover = Prover()
        report = prover.get_performance_report()
        
        return {
            "status": "success",
            "report": report,
            "format": "text"
        }
    except Exception as e:
        logger.error(f"Failed to get ZK performance report: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get ZK performance report: {str(e)}"
        )


@router.get("/metrics/zk/system")
async def get_zk_system_info() -> Dict[str, Any]:
    """
    Get current ZK system status and information.
    
    Returns real-time system metrics including:
    - Current device (CPU/GPU)
    - Memory usage
    - Backend availability
    - Production readiness status
    - Process information
    """
    try:
        prover = Prover()
        system_info = prover.get_system_info()
        
        return {
            "status": "success",
            "system": system_info
        }
    except Exception as e:
        logger.error(f"Failed to get ZK system info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get ZK system info: {str(e)}"
        )
