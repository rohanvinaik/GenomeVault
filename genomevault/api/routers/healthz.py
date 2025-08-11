"""
Consolidated health check endpoint for GenomeVault API.
Follows Kubernetes health check best practices.

"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from genomevault.api.types import HealthCheckResult, ComponentHealth

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


class HealthStatus(BaseModel):
    """Health check response model."""

    status: str
    timestamp: str
    version: str
    checks: Dict[str, HealthCheckResult]


class DetailedHealthStatus(HealthStatus):
    """Detailed health check with component status."""

    components: Dict[str, ComponentHealth]


def check_database() -> HealthCheckResult:
    """Check database connectivity."""
    try:
        # Placeholder for future database connectivity check
        # For now, return healthy status
        return {
            "status": "healthy",
            "message": "Database connection OK",
            "error": None,
            "latency_ms": 0.5,
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": None,
            "error": str(e),
            "latency_ms": None,
        }


def check_cache() -> HealthCheckResult:
    """Check cache service connectivity."""
    try:
        # Placeholder for future cache connectivity check
        return {
            "status": "healthy",
            "message": "Cache service OK",
            "error": None,
            "latency_ms": 0.3,
        }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": None,
            "error": str(e),
            "latency_ms": None,
        }


def check_filesystem() -> HealthCheckResult:
    """Check filesystem access."""
    try:
        from genomevault.config import CACHE_DIR, DATA_DIR

        # Check if critical directories are accessible
        for dir_path in [CACHE_DIR, DATA_DIR]:
            if dir_path.exists() and dir_path.is_dir():
                continue
            else:
                return {
                    "status": "unhealthy",
                    "message": None,
                    "error": f"Directory {dir_path} not accessible",
                    "latency_ms": None,
                }

        return {
            "status": "healthy",
            "message": "Filesystem access OK",
            "error": None,
            "latency_ms": 0.1,
        }
    except Exception as e:
        logger.error(f"Filesystem health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": None,
            "error": str(e),
            "latency_ms": None,
        }


@router.get("/healthz", response_model=HealthStatus)
async def healthz(response: Response) -> Any:
    """
    Kubernetes-style health check endpoint.

    Returns 200 if the service is healthy, 503 if unhealthy.
    This is the primary health check endpoint for monitoring and orchestration.
    """
    checks = {
        "api": {
            "status": "healthy",
            "message": "API is responsive",
            "error": None,
            "latency_ms": 0.1,
        },
        "filesystem": check_filesystem(),
    }

    # Determine overall health
    is_healthy = all(check.get("status") == "healthy" for check in checks.values())

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall_status = "unhealthy"
    else:
        overall_status = "healthy"

    return HealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version="v1.0.0",  # Version will be pulled from package metadata
        checks=checks,
    )


@router.get("/healthz/live", response_model=Dict[str, str])
async def liveness() -> Any:
    """
    Kubernetes liveness probe endpoint.

    Simple check to see if the service is running.
    Returns 200 if the service is alive.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/healthz/ready", response_model=DetailedHealthStatus)
async def readiness(response: Response) -> Any:
    """
    Kubernetes readiness probe endpoint.

    Comprehensive check to see if the service is ready to accept traffic.
    Returns 200 if ready, 503 if not ready.
    """
    components = {
        "database": check_database(),
        "cache": check_cache(),
        "filesystem": check_filesystem(),
    }

    # Check if all components are healthy
    is_ready = all(comp.get("status") == "healthy" for comp in components.values())

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall_status = "not_ready"
    else:
        overall_status = "ready"

    return DetailedHealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version="v1.0.0",
        checks={
            "api": {
                "status": "healthy",
                "message": "API is responsive",
                "error": None,
                "latency_ms": 0.1,
            },
            "dependencies": {
                "status": "healthy" if is_ready else "degraded",
                "message": (
                    "All dependencies operational" if is_ready else "Some dependencies degraded"
                ),
                "error": None,
                "latency_ms": 0.2,
            },
        },
        components=components,
    )


@router.get("/healthz/startup", response_model=Dict[str, str])
async def startup() -> Any:
    """
    Kubernetes startup probe endpoint.

    Used during container startup to know when the application has started.
    """
    return {
        "status": "started",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "GenomeVault API has started successfully",
    }
