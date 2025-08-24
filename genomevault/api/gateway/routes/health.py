"""
Health check routes for GenomeVault API Gateway.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status

from genomevault.api.gateway.models.health import (
    HealthCheckResponse,
    HealthStatus,
    DetailedHealthResponse,
    ReadinessCheckResponse,
    LivenessCheckResponse,
    ServiceDetails,
    ServiceStatus,
)
from genomevault.observability.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Application start time for uptime calculation
_start_time = time.time()


@router.get(
    "/",
    response_model=HealthCheckResponse,
    summary="System health check",
    description="Get overall system health status and service availability",
    responses={
        200: {"description": "System is healthy"},
        503: {"description": "System is unhealthy"},
    }
)
async def health_check() -> HealthCheckResponse:
    """
    Perform comprehensive system health check.
    
    Returns:
        System health status with service details
    """
    try:
        # Check individual services
        services = await _check_all_services()
        
        # Determine overall status
        overall_status = _calculate_overall_status(services)
        
        # Calculate uptime
        uptime_seconds = int(time.time() - _start_time)
        
        response = HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime_seconds=uptime_seconds,
            services=services,
            system_info={
                "environment": "production",  # Could be read from config
                "region": "us-west-2",       # Could be read from config
                "instance_id": "gateway-001" # Could be read from instance metadata
            }
        )
        
        # Log health check result
        if overall_status != HealthStatus.HEALTHY:
            logger.warning(f"Health check returned: {overall_status}")
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "type": "ServiceUnavailable",
                "code": "GV_HEALTH_CHECK_FAILED",
                "message": "Health check service unavailable"
            }
        )


@router.get(
    "/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed health check",
    description="Get detailed system health with performance metrics",
)
async def detailed_health_check() -> DetailedHealthResponse:
    """
    Perform detailed system health check with additional metrics.
    
    Returns:
        Detailed system health status with performance metrics
    """
    try:
        # Get basic health info
        basic_health = await health_check()
        
        # Add detailed metrics
        detailed_response = DetailedHealthResponse(
            **basic_health.dict(),
            memory_usage_mb=await _get_memory_usage(),
            cpu_usage_percent=await _get_cpu_usage(),
            disk_usage_percent=await _get_disk_usage(),
            active_connections=await _get_active_connections(),
            request_rate_per_minute=await _get_request_rate(),
            error_rate_percent=await _get_error_rate(),
        )
        
        return detailed_response
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "type": "ServiceUnavailable", 
                "code": "GV_DETAILED_HEALTH_CHECK_FAILED",
                "message": "Detailed health check service unavailable"
            }
        )


@router.get(
    "/liveness",
    response_model=LivenessCheckResponse,
    summary="Liveness probe",
    description="Kubernetes liveness probe endpoint",
)
async def liveness_check() -> LivenessCheckResponse:
    """
    Kubernetes liveness probe.
    
    Returns:
        Simple liveness status
    """
    return LivenessCheckResponse(
        alive=True,
        timestamp=datetime.utcnow()
    )


@router.get(
    "/readiness",
    response_model=ReadinessCheckResponse,
    summary="Readiness probe", 
    description="Kubernetes readiness probe endpoint",
)
async def readiness_check() -> ReadinessCheckResponse:
    """
    Kubernetes readiness probe.
    
    Returns:
        Service readiness status
    """
    try:
        # Check critical services for readiness
        checks = {
            "database_connection": await _check_database_readiness(),
            "external_services": await _check_external_services_readiness(),
            "cache_available": await _check_cache_readiness(),
        }
        
        # Service is ready if all critical checks pass
        ready = all(checks.values())
        
        return ReadinessCheckResponse(
            ready=ready,
            timestamp=datetime.utcnow(),
            checks=checks
        )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return ReadinessCheckResponse(
            ready=False,
            timestamp=datetime.utcnow(),
            checks={"error": False}
        )


async def _check_all_services() -> Dict[str, ServiceDetails]:
    """
    Check health of all system services.
    
    Returns:
        Dictionary of service health details
    """
    services = {}
    
    # Define service check functions
    service_checks = {
        "database": _check_database_health,
        "pir_engine": _check_pir_engine_health,
        "zk_prover": _check_zk_prover_health,
        "vector_store": _check_vector_store_health,
        "algorithm_marketplace": _check_algorithm_marketplace_health,
        "blockchain": _check_blockchain_health,
    }
    
    # Run all service checks concurrently
    check_tasks = [
        _check_service_with_timeout(name, check_func)
        for name, check_func in service_checks.items()
    ]
    
    results = await asyncio.gather(*check_tasks, return_exceptions=True)
    
    # Process results
    for i, (service_name, _) in enumerate(service_checks.items()):
        result = results[i]
        
        if isinstance(result, Exception):
            services[service_name] = ServiceDetails(
                status=ServiceStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                error_message=str(result)
            )
        else:
            services[service_name] = result
    
    return services


async def _check_service_with_timeout(name: str, check_func, timeout: float = 5.0) -> ServiceDetails:
    """
    Check service health with timeout.
    
    Args:
        name: Service name
        check_func: Health check function
        timeout: Timeout in seconds
        
    Returns:
        Service health details
    """
    try:
        return await asyncio.wait_for(check_func(), timeout=timeout)
    except asyncio.TimeoutError:
        return ServiceDetails(
            status=ServiceStatus.UNHEALTHY,
            last_check=datetime.utcnow(),
            error_message=f"Health check timed out after {timeout}s"
        )


async def _check_database_health() -> ServiceDetails:
    """Check database health."""
    start_time = time.perf_counter()
    
    try:
        # TODO: Implement actual database health check
        # This would typically check database connectivity and run a simple query
        await asyncio.sleep(0.01)  # Simulate database check
        
        response_time = (time.perf_counter() - start_time) * 1000
        
        return ServiceDetails(
            status=ServiceStatus.HEALTHY,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            metadata={"connection_pool": "healthy", "queries_per_second": "250"}
        )
        
    except Exception as e:
        return ServiceDetails(
            status=ServiceStatus.UNHEALTHY,
            last_check=datetime.utcnow(),
            error_message=str(e)
        )


async def _check_pir_engine_health() -> ServiceDetails:
    """Check PIR engine health."""
    start_time = time.perf_counter()
    
    try:
        # TODO: Implement actual PIR engine health check
        await asyncio.sleep(0.02)  # Simulate PIR engine check
        
        response_time = (time.perf_counter() - start_time) * 1000
        
        return ServiceDetails(
            status=ServiceStatus.HEALTHY,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            metadata={"active_queries": "5", "average_latency_ms": "120"}
        )
        
    except Exception as e:
        return ServiceDetails(
            status=ServiceStatus.UNHEALTHY,
            last_check=datetime.utcnow(),
            error_message=str(e)
        )


async def _check_zk_prover_health() -> ServiceDetails:
    """Check ZK prover health."""
    start_time = time.perf_counter()
    
    try:
        # TODO: Implement actual ZK prover health check
        await asyncio.sleep(0.03)  # Simulate ZK prover check
        
        response_time = (time.perf_counter() - start_time) * 1000
        
        return ServiceDetails(
            status=ServiceStatus.HEALTHY,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            metadata={"proving_queue_size": "2", "average_proof_time_ms": "3500"}
        )
        
    except Exception as e:
        return ServiceDetails(
            status=ServiceStatus.UNHEALTHY,
            last_check=datetime.utcnow(),
            error_message=str(e)
        )


async def _check_vector_store_health() -> ServiceDetails:
    """Check vector store health."""
    start_time = time.perf_counter()
    
    try:
        # TODO: Implement actual vector store health check
        await asyncio.sleep(0.015)  # Simulate vector store check
        
        response_time = (time.perf_counter() - start_time) * 1000
        
        return ServiceDetails(
            status=ServiceStatus.HEALTHY,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            metadata={"stored_vectors": "15420", "index_size_mb": "2048"}
        )
        
    except Exception as e:
        return ServiceDetails(
            status=ServiceStatus.UNHEALTHY,
            last_check=datetime.utcnow(),
            error_message=str(e)
        )


async def _check_algorithm_marketplace_health() -> ServiceDetails:
    """Check algorithm marketplace health."""
    start_time = time.perf_counter()
    
    try:
        # TODO: Implement actual marketplace health check
        await asyncio.sleep(0.01)  # Simulate marketplace check
        
        response_time = (time.perf_counter() - start_time) * 1000
        
        return ServiceDetails(
            status=ServiceStatus.HEALTHY,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            metadata={"available_algorithms": "127", "active_executions": "8"}
        )
        
    except Exception as e:
        return ServiceDetails(
            status=ServiceStatus.UNHEALTHY,
            last_check=datetime.utcnow(),
            error_message=str(e)
        )


async def _check_blockchain_health() -> ServiceDetails:
    """Check blockchain service health."""
    start_time = time.perf_counter()
    
    try:
        # TODO: Implement actual blockchain health check
        await asyncio.sleep(0.025)  # Simulate blockchain check
        
        response_time = (time.perf_counter() - start_time) * 1000
        
        return ServiceDetails(
            status=ServiceStatus.HEALTHY,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            metadata={"block_height": "12456", "peer_count": "15"}
        )
        
    except Exception as e:
        return ServiceDetails(
            status=ServiceStatus.UNHEALTHY,
            last_check=datetime.utcnow(),
            error_message=str(e)
        )


def _calculate_overall_status(services: Dict[str, ServiceDetails]) -> HealthStatus:
    """
    Calculate overall system status based on service health.
    
    Args:
        services: Dictionary of service health details
        
    Returns:
        Overall system health status
    """
    if not services:
        return HealthStatus.UNHEALTHY
    
    healthy_services = sum(1 for s in services.values() if s.status == ServiceStatus.HEALTHY)
    total_services = len(services)
    
    # If all services are healthy
    if healthy_services == total_services:
        return HealthStatus.HEALTHY
    
    # If more than half are healthy
    elif healthy_services > total_services / 2:
        return HealthStatus.DEGRADED
    
    # If half or less are healthy
    else:
        return HealthStatus.UNHEALTHY


# Readiness check helper functions
async def _check_database_readiness() -> bool:
    """Check if database is ready."""
    try:
        # TODO: Implement actual database readiness check
        await asyncio.sleep(0.001)
        return True
    except Exception:
        return False


async def _check_external_services_readiness() -> bool:
    """Check if external services are ready.""" 
    try:
        # TODO: Check external service dependencies
        await asyncio.sleep(0.001)
        return True
    except Exception:
        return False


async def _check_cache_readiness() -> bool:
    """Check if cache is ready."""
    try:
        # TODO: Implement actual cache readiness check
        await asyncio.sleep(0.001)
        return True
    except Exception:
        return False


# Detailed health metrics helper functions
async def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback if psutil not available
        return 512.0  # Mock value


async def _get_cpu_usage() -> float:
    """Get current CPU usage percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except ImportError:
        # Fallback if psutil not available
        return 25.0  # Mock value


async def _get_disk_usage() -> float:
    """Get current disk usage percentage."""
    try:
        import psutil
        return psutil.disk_usage('/').percent
    except ImportError:
        # Fallback if psutil not available
        return 45.0  # Mock value


async def _get_active_connections() -> int:
    """Get number of active connections."""
    # TODO: Implement actual connection count
    return 15  # Mock value


async def _get_request_rate() -> float:
    """Get current request rate per minute."""
    # TODO: Implement actual request rate calculation
    return 120.5  # Mock value


async def _get_error_rate() -> float:
    """Get current error rate percentage."""
    # TODO: Implement actual error rate calculation
    return 0.1  # Mock value