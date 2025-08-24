"""
Health check models for GenomeVault API Gateway.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from pydantic import Field

from genomevault.api.gateway.models.base import BaseModel


class HealthStatus(str, Enum):
    """Overall system health status."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ServiceStatus(str, Enum):
    """Individual service status."""
    
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceDetails(BaseModel):
    """Details for individual service health."""
    
    status: ServiceStatus = Field(..., description="Service health status")
    response_time_ms: Optional[float] = Field(None, description="Service response time in milliseconds")
    last_check: datetime = Field(..., description="Last health check timestamp")
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")
    metadata: Optional[Dict[str, str]] = Field(None, description="Additional service metadata")


class HealthCheckResponse(BaseModel):
    """System health check response."""
    
    status: HealthStatus = Field(..., description="Overall system health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: int = Field(..., description="System uptime in seconds")
    services: Dict[str, ServiceDetails] = Field(..., description="Individual service health details")
    system_info: Optional[Dict[str, str]] = Field(None, description="System information")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "uptime_seconds": 3600,
                "services": {
                    "database": {
                        "status": "healthy",
                        "response_time_ms": 12.5,
                        "last_check": "2024-01-15T10:30:00Z"
                    },
                    "pir_engine": {
                        "status": "healthy",
                        "response_time_ms": 25.3,
                        "last_check": "2024-01-15T10:30:00Z"
                    },
                    "zk_prover": {
                        "status": "healthy",
                        "response_time_ms": 45.7,
                        "last_check": "2024-01-15T10:30:00Z"
                    }
                },
                "system_info": {
                    "environment": "production",
                    "region": "us-west-2",
                    "instance_id": "i-1234567890abcdef0"
                }
            }
        }


class DetailedHealthResponse(HealthCheckResponse):
    """Detailed health check response with additional metrics."""
    
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    cpu_usage_percent: Optional[float] = Field(None, description="CPU usage percentage")
    disk_usage_percent: Optional[float] = Field(None, description="Disk usage percentage")
    active_connections: Optional[int] = Field(None, description="Active database connections")
    request_rate_per_minute: Optional[float] = Field(None, description="Current request rate per minute")
    error_rate_percent: Optional[float] = Field(None, description="Error rate percentage (last 5 minutes)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "uptime_seconds": 3600,
                "services": {
                    "database": {
                        "status": "healthy",
                        "response_time_ms": 12.5,
                        "last_check": "2024-01-15T10:30:00Z"
                    }
                },
                "memory_usage_mb": 512.7,
                "cpu_usage_percent": 23.4,
                "disk_usage_percent": 45.2,
                "active_connections": 15,
                "request_rate_per_minute": 120.5,
                "error_rate_percent": 0.1
            }
        }


class ReadinessCheckResponse(BaseModel):
    """Kubernetes readiness check response."""
    
    ready: bool = Field(..., description="Service readiness status")
    timestamp: datetime = Field(..., description="Check timestamp")
    checks: Dict[str, bool] = Field(..., description="Individual readiness checks")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "ready": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "checks": {
                    "database_connection": True,
                    "external_services": True,
                    "cache_available": True
                }
            }
        }


class LivenessCheckResponse(BaseModel):
    """Kubernetes liveness check response."""
    
    alive: bool = Field(..., description="Service liveness status")
    timestamp: datetime = Field(..., description="Check timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "alive": True,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
                }
            }
        }
    }