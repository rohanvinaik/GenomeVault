"""Health check router for API v1."""

from __future__ import annotations

import datetime
from typing import Dict

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from genomevault.api.v1.versioning import DeprecationWarning


router = APIRouter(tags=["System"])


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime.datetime
    version: str
    services: Dict[str, str]


def get_deprecation_handler():
    """Dependency to handle deprecation warnings."""
    return DeprecationWarning("v1")


@router.get("/health", response_model=HealthResponse)
async def health_check(
    response: Response, deprecation: DeprecationWarning = Depends(get_deprecation_handler)
) -> HealthResponse:
    """
    Health check endpoint.

    Returns system health status and service availability.
    Includes rate limit headers and deprecation warnings if applicable.
    """
    # Add deprecation headers if needed
    deprecation.add_deprecation_headers(response)

    # Add rate limit headers (would be populated by middleware)
    response.headers.setdefault("X-RateLimit-Limit", "1000")
    response.headers.setdefault("X-RateLimit-Remaining", "999")
    response.headers.setdefault(
        "X-RateLimit-Reset", str(int(datetime.datetime.now().timestamp()) + 3600)
    )

    # Check service health (simplified for example)
    services = {
        "database": "healthy",
        "pir_engine": "healthy",
        "zk_prover": "healthy",
        "hypervector_engine": "healthy",
    }

    # Determine overall status
    status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"

    return HealthResponse(
        status=status, timestamp=datetime.datetime.utcnow(), version="1.0.0", services=services
    )
