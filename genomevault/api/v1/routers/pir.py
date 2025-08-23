"""PIR router for API v1."""

from __future__ import annotations

from fastapi import APIRouter
from genomevault.api.routers.pir import router as base_router

# Re-export the existing router with v1 tag updates
router = APIRouter(prefix="/pir", tags=["PIR"])

# Include all routes from base router
for route in base_router.routes:
    router.routes.append(route)
