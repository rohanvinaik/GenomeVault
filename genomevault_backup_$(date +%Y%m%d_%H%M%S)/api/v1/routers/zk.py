"""Zero-knowledge proof router for API v1."""

from __future__ import annotations

from fastapi import APIRouter
from genomevault.api.routers.proofs import router as base_router

# Re-export the existing router with v1 tag updates
router = APIRouter(prefix="/zk", tags=["Zero-Knowledge"])

# Include all routes from base router
for route in base_router.routes:
    router.routes.append(route)
