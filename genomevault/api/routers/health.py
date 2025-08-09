from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
