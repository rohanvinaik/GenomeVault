"""Health module."""

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", include_in_schema=False)
@router.get("/status", include_in_schema=False)
def health():
    """Health.

    Returns:
        Operation result.
    """
    return {"status": "ok"}
