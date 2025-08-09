# genomevault/api/example_usage.py
"""
Example showing how to use the standardized GV exception system in API endpoints.
"""

from fastapi import APIRouter
from genomevault.exceptions import (
    GVInputError,
    GVNotFound,
    GVComputeError,
    GVTimeout,
    GVSecurityError,
)

router = APIRouter()


@router.get("/example-input-validation/{item_id}")
async def validate_input_example(item_id: str):
    """Example of input validation using GVInputError."""
    if not item_id or len(item_id) < 3:
        raise GVInputError(
            "Item ID must be at least 3 characters",
            details={"field": "item_id", "min_length": 3},
        )

    return {"item_id": item_id, "status": "valid"}


@router.get("/example-not-found/{resource_id}")
async def not_found_example(resource_id: str):
    """Example of resource not found using GVNotFound."""
    # Simulate database lookup
    if resource_id == "missing":
        raise GVNotFound(
            f"Resource {resource_id} not found",
            details={"resource_type": "genomic_data", "id": resource_id},
        )

    return {"resource_id": resource_id, "data": "found"}


@router.post("/example-compute-error")
async def compute_error_example():
    """Example of computation error using GVComputeError."""
    try:
        # Simulate a computation that fails
        result = 1 / 0
        return {"result": result}
    except ZeroDivisionError as e:
        raise GVComputeError(
            "Mathematical computation failed",
            details={"operation": "division", "error_type": "zero_division"},
        ) from e


@router.post("/example-timeout")
async def timeout_example():
    """Example of timeout error using GVTimeout."""
    # Simulate a long-running operation
    import asyncio

    try:
        await asyncio.wait_for(asyncio.sleep(10), timeout=1.0)
        return {"status": "completed"}
    except asyncio.TimeoutError as e:
        raise GVTimeout(
            "Operation timed out after 1 second",
            details={"timeout_seconds": 1.0, "operation": "data_processing"},
        ) from e


@router.post("/example-security-error")
async def security_error_example():
    """Example of security error using GVSecurityError."""
    # Simulate access control check
    user_role = "guest"  # This would come from auth middleware

    if user_role != "admin":
        raise GVSecurityError(
            "Insufficient permissions for this operation",
            details={"required_role": "admin", "user_role": user_role},
        )

    return {"status": "authorized", "data": "sensitive_info"}


# How to register this router in your main app:
# from genomevault.api.example_usage import router as example_router
# app.include_router(example_router, prefix="/api/v1", tags=["examples"])
