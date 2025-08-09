# genomevault/api/errors.py
from __future__ import annotations

"""Errors module."""
"""Errors module."""
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from genomevault.exceptions import GVError


async def gv_error_handler(request: Request, exc: GVError) -> JSONResponse:
    """Global exception handler for GV errors that returns consistent JSON."""
    body = exc.to_dict()
    return JSONResponse(status_code=getattr(exc, "http_status", 500), content=body)


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Generic exception handler for unexpected errors."""
    rid = str(uuid.uuid4())
    return JSONResponse(
        status_code=500,
        content={
            "type": "InternalServerError",
            "code": "GV_INTERNAL",
            "message": "Internal error",
            "details": {"request_id": rid},
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all error handlers for the FastAPI app."""
    app.add_exception_handler(GVError, gv_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)
