from __future__ import annotations

import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from genomevault.core.exceptions import GenomeVaultError


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(GenomeVaultError)
    async def handle_domain_error(request: Request, exc: GenomeVaultError):
        return JSONResponse(
            status_code=400,
            content={
                "code": exc.__class__.__name__,
                "message": str(exc),
                "context": getattr(exc, "context", {}) or {},
            },
        )

    @app.exception_handler(Exception)
    async def handle_generic_error(request: Request, exc: Exception):
        rid = str(uuid.uuid4())
        return JSONResponse(status_code=500, content={"code": "InternalServerError", "message": "Internal error", "request_id": rid})
