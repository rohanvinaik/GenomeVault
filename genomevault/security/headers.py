"""Headers module."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    # HSTS disabled by default for local dev; enable behind TLS in production
    # "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
}


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """SecurityHeadersMiddleware implementation."""

    async def dispatch(self, request, call_next):
        """Async operation to Dispatch.

        Args:
            request: Client request.
            call_next: Call next.

        Returns:
            Operation result.
        """
        resp: Response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            if k not in resp.headers:
                resp.headers[k] = v
        return resp


def register_security(app: FastAPI, *, allow_origins: list[str] | None = None) -> None:
    """Register security.

    Args:
        app: App.
    """
    # CORS: default to no cross-site; override with explicit origins if needed
    if allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
    app.add_middleware(_SecurityHeadersMiddleware)
