from __future__ import annotations

"""Body Limit module."""
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse


class MaximumBodySizeMiddleware(BaseHTTPMiddleware):
    """MaximumBodySizeMiddleware implementation."""

    def __init__(self, app, max_body: int | None = None):
        """Initialize instance.

        Args:
            app: App.
            max_body: Max body.
        """
        super().__init__(app)
        self.max_body = int(
            max_body or int(os.getenv("MAX_BODY_SIZE", "10485760"))
        )  # 10 MiB default

    async def dispatch(self, request: Request, call_next):
        """Async operation to Dispatch.

        Args:
            request: Client request.
            call_next: Call next.

        Returns:
            Operation result.
        """
        # For multipart, rely on Content-Length; otherwise, read body into memory once (ASGI servers buffer anyway)
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > self.max_body:
            return PlainTextResponse("Request entity too large", status_code=413)
        body = await request.body()
        if len(body) > self.max_body:
            return PlainTextResponse("Request entity too large", status_code=413)
        # Recreate request with body for downstream
        request._body = body  # FastAPI/TestClient compatibility
        return await call_next(request)
