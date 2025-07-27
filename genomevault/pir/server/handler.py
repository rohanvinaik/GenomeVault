"""
PIR server request handler with fixed-size responses and timing protection.
Implements the server-side logic for IT-PIR protocol.
"""
import asyncio
import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import jsonschema
import numpy as np
from aiohttp import web

from genomevault.utils.logging import audit_logger, get_logger, logger
from genomevault.version import PIR_PROTOCOL_VERSION

from .pir_server import PIRServer

# Load schemas

logger = get_logger(__name__)

with open("schemas/pir_query.json", "r") as f:
    QUERY_SCHEMA = json.load(f)

with open("schemas/pir_response.json", "r") as f:
    RESPONSE_SCHEMA = json.load(f)


class PIRHandler:
    """HTTP request handler for PIR queries."""

    def __init__(self, pir_server: PIRServer) -> None:
            """TODO: Add docstring for __init__"""
    self.pir_server = pir_server
        self.query_cache = {}  # For replay detection
        self.cache_max_size = 10000

    async def handle_query(self, request: web.Request) -> web.Response:
           """TODO: Add docstring for handle_query"""
     """
        Handle incoming PIR query request.

        Args:
            request: HTTP request containing PIR query

        Returns:
            HTTP response with PIR result
        """
        start_time = time.time()

        try:
            # Parse request
            request_data = await request.json()

            # Validate against schema
            try:
                jsonschema.validate(request_data, QUERY_SCHEMA)
            except jsonschema.ValidationError as e:
                return self._error_response(
                    "INVALID_SCHEMA", f"Query validation failed: {str(e)}", 400
                )

            # Check protocol version
            if request_data.get("protocol_version") not in ["1.0", "1.1"]:
                return self._error_response(
                    "UNSUPPORTED_VERSION",
                    f"Protocol version {request_data.get('protocol_version')} not supported",
                    400,
                )

            # Replay protection
            query_id = request_data["query_id"]
            nonce = request_data.get("nonce")

            if nonce and nonce in self.query_cache:
                return self._error_response("REPLAY_DETECTED", "Query nonce already used", 400)

            # Add to cache
            if nonce:
                self._add_to_cache(nonce)

            # Process query
            response_data = await self.pir_server.process_query(request_data)

            # Ensure fixed-size response (1024 bytes)
            response_bytes = self._ensure_fixed_size(response_data["response"])

            # Build response
            response = {
                "query_id": query_id,
                "server_id": self.pir_server.server_id,
                "response_data": base64.b64encode(response_bytes).decode("ascii"),
                "computation_time_ms": response_data["computation_time_ms"],
                "timestamp": time.time(),
                "server_type": "TS" if self.pir_server.is_trusted_signatory else "LN",
            }

            # Validate response
            jsonschema.validate(response, RESPONSE_SCHEMA)

            # Add timing padding to prevent timing attacks
            await self._timing_padding(start_time)

            # Audit log (privacy-safe)
            audit_logger.log_event(
                event_type="pir_query_success",
                actor="anonymous",
                action="query_processed",
                resource=self.pir_server.server_id,
                metadata={
                    "query_id": query_id,
                    "computation_time_ms": response_data["computation_time_ms"],
                    "response_size": len(response_bytes),
                },
            )

            return web.json_response(response)

        except Exception as e:
            logger.error(f"Error handling PIR query: {str(e)}")
            return self._error_response("INTERNAL_ERROR", "Query processing failed", 500)

    def _ensure_fixed_size(self, data: Any) -> bytes:
           """TODO: Add docstring for _ensure_fixed_size"""
     """
        Ensure response is exactly 1024 bytes.

        Args:
            data: Response data

        Returns:
            Fixed-size byte array
        """
        if isinstance(data, list):
            # Convert from list to bytes
            data_bytes = bytes(data)
        elif isinstance(data, np.ndarray):
            # Convert from numpy array
            data_bytes = data.tobytes()
        else:
            # Already bytes
            data_bytes = data

        # Pad or truncate to 1024 bytes
        if len(data_bytes) < 1024:
            # Pad with random bytes
            padding_size = 1024 - len(data_bytes)
            padding = np.random.bytes(padding_size)
            return data_bytes + padding
        else:
            # Truncate
            return data_bytes[:1024]

    async def _timing_padding(self, start_time: float) -> None:
           """TODO: Add docstring for _timing_padding"""
     """
        Add timing padding to prevent timing attacks.

        Args:
            start_time: Query processing start time
        """
        # Target 100ms response time
        target_time = 0.1  # 100ms
        elapsed = time.time() - start_time

        if elapsed < target_time:
            await asyncio.sleep(target_time - elapsed)

    def _add_to_cache(self, nonce: str) -> None:
           """TODO: Add docstring for _add_to_cache"""
     """Add nonce to replay cache with LRU eviction."""
        if len(self.query_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest = min(self.query_cache.items(), key=lambda x: x[1])
            del self.query_cache[oldest[0]]

        self.query_cache[nonce] = time.time()

    def _error_response(self, code: str, message: str, status: int) -> web.Response:
           """TODO: Add docstring for _error_response"""
     """Create error response."""
        response = {"error": {"code": code, "message": message}, "timestamp": time.time()}

        return web.json_response(response, status=status)

    async def handle_health(self, request: web.Request) -> web.Response:
           """TODO: Add docstring for handle_health"""
     """Health check endpoint."""
        stats = self.pir_server.get_server_statistics()

        health = {
            "status": "healthy",
            "server_id": self.pir_server.server_id,
            "server_type": "TS" if self.pir_server.is_trusted_signatory else "LN",
            "protocol_version": PIR_PROTOCOL_VERSION,
            "statistics": stats,
        }

        return web.json_response(health)

    async def handle_info(self, request: web.Request) -> web.Response:
           """TODO: Add docstring for handle_info"""
     """Server information endpoint."""
        info = {
            "server_id": self.pir_server.server_id,
            "server_type": "TS" if self.pir_server.is_trusted_signatory else "LN",
            "protocol_version": PIR_PROTOCOL_VERSION,
            "supported_versions": ["1.0", "1.1"],
            "database_size": self.pir_server.database_size,
            "shard_count": len(self.pir_server.shards),
            "features": {
                "batch_queries": True,
                "compression": True,
                "timing_protection": True,
                "replay_protection": True,
            },
        }

        return web.json_response(info)


def create_app(pir_server: PIRServer) -> web.Application:
       """TODO: Add docstring for create_app"""
     """
    Create aiohttp application for PIR server.

    Args:
        pir_server: PIR server instance

    Returns:
        Configured aiohttp application
    """
    app = web.Application()
    handler = PIRHandler(pir_server)

    # Add routes
    app.router.add_post("/pir/query", handler.handle_query)
    app.router.add_get("/health", handler.handle_health)
    app.router.add_get("/info", handler.handle_info)

    # Add middleware for logging
    @web.middleware
    async def logging_middleware(request, handler) -> None:
            """TODO: Add docstring for logging_middleware"""
    start = time.time()
        try:
            response = await handler(request)
            elapsed = (time.time() - start) * 1000
            logger.info(
                f"{request.method} {request.path} - {response.status} - {elapsed:.1f}ms",
                extra={"privacy_safe": True},
            )
            return response
        except web.HTTPException:
            raise
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(f"{request.method} {request.path} - ERROR - {elapsed:.1f}ms - {str(e)}")
            raise

    app.middlewares.append(logging_middleware)

    return app


if __name__ == "__main__":
    # Example server startup
    from pathlib import Path

    # Create PIR server
    data_dir = Path("data/pir_shards")
    server = PIRServer("server_1", data_dir, is_trusted_signatory=True)

    # Create web app
    app = create_app(server)

    # Run server
    web.run_app(app, host="0.0.0.0", port=8080)
