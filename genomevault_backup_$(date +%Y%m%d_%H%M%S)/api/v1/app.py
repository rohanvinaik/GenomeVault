"""GenomeVault API v1 application factory."""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from genomevault.api.v1.versioning import deprecation_middleware
from genomevault.api.v1.routers import (
    health_router,
    hv_router,
    pir_router,
    zk_router,
    clinical_router,
)
from genomevault.api.v1.errors import register_error_handlers
from genomevault.api.middleware.rate_limiter import RateLimiterMiddleware
from genomevault.api.middleware.authentication import AuthenticationMiddleware


def create_app() -> FastAPI:
    """Create and configure FastAPI application for API v1."""

    app = FastAPI(
        title="GenomeVault API",
        version="1.0.0",
        description="Privacy-preserving genomic computing platform - API v1",
        docs_url="/v1/docs",
        redoc_url="/v1/redoc",
        openapi_url="/v1/openapi.json",
        # Custom OpenAPI schema
        openapi_tags=[
            {
                "name": "System",
                "description": "System health and status endpoints",
            },
            {
                "name": "Hypervector",
                "description": "Hyperdimensional computing operations",
            },
            {
                "name": "PIR",
                "description": "Private Information Retrieval queries",
            },
            {
                "name": "Zero-Knowledge",
                "description": "Zero-knowledge proof generation and verification",
            },
            {
                "name": "Clinical",
                "description": "HIPAA-compliant clinical genomic analysis",
            },
        ],
    )

    # CORS configuration
    allowed_origins = [
        origin.strip()
        for origin in os.getenv("GENOMEVAULT_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]

    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    # Custom middleware
    app.middleware("http")(deprecation_middleware)

    # Rate limiting middleware
    app.add_middleware(
        RateLimiterMiddleware,
        requests_per_minute=60,  # Default rate limit
        storage_uri=os.getenv("REDIS_URL", "memory://"),
    )

    # Authentication middleware
    app.add_middleware(
        AuthenticationMiddleware,
        api_key_header="X-API-Key",
        oauth_endpoint=os.getenv("OAUTH_ENDPOINT", "https://auth.genomevault.io"),
    )

    # Error handlers
    register_error_handlers(app)

    # API routers with v1 prefix
    app.include_router(health_router, prefix="/v1")
    app.include_router(hv_router, prefix="/v1")
    app.include_router(pir_router, prefix="/v1")
    app.include_router(zk_router, prefix="/v1")
    app.include_router(clinical_router, prefix="/v1")

    # Root endpoint for version discovery
    @app.get("/", include_in_schema=False)
    async def root():
        """API version discovery endpoint."""
        return JSONResponse(
            {
                "api": "GenomeVault",
                "version": "1.0.0",
                "versions": {
                    "v1": {"status": "active", "docs": "/v1/docs", "openapi": "/v1/openapi.json"}
                },
                "links": {
                    "documentation": "https://docs.genomevault.io",
                    "support": "https://github.com/genomevault/genomevault/issues",
                },
            }
        )

    # Version-specific health endpoint
    @app.get("/v1", include_in_schema=False)
    async def v1_info(request: Request):
        """Version 1 information endpoint."""
        return JSONResponse(
            {
                "version": "v1",
                "status": "active",
                "features": {
                    "hypervector_encoding": True,
                    "pir_queries": True,
                    "zk_proofs": True,
                    "clinical_analysis": True,
                    "federated_learning": False,  # Feature flag controlled
                },
                "deprecation": None,
                "docs": "/v1/docs",
            }
        )

    return app
