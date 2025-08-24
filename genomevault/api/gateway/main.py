"""
GenomeVault API Gateway Main Application

Comprehensive FastAPI gateway implementing OpenAPI specifications for
privacy-preserving genomic computing platform.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse

from genomevault.api.gateway.middleware.authentication import AuthenticationMiddleware
from genomevault.api.gateway.middleware.rate_limiting import RateLimitingMiddleware
from genomevault.api.gateway.middleware.error_handling import ErrorHandlingMiddleware
from genomevault.api.gateway.middleware.logging import LoggingMiddleware
from genomevault.api.gateway.routes import (
    algorithms,
    models,
    pipelines,
    proofs,
    queries,
    vectors,
    health,
    specialized,
    ai_models,
)
from genomevault.api.gateway.websockets import websocket_router
from genomevault.observability.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager for startup/shutdown events.
    """
    # Startup
    logger.info("Starting GenomeVault API Gateway")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Version: {app.version}")

    # Initialize services
    try:
        # Add any startup initialization here
        logger.info("Gateway initialized successfully")
        yield
    finally:
        # Shutdown
        logger.info("Shutting down GenomeVault API Gateway")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title="GenomeVault API Gateway",
        version="1.0.0",
        description="""
        Privacy-preserving genomic computing platform using hyperdimensional computing,
        zero-knowledge proofs, and federated learning for secure genomic data analysis.

        ## Privacy Model

        GenomeVault uses mathematical privacy guarantees rather than traditional encryption:
        - **Hyperdimensional Computing**: Genomic variants encoded as high-dimensional vectors
        - **Private Information Retrieval (PIR)**: Query databases without revealing what you're looking for
        - **Zero-Knowledge Proofs**: Verify computations without revealing the data
        - **Differential Privacy**: Mathematical bounds on information leakage

        ## Rate Limits

        API requests are subject to rate limiting:
        - **Standard tier**: 1000 requests/hour per API key
        - **Clinical tier**: 10000 requests/hour per API key
        - **Research tier**: 50000 requests/hour per API key
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        contact={
            "name": "GenomeVault Support",
            "url": "https://github.com/genomevault/genomevault",
            "email": "support@genomevault.io",
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        servers=[
            {
                "url": "https://api.genomevault.io/gateway/v1",
                "description": "Production server",
            },
            {
                "url": "https://staging-api.genomevault.io/gateway/v1",
                "description": "Staging server",
            },
            {
                "url": "http://localhost:8000/gateway/v1",
                "description": "Development server",
            },
        ],
    )

    # Configure CORS
    allowed_origins = [
        origin.strip()
        for origin in os.getenv("GENOMEVAULT_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]

    if not allowed_origins:
        allowed_origins = ["*"]  # Allow all origins in development

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-*", "X-Request-ID"],
    )

    # Add custom middleware in order
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RateLimitingMiddleware)
    app.add_middleware(AuthenticationMiddleware)

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(pipelines.router, prefix="/pipelines", tags=["Pipelines"])
    app.include_router(vectors.router, prefix="/vectors", tags=["Vectors"])
    app.include_router(proofs.router, prefix="/proofs", tags=["Proofs"])
    app.include_router(queries.router, prefix="/queries", tags=["Queries"])
    app.include_router(models.router, prefix="/models", tags=["Models"])
    app.include_router(algorithms.router, prefix="/algorithms", tags=["Algorithms"])
    app.include_router(specialized.router, prefix="/specialized", tags=["Specialized"])
    app.include_router(ai_models.router, prefix="/ai", tags=["AI Models"])
    app.include_router(websocket_router, prefix="/ws", tags=["WebSockets"])

    # Add root redirect
    @app.get("/", include_in_schema=False)
    async def redirect_to_docs() -> RedirectResponse:
        """Redirect root to documentation."""
        return RedirectResponse(url="/docs")

    # Custom OpenAPI schema
    def custom_openapi() -> dict[str, Any]:
        """Generate custom OpenAPI schema."""
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            servers=app.servers,
        )

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication. Get your API key from the GenomeVault console.",
            },
            "OAuth2": {
                "type": "oauth2",
                "description": "OAuth 2.0 with PKCE for secure authentication",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": "https://auth.genomevault.io/oauth/authorize",
                        "tokenUrl": "https://auth.genomevault.io/oauth/token",
                        "scopes": {
                            "genomic:read": "Read genomic data",
                            "pir:query": "Execute PIR queries",
                            "zk:prove": "Generate ZK proofs",
                            "clinical:analyze": "Clinical analysis",
                            "admin:manage": "Administrative access",
                        },
                    }
                },
            },
        }

        # Add rate limit headers
        openapi_schema["components"]["headers"] = {
            "X-RateLimit-Limit": {
                "description": "Request limit per time window",
                "schema": {"type": "integer", "example": 1000},
            },
            "X-RateLimit-Remaining": {
                "description": "Requests remaining in current window",
                "schema": {"type": "integer", "example": 999},
            },
            "X-RateLimit-Reset": {
                "description": "Time when rate limit resets (Unix timestamp)",
                "schema": {"type": "integer", "example": 1642248600},
            },
            "X-Request-ID": {
                "description": "Unique request identifier",
                "schema": {"type": "string", "format": "uuid"},
            },
        }

        # Add common responses
        openapi_schema["components"]["responses"] = {
            "RateLimited": {
                "description": "Rate limit exceeded",
                "headers": {
                    "X-RateLimit-Limit": {"$ref": "#/components/headers/X-RateLimit-Limit"},
                    "X-RateLimit-Remaining": {"$ref": "#/components/headers/X-RateLimit-Remaining"},
                    "X-RateLimit-Reset": {"$ref": "#/components/headers/X-RateLimit-Reset"},
                },
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
                },
            },
            "Unauthorized": {
                "description": "Authentication required",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
                },
            },
            "Forbidden": {
                "description": "Insufficient permissions",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
                },
            },
            "InternalError": {
                "description": "Internal server error",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
                },
            },
        }

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app


# Create the app instance
app = create_app()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next) -> Response:
    """Add process time header to responses."""
    import time

    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response
