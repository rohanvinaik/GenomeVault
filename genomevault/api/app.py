# genomevault/api/app.py
from __future__ import annotations

"""App module."""
"""App module."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from genomevault.api.errors import gv_error_handler  # you'll add this below if missing
from genomevault.api.routers import healthz
from genomevault.exceptions import GVError

app = FastAPI(
    title="GenomeVault API",
    version="0.1.0",
    description="Privacy-preserving genomic computing platform",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(healthz.router)

# Import and include API v1 routers
try:
    from genomevault.api.routers import hv, metrics

    app.include_router(hv.router)
    app.include_router(metrics.router)
except ImportError as e:
    # Log but don't fail if routers aren't available yet
    print(f"Warning: Could not import API routers: {e}")

# Uniform error responses
app.add_exception_handler(GVError, gv_error_handler)
