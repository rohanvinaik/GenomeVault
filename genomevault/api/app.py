"""App module."""

from __future__ import annotations

import os

from fastapi import FastAPI

from genomevault.api.errors import gv_error_handler
from genomevault.api.routers.healthz import router as healthz_router
from genomevault.exceptions import GVError
from genomevault.security import register_security

allowed_origins = [
    origin.strip()
    for origin in os.getenv("GENOMEVAULT_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app = FastAPI(
    title="GenomeVault API",
    version="0.1.0",
    description="Privacy-preserving genomic computing platform",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

register_security(app, allow_origins=allowed_origins)

app.include_router(healthz_router)

try:
    import genomevault.api.routers.hv as hv
    import genomevault.api.routers.metrics as metrics
    import genomevault.api.routers.zk as zk
    import genomevault.api.routers.hdc as hdc
    import genomevault.api.routers.pir as pir

    app.include_router(hv.router)
    app.include_router(metrics.router)
    app.include_router(zk.router)
    app.include_router(hdc.router)
    app.include_router(pir.router)
except Exception as e:  # pragma: no cover - optional routers may fail
    print(f"Warning: Could not import API routers: {e}")

# Analysis router - loaded separately to avoid dependency issues
try:
    import genomevault.api.routers.analysis as analysis
    app.include_router(analysis.router)
except Exception as e:  # pragma: no cover
    print(f"Warning: Could not import analysis router: {e}")

app.add_exception_handler(GVError, gv_error_handler)
