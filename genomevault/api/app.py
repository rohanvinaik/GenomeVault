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

# Register routers individually to allow partial failures
try:
    import genomevault.api.routers.hv as hv
    app.include_router(hv.router)
except Exception as e:
    print(f"Warning: Could not import HV router: {e}")

try:
    import genomevault.api.routers.metrics as metrics
    app.include_router(metrics.router)
except Exception as e:
    print(f"Warning: Could not import metrics router: {e}")

try:
    import genomevault.api.routers.hdc as hdc
    app.include_router(hdc.router)
except Exception as e:
    print(f"Warning: Could not import HDC router: {e}")

try:
    import genomevault.api.routers.zk as zk
    app.include_router(zk.router)
except Exception as e:
    print(f"Warning: Could not import ZK router: {e}")

try:
    import genomevault.api.routers.pir as pir
    app.include_router(pir.router)
except Exception as e:
    print(f"Warning: Could not import PIR router: {e}")

# Analysis router - loaded separately to avoid dependency issues
try:
    import genomevault.api.routers.analysis as analysis
    app.include_router(analysis.router)
except Exception as e:  # pragma: no cover
    print(f"Warning: Could not import analysis router: {e}")

# Clinical query router - for querying ClinVar database
try:
    import genomevault.api.routers.clinical_query as clinical_query
    app.include_router(clinical_query.router)

    # Initialize clinical database at startup if available
    @app.on_event("startup")
    async def startup_clinical_database():
        try:
            clinical_query.init_clinical_database("data/clinical_snps_v1.0.0.json.gz")
        except Exception as e:
            print(f"Warning: Could not load clinical database: {e}")
except Exception as e:  # pragma: no cover
    print(f"Warning: Could not import clinical query router: {e}")

# GDiff/HDV router - for GDiff-based HDV encoding with caching
try:
    import genomevault.api.routers.gdiff as gdiff
    app.include_router(gdiff.router)
except Exception as e:  # pragma: no cover
    print(f"Warning: Could not import GDiff router: {e}")

app.add_exception_handler(GVError, gv_error_handler)
