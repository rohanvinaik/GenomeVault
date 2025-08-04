from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI

from genomevault.api.errors import \
    register_error_handlers  # created in task 26A
from genomevault.api.routers import (clinical, federated, governance, ledger,
                                     pir, proofs, vectors)
from genomevault.observability.logging import configure_logging
from genomevault.observability.metrics import metrics_router
from genomevault.observability.middleware import add_observability_middleware
from genomevault.security.auth import require_api_key
from genomevault.security.body_limit import MaximumBodySizeMiddleware
from genomevault.security.headers import register_security
from genomevault.security.rate_limit import RateLimitMiddleware

app = FastAPI(
    title="GenomeVault API",
    version="3.0.0",
    description="Privacy-preserving genomic data platform",
)

# configure observability
configure_logging()
add_observability_middleware(app)
app.include_router(metrics_router)

# configure security
register_security(app, allow_origins=None)  # pass explicit origins list in production
app.add_middleware(MaximumBodySizeMiddleware)
app.add_middleware(RateLimitMiddleware)

# Create a sub-router with auth dependency
protected = APIRouter(dependencies=[Depends(require_api_key)])
protected.include_router(vectors.router)
protected.include_router(proofs.router)
protected.include_router(pir.router)
protected.include_router(federated.router)
protected.include_router(clinical.router)
protected.include_router(ledger.router)

# Mount protected routes
app.include_router(protected)

# Include governance router
app.include_router(governance.router)


# health
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "3.0.0"}


# register error handlers
register_error_handlers(app)
