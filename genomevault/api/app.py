from __future__ import annotations

from fastapi import FastAPI

from genomevault.api.routers import vectors
from genomevault.api.routers import proofs
from genomevault.api.routers import pir
from genomevault.api.routers import federated
from genomevault.api.routers import ledger
from genomevault.api.routers import clinical
from genomevault.api.errors import register_error_handlers  # created in task 26A

app = FastAPI(title="GenomeVault API", version="3.0.0", description="Privacy-preserving genomic data platform")

# include routers
app.include_router(vectors.router)
app.include_router(proofs.router)
app.include_router(pir.router)
app.include_router(federated.router)
app.include_router(ledger.router)
app.include_router(clinical.router)

# health
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "3.0.0"}

# register error handlers
register_error_handlers(app)
