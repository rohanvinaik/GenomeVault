from __future__ import annotations

from fastapi import FastAPI

from genomevault.api.routers import vectors
from genomevault.api.routers import proofs
from genomevault.api.errors import register_error_handlers  # created in task 26A

app = FastAPI(title="GenomeVault API", version="3.0.0", description="Privacy-preserving genomic data platform")

# include routers
app.include_router(vectors.router)
app.include_router(proofs.router)

# health
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "3.0.0"}

# register error handlers
register_error_handlers(app)
