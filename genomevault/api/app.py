# genomevault/api/app.py
from __future__ import annotations
from fastapi import FastAPI
from genomevault.exceptions import GVError
from genomevault.api.errors import gv_error_handler  # you'll add this below if missing

app = FastAPI(title="GenomeVault API", version="0.1.0")


# Health routes (keeps tests happy)
@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "service": "GenomeVault API"}


# Uniform error responses
app.add_exception_handler(GVError, gv_error_handler)
