#!/usr/bin/env python3
"""Simple API server for testing GenomeVault endpoints."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="GenomeVault API", version="0.1.0")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "GenomeVault API is running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0"
    )


@app.get("/api/v1/status")
async def status():
    """API status endpoint."""
    return {
        "status": "operational",
        "components": {
            "api": "healthy",
            "database": "not_configured",
            "cache": "not_configured"
        }
    }


if __name__ == "__main__":
    import uvicorn
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    uvicorn.run(app, host="127.0.0.1", port=port)