"""GenomeVault FastAPI Application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers (create minimal versions if they don't exist)
from genomevault.api.routers import encode, health

app = FastAPI(
    title="GenomeVault",
    description="Privacy-preserving genomic data platform",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)  # Health at /health for CI
app.include_router(encode.router, prefix="/api/v1", tags=["encoding"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to GenomeVault API",
        "version": "0.1.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
