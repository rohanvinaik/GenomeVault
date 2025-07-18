"""
GenomeVault API Application
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
import time

from core.config import get_config
from api.routers import topology, credit, audit, health, beacon
from api.middleware.auth import AuthMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.middleware.privacy import DifferentialPrivacyMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting GenomeVault API...")
    logger.info(f"Node type: {config.node_type}")
    logger.info(f"Signatory status: {config.signatory_status}")
    logger.info(f"Total voting power: {config.total_voting_power}")
    logger.info(f"Credits per block: {config.credits_per_block}")
    
    # Initialize connections and services here
    # TODO: Initialize blockchain connection
    # TODO: Initialize PIR network connection
    # TODO: Initialize IPFS connection
    
    yield
    
    # Shutdown
    logger.info("Shutting down GenomeVault API...")
    # TODO: Clean up connections


# Create FastAPI application
app = FastAPI(
    title="GenomeVault API",
    description="Privacy-preserving genomic data platform",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs" if config.api_host == "0.0.0.0" else None,  # Disable in production
    redoc_url="/redoc" if config.api_host == "0.0.0.0" else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Custom middleware
app.add_middleware(RateLimitMiddleware)
app.add_middleware(DifferentialPrivacyMiddleware)
app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(topology.router, prefix="/topology", tags=["network"])
app.include_router(credit.router, prefix="/credit", tags=["credits"])
app.include_router(audit.router, prefix="/audit", tags=["audit"])
app.include_router(beacon.router, prefix="/beacon", tags=["beacon"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "service": "GenomeVault",
        "version": "3.0.0",
        "node": {
            "type": config.node_type,
            "signatory": config.signatory_status,
            "voting_power": config.total_voting_power,
            "credits_per_block": config.credits_per_block
        },
        "features": {
            "diabetes_pilot": config.enable_diabetes_pilot,
            "pharmacogenomics": config.enable_pharmacogenomics,
            "trial_matching": config.enable_trial_matching
        },
        "endpoints": {
            "health": "/health",
            "topology": "/topology",
            "credits": "/credit",
            "audit": "/audit",
            "beacon": "/beacon/v2",
            "docs": "/docs" if config.api_host == "0.0.0.0" else None
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return {
        "error": "Not Found",
        "message": f"The requested endpoint {request.url.path} does not exist",
        "status_code": 404
    }

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal error: {str(exc)}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }

# Metrics endpoint (Prometheus format)
@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint"""
    # TODO: Implement actual metrics collection
    metrics_data = f"""
# HELP genomevault_node_info Node information
# TYPE genomevault_node_info gauge
genomevault_node_info{{type="{config.node_type}",signatory="{config.signatory_status}"}} 1

# HELP genomevault_voting_power Total voting power of node
# TYPE genomevault_voting_power gauge
genomevault_voting_power {config.total_voting_power}

# HELP genomevault_credits_per_block Credits earned per block
# TYPE genomevault_credits_per_block gauge
genomevault_credits_per_block {config.credits_per_block}

# HELP genomevault_api_requests_total Total API requests
# TYPE genomevault_api_requests_total counter
genomevault_api_requests_total 0

# HELP genomevault_proofs_generated_total Total ZK proofs generated
# TYPE genomevault_proofs_generated_total counter
genomevault_proofs_generated_total 0
"""
    return metrics_data


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.app:app",
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        log_level=config.log_level.lower()
    )
