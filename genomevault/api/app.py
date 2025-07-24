"""
API entrypoint using FastAPI.
Implements core GenomeVault network endpoints.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from genomevault.utils.config import get_config

config = get_config()
from genomevault.api.routers.tuned_query import router as tuned_query_router
from genomevault.hypervector.error_handling import router as hdc_router
from genomevault.utils.logging import audit_logger, get_logger

# Create FastAPI app

logger = get_logger(__name__)

app = FastAPI(
    title="GenomeVault API",
    description="Privacy-preserving genomic analysis platform",
    version="3.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include HDC error handling router
app.include_router(hdc_router, prefix="/api")

# Include tuned query router
app.include_router(tuned_query_router, prefix="/api")


# Request/Response Models
class TopologyRequest(BaseModel):
    """Request for network topology information."""

    location: Optional[Dict[str, float]] = Field(
        None, description="User location for nearest node discovery"
    )
    node_requirements: Optional[Dict[str, Any]] = Field(
        None, description="Specific node requirements"
    )


class TopologyResponse(BaseModel):
    """Network topology response."""

    nearestLNs: List[str] = Field(description="List of nearest light node IDs")
    tsNodes: List[str] = Field(description="List of trusted signatory node IDs")
    optimal_configuration: Optional[Dict] = Field(
        None, description="Optimal node configuration for PIR"
    )


class CreditRedeemRequest(BaseModel):
    """Credit redemption request."""

    invoiceId: str = Field(description="Invoice identifier")
    creditsBurned: int = Field(description="Number of credits to burn", gt=0)
    purpose: str = Field(description="Purpose of credit redemption")


class CreditRedeemResponse(BaseModel):
    """Credit redemption response."""

    success: bool
    transactionId: str
    remainingCredits: int
    timestamp: float


class AuditChallengeRequest(BaseModel):
    """Audit challenge request."""

    challenger: str = Field(description="Challenging node ID")
    target: str = Field(description="Target node ID")
    epoch: int = Field(description="Epoch being challenged", gt=0)
    resultHash: str = Field(description="Expected result hash")


class AuditChallengeResponse(BaseModel):
    """Audit challenge response."""

    success: bool
    valid: bool
    challengeId: str
    slashAmount: Optional[int] = None
    timestamp: float


class ProofSubmissionRequest(BaseModel):
    """Zero-knowledge proof submission."""

    circuit_type: str
    proof_data: str
    public_inputs: Dict[str, Any]
    metadata: Optional[Dict] = None


class ProofSubmissionResponse(BaseModel):
    """Proof submission response."""

    proof_id: str
    transaction_hash: str
    status: str
    timestamp: float


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime_seconds: float
    services: Dict[str, str]


# Dependency for API key authentication
async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key for authentication."""
    # In production, validate against actual API keys
    if not x_api_key or len(x_api_key) < 32:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# Global state (in production, use proper state management)
app_start_time = time.time()
node_registry = {}
credit_ledger = {}


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint with health check."""
    return HealthCheckResponse(
        status="healthy",
        version="3.0.0",
        uptime_seconds=time.time() - app_start_time,
        services={
            "pir": "operational",
            "blockchain": "operational",
            "proofs": "operational",
        },
    )


@app.post("/topology", response_model=TopologyResponse)
async def get_topology(request: TopologyRequest, api_key: str = Depends(verify_api_key)):
    """
    Get network topology information.
    Returns nearest light nodes and trusted signatories.
    """
    try:
        # In production, would query actual node registry
        # For now, return mock data
        nearest_lns = ["ln_us_east_001", "ln_us_west_002", "ln_eu_central_001"]

        ts_nodes = ["ts_hospital_001", "ts_clinic_002"]

        # Calculate optimal configuration
        optimal_config = {
            "configuration": "1 LN + 2 TS",
            "expected_latency_ms": 210,
            "privacy_failure_probability": 4e-4,
            "communication_cost_kb": 150,
        }

        # Audit log
        audit_logger.log_event(
            event_type="api_access",
            actor=api_key[:8] + "...",
            action="get_topology",
            metadata={"location": request.location},
        )

        return TopologyResponse(
            nearestLNs=nearest_lns,
            tsNodes=ts_nodes,
            optimal_configuration=optimal_config,
        )

    except Exception as e:
        logger.error(f"Topology request failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/credit/vault/redeem", response_model=CreditRedeemResponse)
async def redeem_credits(request: CreditRedeemRequest, api_key: str = Depends(verify_api_key)):
    """
    Redeem credits from vault.
    Burns credits for network services.
    """
    try:
        # Verify user has sufficient credits
        user_id = api_key[:8]  # In production, map API key to user
        user_credits = credit_ledger.get(user_id, 0)

        if user_credits < request.creditsBurned:
            raise HTTPException(status_code=400, detail="Insufficient credits")

        # Process redemption
        tx_id = hashlib.sha256(f"{request.invoiceId}:{time.time()}".encode()).hexdigest()[:16]

        # Update ledger
        credit_ledger[user_id] = user_credits - request.creditsBurned

        # Audit log
        audit_logger.log_event(
            event_type="credit_redemption",
            actor=user_id,
            action="redeem_credits",
            resource=request.invoiceId,
            metadata={"amount": request.creditsBurned, "purpose": request.purpose},
        )

        return CreditRedeemResponse(
            success=True,
            transactionId=tx_id,
            remainingCredits=credit_ledger[user_id],
            timestamp=time.time(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Credit redemption failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/audit/challenge", response_model=AuditChallengeResponse)
async def create_audit_challenge(
    request: AuditChallengeRequest, api_key: str = Depends(verify_api_key)
):
    """
    Create audit challenge for node.
    Verifies node behavior and slashes stake if invalid.
    """
    try:
        # Generate challenge ID
        challenge_id = hashlib.sha256(
            f"{request.challenger}:{request.target}:{request.epoch}".encode()
        ).hexdigest()[:16]

        # In production, would verify challenge and process slashing
        # For now, simulate validation
        is_valid = True  # Placeholder
        slash_amount = None

        if not is_valid:
            # Calculate slash (25% of stake)
            target_stake = 1000  # Placeholder
            slash_amount = int(target_stake * 0.25)

        # Audit log
        audit_logger.log_event(
            event_type="audit_challenge",
            actor=request.challenger,
            action="challenge_node",
            resource=request.target,
            metadata={
                "epoch": request.epoch,
                "valid": is_valid,
                "slash_amount": slash_amount,
            },
        )

        return AuditChallengeResponse(
            success=True,
            valid=is_valid,
            challengeId=challenge_id,
            slashAmount=slash_amount,
            timestamp=time.time(),
        )

    except Exception as e:
        logger.error(f"Audit challenge failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/proof/submit", response_model=ProofSubmissionResponse)
async def submit_proof(request: ProofSubmissionRequest, api_key: str = Depends(verify_api_key)):
    """
    Submit zero-knowledge proof for recording on blockchain.
    """
    try:
        # Generate proof ID
        proof_id = hashlib.sha256(
            f"{request.circuit_type}:{request.proof_data[:32]}:{time.time()}".encode()
        ).hexdigest()[:16]

        # In production, would submit to blockchain
        tx_hash = hashlib.sha256(f"tx:{proof_id}".encode()).hexdigest()

        # Audit log
        audit_logger.log_event(
            event_type="proof_submission",
            actor=api_key[:8] + "...",
            action="submit_proof",
            resource=proof_id,
            metadata={
                "circuit_type": request.circuit_type,
                "public_inputs": request.public_inputs,
            },
        )

        return ProofSubmissionResponse(
            proof_id=proof_id,
            transaction_hash=tx_hash,
            status="pending",
            timestamp=time.time(),
        )

    except Exception as e:
        logger.error(f"Proof submission failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/node/stats")
async def get_node_statistics(api_key: str = Depends(verify_api_key)):
    """Get network node statistics."""
    try:
        # Calculate network statistics
        stats = {
            "total_nodes": len(node_registry),
            "trusted_signatories": sum(
                1 for n in node_registry.values() if n.get("is_trusted_signatory", False)
            ),
            "total_voting_power": sum(n.get("voting_power", 0) for n in node_registry.values()),
            "node_distribution": {
                "light": sum(1 for n in node_registry.values() if n.get("class") == "light"),
                "full": sum(1 for n in node_registry.values() if n.get("class") == "full"),
                "archive": sum(1 for n in node_registry.values() if n.get("class") == "archive"),
            },
            "network_health": "healthy",
        }

        return JSONResponse(content=stats)

    except Exception as e:
        logger.error(f"Failed to get node statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/pir/config")
async def get_pir_configuration(api_key: str = Depends(verify_api_key)):
    """Get PIR network configuration."""
    try:
        config_data = {
            "num_servers": config.pir.num_servers,
            "min_honest_servers": config.pir.min_honest_servers,
            "server_honesty_generic": config.pir.server_honesty_generic,
            "server_honesty_hipaa": config.pir.server_honesty_hipaa,
            "target_failure_probability": config.pir.target_failure_probability,
            "configurations": [
                {
                    "name": "3 LN + 2 TS",
                    "servers": 5,
                    "expected_latency_ms": 350,
                    "failure_probability": 4e-4,
                },
                {
                    "name": "1 LN + 2 TS",
                    "servers": 3,
                    "expected_latency_ms": 210,
                    "failure_probability": 4e-4,
                },
            ],
        }

        return JSONResponse(content=config_data)

    except Exception as e:
        logger.error(f"Failed to get PIR configuration: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("GenomeVault API starting up", extra={"privacy_safe": True})

    # Initialize mock data
    global node_registry, credit_ledger

    # Add some mock nodes
    node_registry = {
        "ln_us_east_001": {
            "class": "light",
            "is_trusted_signatory": False,
            "voting_power": 1,
            "region": "us-east",
        },
        "ts_hospital_001": {
            "class": "light",
            "is_trusted_signatory": True,
            "voting_power": 11,
            "region": "us-west",
        },
        "full_university_001": {
            "class": "full",
            "is_trusted_signatory": False,
            "voting_power": 4,
            "region": "eu-central",
        },
    }

    # Initialize some mock credits
    credit_ledger = {"testuser": 1000, "apikey12": 500}

    logger.info("GenomeVault API ready", extra={"privacy_safe": True})


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("GenomeVault API shutting down", extra={"privacy_safe": True})


if __name__ == "__main__":
    import uvicorn

    # Run the API server
    uvicorn.run(
        app,
        host=config.network.api_host,
        port=config.network.api_port,
        log_level="info",
    )
