"""
GenomeVault Core API Implementation

Implements the main API endpoints as specified in the System Breakdown:
- Network topology management
- Credit and vault operations
- Audit and challenge handling
- Client-facing endpoints for pipelines, vectors, and proofs
"""

import hashlib
import json
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from genomevault.blockchain.node import BlockchainNode, NodeInfo
from genomevault.core.config import get_config
from genomevault.core.constants import DEFAULT_PIPELINE_TYPES, NodeType
from genomevault.utils.logging import audit_logger, logger

config = get_config()

# Initialize FastAPI app
app = FastAPI(
    title="GenomeVault API",
    description="Privacy-preserving genomic data platform",
    version="3.0.0",
)

# Security
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, would use proper state management)
node_registry: dict[str, NodeInfo] = {}
blockchain_node: BlockchainNode | None = None


# Pydantic models for API
class TopologyRequest(BaseModel):
    """Request for network topology information"""

    node_id: str
    location: dict[str, float] | None = None  # lat, lon


class TopologyResponse(BaseModel):
    """Network topology response"""

    nearestLNs: list[str]
    tsNodes: list[str]


class CreditVaultRequest(BaseModel):
    """Credit vault operation request"""

    invoiceId: str
    creditsBurned: int
    proof: str | None = None


class CreditVaultResponse(BaseModel):
    """Credit vault operation response"""

    success: bool
    transactionId: str
    remainingCredits: int


class AuditChallengeRequest(BaseModel):
    """Audit challenge request"""

    challenger: str
    target: str
    epoch: int
    resultHash: str


class AuditChallengeResponse(BaseModel):
    """Audit challenge response"""

    success: bool
    valid: bool
    slashAmount: int | None = None
    rewardAmount: int | None = None


class PipelineRequest(BaseModel):
    """Processing pipeline request"""

    pipeline_type: str
    params: dict[str, Any] = {}


class PipelineCreateResponse(BaseModel):
    """Processing pipeline response"""

    job_id: str
    pipeline_type: str
    accepted: bool


class VectorRequest(BaseModel):
    """Hypervector operation request"""

    operation: str  # encode, bind, similarity
    data: dict[str, Any]
    domain: str | None = "general"


class VectorResponse(BaseModel):
    """Hypervector operation response"""

    result: Any
    metadata: dict[str, Any]


class ProofRequest(BaseModel):
    """Zero-knowledge proof request"""

    circuit_name: str
    public_inputs: dict[str, Any]
    private_inputs: dict[str, Any]


class ProofResponse(BaseModel):
    """Zero-knowledge proof response"""

    proof_id: str
    proof_data: str  # Base64 encoded
    verification_key: str | None = None
    metadata: dict[str, Any]


# Dependency for authentication
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """Verify JWT token and return user ID"""
    # In production, would verify actual JWT
    # For now, return mock user ID
    return "user_" + hashlib.sha256(credentials.credentials.encode()).hexdigest()[:8]


# Network API Endpoints


@app.post("/topology", response_model=TopologyResponse)
async def get_network_topology(
    request: TopologyRequest, user_id: str = Depends(verify_token)
):
    """
    Get network topology information for optimal PIR server selection.

    Returns nearest light nodes (LN) and trusted signatories (TS).
    """
    logger.info("Topology request from %s", request.node_id)

    # Get all nodes
    light_nodes = []
    ts_nodes = []

    for node_id, node_info in node_registry.items():
        if node_info.signatory:
            ts_nodes.append(node_id)
        elif node_info.node_type == NodeType.LIGHT.value:
            light_nodes.append(node_id)

    # Sort by geographic proximity if location provided
    if request.location:
        # Simplified - in production would use actual geographic calculations
        light_nodes.sort(key=lambda n: hash(n))
        ts_nodes.sort(key=lambda n: hash(n))

    # Return optimal configuration (1 LN + 2 TS or 3 LN + 2 TS)
    response = TopologyResponse(nearestLNs=light_nodes[:3], tsNodes=ts_nodes[:2])

    return response


@app.post("/credit/vault/redeem", response_model=CreditVaultResponse)
async def redeem_credits(
    request: CreditVaultRequest, user_id: str = Depends(verify_token)
):
    """
    Redeem credits from vault for services.

    Burns credits and processes payment/service delivery.
    """
    logger.info("Credit redemption: %s for %s", request.creditsBurned, user_id)

    # Verify user has sufficient credits
    user_credits = _get_user_credits(user_id)

    if user_credits < request.creditsBurned:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="Insufficient credits"
        )

    # Process redemption
    tx_id = _process_credit_redemption(
        user_id, request.invoiceId, request.creditsBurned
    )

    # Get remaining balance
    remaining = user_credits - request.creditsBurned

    # Audit log
    audit_logger.log_event(
        event_type="credit_redemption",
        actor=user_id,
        action="redeem_credits",
        resource=request.invoiceId,
        metadata={
            "credits_burned": request.creditsBurned,
            "remaining_credits": remaining,
        },
    )

    response = CreditVaultResponse(
        success=True, transactionId=tx_id, remainingCredits=remaining
    )

    return response


@app.post("/audit/challenge", response_model=AuditChallengeResponse)
async def create_audit_challenge(
    request: AuditChallengeRequest, user_id: str = Depends(verify_token)
):
    """
    Create audit challenge for node verification.

    Validates node behavior and handles slashing if needed.
    """
    logger.info("Audit challenge from %s to %s", request.challenger, request.target)

    if not blockchain_node:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Blockchain node not initialized",
        )

    # Process challenge
    result = await blockchain_node.handle_audit_challenge(
        challenger=request.challenger, target=request.target, epoch=request.epoch
    )

    response = AuditChallengeResponse(
        success=result["success"],
        valid=result.get("valid", False),
        slashAmount=result.get("slash_amount"),
        rewardAmount=result.get("reward_amount"),
    )

    return response


# Client API Endpoints


@app.post("/pipelines/create", response_model=PipelineCreateResponse)
async def create_processing_pipeline(
    request: PipelineRequest, user_id: str = Depends(verify_token)
):
    """
    Create new data processing pipeline job.

    Initiates local processing for specified omics type.
    """
    if request.pipeline_type not in DEFAULT_PIPELINE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported pipeline type")

    job_id = f"{user_id}:{request.pipeline_type}:{int(datetime.now().timestamp())}"
    logger.info("Pipeline requested: %s by %s", request.pipeline_type, user_id)

    _queue_pipeline_job(job_id, request)

    return PipelineCreateResponse(
        job_id=job_id, pipeline_type=request.pipeline_type, accepted=True
    )


@app.get("/pipelines/{job_id}")
async def get_pipeline_status(job_id: str, user_id: str = Depends(verify_token)):
    """Get status of processing pipeline job."""
    # In production, would check actual job status
    return {
        "job_id": job_id,
        "status": "processing",
        "progress": 0.45,
        "estimated_completion": datetime.now().isoformat(),
    }


@app.post("/vectors", response_model=VectorResponse)
async def perform_vector_operation(
    request: VectorRequest, user_id: str = Depends(verify_token)
):
    """
    Perform hypervector operations.

    Supports encoding, binding, and similarity operations.
    """
    logger.info("Vector operation: %s", request.operation)

    if request.operation == "encode":
        # Encode data to hypervector
        result = _encode_to_hypervector(request.data, request.domain)
        metadata = {
            "operation": "encode",
            "domain": request.domain,
            "dimensions": len(result) if isinstance(result, list) else "hierarchical",
        }

    elif request.operation == "bind":
        # Bind two hypervectors
        if "vector1" not in request.data or "vector2" not in request.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Binding requires vector1 and vector2",
            )

        result = _bind_hypervectors(
            request.data["vector1"],
            request.data["vector2"],
            request.data.get("method", "circular"),
        )
        metadata = {
            "operation": "bind",
            "method": request.data.get("method", "circular"),
        }

    elif request.operation == "similarity":
        # Calculate similarity
        if "vector1" not in request.data or "vector2" not in request.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Similarity requires vector1 and vector2",
            )

        result = _calculate_similarity(
            request.data["vector1"],
            request.data["vector2"],
            request.data.get("metric", "cosine"),
        )
        metadata = {
            "operation": "similarity",
            "metric": request.data.get("metric", "cosine"),
            "value": result,
        }

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown operation: {request.operation}",
        )

    response = VectorResponse(result=result, metadata=metadata)

    return response


@app.post("/proofs", response_model=ProofResponse)
async def generate_proof(request: ProofRequest, user_id: str = Depends(verify_token)):
    """
    Generate zero-knowledge proof.

    Creates privacy-preserving proofs for various circuits.
    """
    logger.info("Proof generation request: %s", request.circuit_name)

    # Validate circuit
    valid_circuits = [
        "variant_presence",
        "polygenic_risk_score",
        "ancestry_composition",
        "pharmacogenomic",
        "pathway_enrichment",
        "diabetes_risk_alert",
    ]

    if request.circuit_name not in valid_circuits:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid circuit. Must be one of: {valid_circuits}",
        )

    # Generate proof (simplified)
    proof_id = hashlib.sha256(
        f"{user_id}:{request.circuit_name}:{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]

    # Mock proof data
    proof_data = hashlib.sha256(
        json.dumps(
            {
                "circuit": request.circuit_name,
                "public": request.public_inputs,
                "timestamp": datetime.now().isoformat(),
            }
        ).encode()
    ).hexdigest()

    response = ProofResponse(
        proof_id=proof_id,
        proof_data=proof_data,
        metadata={
            "circuit": request.circuit_name,
            "generation_time_ms": 23.5,  # Mock timing
            "proof_size_bytes": 384,
        },
    )

    # Audit log
    audit_logger.log_event(
        event_type="proof_generation",
        actor=user_id,
        action="generate_proof",
        resource=proof_id,
        metadata={"circuit": request.circuit_name},
    )

    return response


@app.get("/proofs/{proof_id}/verify")
async def verify_proof(proof_id: str, user_id: str = Depends(verify_token)):
    """Verify a zero-knowledge proof."""
    # In production, would perform actual verification
    return {"proof_id": proof_id, "valid": True, "verification_time_ms": 12.3}


# Health and status endpoints


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/status")
async def get_status(user_id: str = Depends(verify_token)):
    """Get system status."""
    return {
        "blockchain_height": blockchain_node.current_height if blockchain_node else 0,
        "active_nodes": len(node_registry),
        "network_voting_power": sum(n.voting_power for n in node_registry.values()),
        "services": {
            "processing": "operational",
            "pir": "operational",
            "proofs": "operational",
            "blockchain": "operational" if blockchain_node else "offline",
        },
    }


# Helper functions


def _get_user_credits(user_id: str) -> int:
    """Get user credit balance."""
    # In production, would query blockchain state
    return 1000  # Mock balance


def _process_credit_redemption(user_id: str, invoice_id: str, amount: int) -> str:
    """Process credit redemption transaction."""
    # In production, would submit to blockchain
    tx_id = hashlib.sha256(
        f"{user_id}:{invoice_id}:{amount}:{datetime.now().isoformat()}".encode()
    ).hexdigest()
    return tx_id


def _queue_pipeline_job(job_id: str, request: PipelineRequest):
    """Queue processing pipeline job."""
    # In production, would add to job queue
    logger.info("Job %s queued for processing", job_id)


def _encode_to_hypervector(data: dict[str, Any], domain: str) -> Any:
    """Encode data to hypervector."""
    # In production, would use actual hypervector encoder
    return {"base": [0.1] * 10000, "mid": [0.2] * 15000, "high": [0.3] * 20000}


def _bind_hypervectors(v1: Any, v2: Any, method: str) -> Any:
    """Bind two hypervectors."""
    # In production, would use actual binding operations
    return {"bound_vector": [0.5] * 10000, "binding_method": method}


def _calculate_similarity(v1: Any, v2: Any, metric: str) -> float:
    """Calculate hypervector similarity."""
    # In production, would use actual similarity calculation
    return 0.85


# Startup and shutdown events


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global blockchain_node

    logger.info("Starting GenomeVault API...")

    # Initialize blockchain node
    node_type = (
        NodeType(config.node_type)
        if not isinstance(config.node_type, NodeType)
        else config.node_type
    )
    blockchain_node = BlockchainNode(
        node_id=config.node_id,
        node_type=node_type,
        signatory=config.signatory_status,
    )
    logger.info("Blockchain node started: %s (%s)", config.node_id, node_type.value)

    # Register some example nodes
    example_nodes = [
        NodeInfo(
            node_id="ln1",
            ip_address="10.0.0.1",
            node_type=NodeType.LIGHT.value,
            signatory=False,
            class_weight=1,
            stake=100,
            credits=100,
            last_seen=datetime.now().timestamp(),
        ),
        NodeInfo(
            node_id="ln2",
            ip_address="10.0.0.2",
            node_type=NodeType.LIGHT.value,
            signatory=False,
            class_weight=1,
            stake=100,
            credits=100,
            last_seen=datetime.now().timestamp(),
        ),
        NodeInfo(
            node_id="ts1",
            ip_address="10.0.0.3",
            node_type=NodeType.FULL.value,
            signatory=True,
            class_weight=4,
            stake=1000,
            credits=300,
            last_seen=datetime.now().timestamp(),
        ),
        NodeInfo(
            node_id="ts2",
            ip_address="10.0.0.4",
            node_type=NodeType.FULL.value,
            signatory=True,
            class_weight=4,
            stake=1000,
            credits=300,
            last_seen=datetime.now().timestamp(),
        ),
    ]

    for node in example_nodes:
        node_registry[node.node_id] = node
        blockchain_node.add_peer(node)

    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("API shutdown")


# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
