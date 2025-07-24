"""
GenomeVault ZK Proof API Integration
Provides REST API endpoints and service integration for the ZK proof system.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# Import our ZK system
from genomevault_zk_integration import (
    CircuitType,
    GenomeVaultZKSystem,
    ZKProofRequest,
    ZKProofResult,
)


# Pydantic models for API
class VariantData(BaseModel):
    chr: str = Field(..., description="Chromosome")
    pos: int = Field(..., description="Position")
    ref: str = Field(..., description="Reference allele")
    alt: str = Field(..., description="Alternate allele")


class MerkleProof(BaseModel):
    path: List[str] = Field(..., description="Merkle path hashes")
    indices: List[int] = Field(..., description="Path directions (0=left, 1=right)")


class VariantProofRequest(BaseModel):
    variant_data: VariantData
    merkle_proof: MerkleProof
    commitment_root: str = Field(..., description="Genome commitment root")


class PRSProofRequest(BaseModel):
    variants: List[int] = Field(..., description="Genotype values (0,1,2)")
    weights: List[float] = Field(..., description="PRS weights")
    score_range: Dict[str, float] = Field(..., description="Valid score range")


class DiabetesRiskRequest(BaseModel):
    glucose_reading: float = Field(..., description="Glucose level in mg/dL")
    risk_score: float = Field(..., description="Genetic risk score (0-1)")
    glucose_threshold: float = Field(126.0, description="Glucose threshold")
    risk_threshold: float = Field(0.75, description="Risk score threshold")


class ProofResponse(BaseModel):
    success: bool
    proof_id: Optional[str] = None
    transaction_id: Optional[str] = None
    proof_size_bytes: Optional[int] = None
    generation_time: Optional[float] = None
    verification_time: Optional[float] = None
    error_message: Optional[str] = None


class SystemHealthResponse(BaseModel):
    status: str
    metrics: Dict[str, Any]
    cache_size: int
    queue_size: int
    uptime_seconds: float


# Authentication (simplified for demo)
security = HTTPBearer()


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token (simplified implementation)"""
    # In production, implement proper JWT verification
    if credentials.credentials != "demo_token":
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials


# FastAPI app setup
app = FastAPI(
    title="GenomeVault ZK Proof API",
    description="Zero-Knowledge Proof generation and verification for genomic data",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ZK system instance
zk_system: Optional[GenomeVaultZKSystem] = None
start_time = datetime.now()


@app.on_event("startup")
async def startup_event():
    """Initialize ZK system on startup"""
    global zk_system
    zk_system = GenomeVaultZKSystem(max_workers=4, cache_size=1000)
    await zk_system.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup ZK system on shutdown"""
    global zk_system
    if zk_system:
        await zk_system.stop()


# API Endpoints


@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """Get system health and metrics"""
    if not zk_system:
        raise HTTPException(status_code=503, detail="ZK system not initialized")

    health = await zk_system.health_check()
    uptime = (datetime.now() - start_time).total_seconds()

    return SystemHealthResponse(
        status=health["status"],
        metrics=health["metrics"],
        cache_size=health["cache_size"],
        queue_size=health["queue_size"],
        uptime_seconds=uptime,
    )


@app.post("/proofs/variant", response_model=ProofResponse)
async def generate_variant_proof(
    request: VariantProofRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token),
):
    """Generate zero-knowledge proof for variant presence"""
    if not zk_system:
        raise HTTPException(status_code=503, detail="ZK system not initialized")

    try:
        result = await zk_system.zk_integration.prove_variant_presence(
            variant_data=request.variant_data.dict(),
            merkle_proof=request.merkle_proof.dict(),
            commitment_root=request.commitment_root,
        )

        if result.success:
            proof_id = str(uuid.uuid4())

            return ProofResponse(
                success=True,
                proof_id=proof_id,
                proof_size_bytes=len(result.proof.proof_bytes),
                generation_time=result.generation_time,
                verification_time=result.verification_time,
            )
        else:
            return ProofResponse(success=False, error_message=result.error_message)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proof generation failed: {str(e)}")


@app.post("/proofs/prs", response_model=ProofResponse)
async def generate_prs_proof(
    request: PRSProofRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token),
):
    """Generate zero-knowledge proof for polygenic risk score"""
    if not zk_system:
        raise HTTPException(status_code=503, detail="ZK system not initialized")

    try:
        result = await zk_system.zk_integration.prove_polygenic_risk_score(
            variants=request.variants,
            weights=request.weights,
            score_range=request.score_range,
        )

        if result.success:
            proof_id = str(uuid.uuid4())

            return ProofResponse(
                success=True,
                proof_id=proof_id,
                proof_size_bytes=len(result.proof.proof_bytes),
                generation_time=result.generation_time,
                verification_time=result.verification_time,
            )
        else:
            return ProofResponse(success=False, error_message=result.error_message)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PRS proof generation failed: {str(e)}")


@app.post("/proofs/diabetes-risk", response_model=ProofResponse)
async def generate_diabetes_risk_proof(
    request: DiabetesRiskRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token),
):
    """Generate zero-knowledge proof for diabetes risk assessment"""
    if not zk_system:
        raise HTTPException(status_code=503, detail="ZK system not initialized")

    try:
        result = await zk_system.zk_integration.prove_diabetes_risk_alert(
            glucose_reading=request.glucose_reading,
            risk_score=request.risk_score,
            glucose_threshold=request.glucose_threshold,
            risk_threshold=request.risk_threshold,
        )

        if result.success:
            proof_id = str(uuid.uuid4())

            return ProofResponse(
                success=True,
                proof_id=proof_id,
                proof_size_bytes=len(result.proof.proof_bytes),
                generation_time=result.generation_time,
                verification_time=result.verification_time,
            )
        else:
            return ProofResponse(success=False, error_message=result.error_message)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diabetes risk proof failed: {str(e)}")


@app.get("/circuits")
async def list_available_circuits():
    """List available ZK circuits"""
    return {
        "circuits": [
            {
                "type": "variant_verification",
                "description": "Prove variant presence without revealing position",
                "constraints": "~5,000",
                "proof_size": "192 bytes",
                "verification_time": "<10ms",
            },
            {
                "type": "polygenic_risk_score",
                "description": "Prove PRS calculation without revealing variants",
                "constraints": "~20,000",
                "proof_size": "384 bytes",
                "verification_time": "<25ms",
            },
            {
                "type": "diabetes_risk_alert",
                "description": "Prove risk condition without revealing values",
                "constraints": "~15,000",
                "proof_size": "384 bytes",
                "verification_time": "<25ms",
            },
        ]
    }


@app.get("/metrics")
async def get_detailed_metrics(token: str = Depends(verify_token)):
    """Get detailed system metrics"""
    if not zk_system:
        raise HTTPException(status_code=503, detail="ZK system not initialized")

    metrics = zk_system.proof_service.get_metrics()

    return {
        "metrics": metrics,
        "system_info": {
            "max_workers": zk_system.proof_service.max_workers,
            "cache_size": len(zk_system.proof_service.cache),
            "uptime_seconds": (datetime.now() - start_time).total_seconds(),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
