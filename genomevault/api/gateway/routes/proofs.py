"""
Zero-knowledge proof routes for GenomeVault API Gateway.
"""

from __future__ import annotations

from fastapi import APIRouter
from genomevault.api.gateway.models.proofs import *

router = APIRouter()


@router.post("/generate", response_model=ProofGenerationResponse)
async def generate_proof(request: ProofGenerationRequest) -> ProofGenerationResponse:
    """Generate a zero-knowledge proof."""
    # TODO: Implement proof generation logic
    pass


@router.post("/verify", response_model=ProofVerificationResponse)
async def verify_proof(request: ProofVerificationRequest) -> ProofVerificationResponse:
    """Verify a zero-knowledge proof."""
    # TODO: Implement proof verification logic
    pass


@router.get("/list", response_model=ProofListResponse)
async def list_proofs(request: ProofListRequest) -> ProofListResponse:
    """List available proofs."""
    # TODO: Implement proof listing logic
    pass
