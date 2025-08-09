from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from genomevault.zk.engine import ZKProofEngine
from genomevault.zk.models import ProofCreateRequest, ProofVerifyRequest

router = APIRouter(prefix="/proofs", tags=["proofs"])

_engine = ZKProofEngine()  # Will use default PROJECT_ROOT


@router.post("/create")
def create_proof(request: ProofCreateRequest) -> Dict[str, Any]:
    try:
        proof = _engine.create_proof(circuit_type=request.circuit_type, inputs=request.inputs)
        # proof is a shim object with to_base64()
        return {
            "proof": proof.to_base64(),
            "public_inputs": getattr(proof, "public_inputs", {}),
            "circuit_type": request.circuit_type,
        }
    except Exception as e:
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=400, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.post("/verify")
def verify_proof(request: ProofVerifyRequest) -> Dict[str, bool]:
    try:
        ok = _engine.verify_proof(proof_data=request.proof, public_inputs=request.public_inputs)
        return {"valid": bool(ok)}
    except Exception as e:
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=400, detail=str(e))
        raise RuntimeError("Unspecified error")
