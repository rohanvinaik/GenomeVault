"""Proofs module."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from typing import Any

from genomevault.utils.logging import get_logger
from genomevault.zk.engine import ZKProofEngine
from genomevault.zk.models import ProofCreateRequest, ProofVerifyRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/proofs", tags=["proofs"])

_engine = ZKProofEngine()  # Will use default PROJECT_ROOT


@router.post("/create", status_code=status.HTTP_200_OK)
def create_proof(request: ProofCreateRequest) -> Any:
    """Create proof.

    Args:
        request: Client request.

    Returns:
        Newly created proof. Returns 200 even with fallback for graceful degradation.
        The response includes metadata indicating whether a real ZK proof or
        transcript fallback was used.

    Raises:
        HTTPException: When operation fails completely.
    """
    try:
        proof = _engine.create_proof(circuit_type=request.circuit_type, inputs=request.inputs)

        # Check if we got a fallback transcript
        proof_data = proof.to_base64() if hasattr(proof, "to_base64") else str(proof)
        public_inputs = getattr(proof, "public_inputs", {})

        # Try to detect if this is a transcript fallback
        is_fallback = False
        engine_type = "zk_proof"

        # Check if the proof contains transcript metadata
        if hasattr(proof, "proof") and isinstance(proof.proof, dict):
            if proof.proof.get("engine") == "transcript":
                is_fallback = True
                engine_type = "transcript"
                logger.info(f"Using transcript fallback for circuit type: {request.circuit_type}")

        response = {
            "proof": proof_data,
            "public_inputs": public_inputs,
            "circuit_type": request.circuit_type,
            "metadata": {
                "engine": engine_type,
                "fallback_used": is_fallback,
                "status": "success",
            },
        }

        # Log if fallback was used
        if is_fallback:
            logger.info(
                f"Proof created using transcript fallback for circuit: {request.circuit_type}"
            )
        else:
            logger.debug(f"Proof created using ZK engine for circuit: {request.circuit_type}")

        return response

    except ValueError as e:
        # Invalid input or unsupported circuit type
        logger.warning(f"Invalid proof request: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # Unexpected error - log but still try to provide useful response
        logger.error(f"Error creating proof: {e}", exc_info=True)

        # Even on error, try to provide a degraded response if possible
        return {
            "proof": None,
            "public_inputs": {},
            "circuit_type": request.circuit_type,
            "metadata": {
                "engine": "error",
                "fallback_used": True,
                "status": "error",
                "error_message": str(e),
            },
        }


@router.post("/verify", status_code=status.HTTP_200_OK)
def verify_proof(request: ProofVerifyRequest) -> Any:
    """Verify proof.

    Args:
        request: Client request.

    Returns:
        Verification result with metadata. Returns 200 even for transcript proofs.

    Raises:
        HTTPException: When request is malformed.
    """
    try:
        # Attempt verification
        ok = _engine.verify_proof(proof_data=request.proof, public_inputs=request.public_inputs)

        # Try to detect proof type from the proof data
        engine_type = "unknown"
        if isinstance(request.proof, str):
            # Try to decode and check
            try:
                import base64
                import json

                decoded = base64.b64decode(request.proof)
                proof_obj = json.loads(decoded)
                if proof_obj.get("engine") == "transcript":
                    engine_type = "transcript"
                elif "_metadata" in proof_obj:
                    engine_type = proof_obj["_metadata"].get("engine", "zk_proof")
                else:
                    engine_type = "zk_proof"
            except Exception:
                engine_type = "zk_proof"

        response = {
            "valid": bool(ok),
            "metadata": {
                "engine": engine_type,
                "verification_method": "signature" if engine_type == "transcript" else "zk_proof",
            },
        }

        if ok:
            logger.info(f"Proof verified successfully using {engine_type}")
        else:
            logger.warning(f"Proof verification failed using {engine_type}")

        return response

    except ValueError as e:
        logger.warning(f"Invalid verification request: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # Log error but still return a response
        logger.error(f"Error verifying proof: {e}", exc_info=True)

        return {
            "valid": False,
            "metadata": {
                "engine": "error",
                "verification_method": "none",
                "error_message": str(e),
            },
        }
