"""High-level ZK proof service API."""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from genomevault.zk_proofs.prover import ZKProver
from genomevault.zk_proofs.verifier import ZKVerifier
from genomevault.zk_proofs.circuits import PRSProofCircuit
from genomevault.utils.logging import logger


class ProofRequest:
    """Request for proof generation."""

    def __init__(
        self, proof_type: str, private_inputs: Dict[str, Any], public_inputs: Dict[str, Any]
    ):
        self.proof_type = proof_type
        self.private_inputs = private_inputs
        self.public_inputs = public_inputs


class ProofResponse:
    """Response containing generated proof."""

    def __init__(self, proof: bytes, proof_id: str, verification_key: str, generated_at: datetime):
        self.proof = proof
        self.proof_id = proof_id
        self.verification_key = verification_key
        self.generated_at = generated_at


class VerificationResult:
    """Result of proof verification."""

    def __init__(
        self, is_valid: bool, proof_id: Optional[str] = None, verifier_id: Optional[str] = None
    ):
        self.is_valid = is_valid
        self.proof_id = proof_id
        self.verifier_id = verifier_id


class ZKProofService:
    """High-level service for ZK proof operations."""

    def __init__(self):
        self.circuits = {
            "prs_range": PRSProofCircuit(),
            "age_range": PRSProofCircuit(),  # Reuse for demo
            "variant_count": PRSProofCircuit(),  # Reuse for demo
        }
        self.provers = {}
        self.verifiers = {}

        # Initialize provers and verifiers for each circuit
        for name, circuit in self.circuits.items():
            self.provers[name] = ZKProver(circuit)
            self.verifiers[name] = ZKVerifier(circuit)

    async def generate_proof(self, request: ProofRequest) -> ProofResponse:
        """Generate a ZK proof based on the request."""
        logger.info(f"Generating {request.proof_type} proof")

        if request.proof_type not in self.provers:
            raise ValueError(f"Unknown proof type: {request.proof_type}")

        prover = self.provers[request.proof_type]

        # Generate proof based on type
        if request.proof_type == "prs_range":
            proof = await self._generate_prs_proof(prover, request)
        elif request.proof_type == "age_range":
            proof = await self._generate_age_proof(prover, request)
        elif request.proof_type == "variant_count":
            proof = await self._generate_variant_proof(prover, request)
        else:
            raise ValueError(f"Unsupported proof type: {request.proof_type}")

        # Create response
        proof_id = str(uuid.uuid4())
        verification_key = self._get_verification_key(request.proof_type)

        return ProofResponse(
            proof=proof.serialize(),
            proof_id=proof_id,
            verification_key=verification_key,
            generated_at=datetime.utcnow(),
        )

    async def verify_proof(
        self,
        proof: bytes,
        verification_key: str,
        public_inputs: Dict[str, Any],
        verifier_id: Optional[str] = None,
    ) -> VerificationResult:
        """Verify a ZK proof."""
        logger.info(f"Verifying proof{f' for {verifier_id}' if verifier_id else ''}")

        # Determine proof type from verification key
        proof_type = self._get_proof_type_from_key(verification_key)

        if proof_type not in self.verifiers:
            raise ValueError(f"Unknown proof type for key: {verification_key}")

        verifier = self.verifiers[proof_type]
        circuit = self.circuits[proof_type]

        # Deserialize proof
        proof_obj = circuit.deserialize_proof(proof)

        # Verify
        is_valid = verifier.verify(proof_obj, public_inputs)

        return VerificationResult(
            is_valid=is_valid,
            proof_id=None,  # Could extract from proof metadata
            verifier_id=verifier_id,
        )

    async def _generate_prs_proof(self, prover: ZKProver, request: ProofRequest) -> Any:
        """Generate PRS range proof."""
        prs_score = request.private_inputs.get("prs_score")
        min_val = request.public_inputs.get("min", 0.0)
        max_val = request.public_inputs.get("max", 1.0)

        if prs_score is None:
            raise ValueError("Missing prs_score in private inputs")

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        proof = await loop.run_in_executor(
            None, prover.prove_prs_in_range, prs_score, min_val, max_val
        )

        return proof

    async def _generate_age_proof(self, prover: ZKProver, request: ProofRequest) -> Any:
        """Generate age range proof."""
        # For demo, reuse PRS proof logic
        age = request.private_inputs.get("age")
        min_age = request.public_inputs.get("min", 0)
        max_age = request.public_inputs.get("max", 150)

        if age is None:
            raise ValueError("Missing age in private inputs")

        # Normalize to [0, 1] range
        normalized = (age - min_age) / (max_age - min_age)

        loop = asyncio.get_event_loop()
        proof = await loop.run_in_executor(None, prover.prove_prs_in_range, normalized, 0.0, 1.0)

        return proof

    async def _generate_variant_proof(self, prover: ZKProver, request: ProofRequest) -> Any:
        """Generate variant count proof."""
        # For demo, reuse PRS proof logic
        count = request.private_inputs.get("count", 0)
        max_count = request.public_inputs.get("max", 100)

        # Normalize to [0, 1] range
        normalized = count / max_count

        loop = asyncio.get_event_loop()
        proof = await loop.run_in_executor(None, prover.prove_prs_in_range, normalized, 0.0, 1.0)

        return proof

    def _get_verification_key(self, proof_type: str) -> str:
        """Get verification key for proof type."""
        # In production, these would be actual verification keys
        return f"vk_{proof_type}_v1"

    def _get_proof_type_from_key(self, verification_key: str) -> str:
        """Extract proof type from verification key."""
        # Simple parsing for demo
        if verification_key.startswith("vk_"):
            parts = verification_key.split("_")
            if len(parts) >= 3:
                return "_".join(parts[1:-1])

        raise ValueError(f"Invalid verification key format: {verification_key}")

    async def aggregate_proofs(self, proofs: List[ProofResponse]) -> Any:
        """Aggregate multiple proofs (placeholder for future implementation)."""
        # This would implement proof aggregation/batching
        logger.info(f"Aggregating {len(proofs)} proofs")

        # For now, return a simple aggregation result
        return {
            "is_valid": True,
            "sub_proofs": [p.proof_id for p in proofs],
            "aggregated_at": datetime.utcnow(),
        }
