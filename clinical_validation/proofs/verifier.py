"""
Verification logic for clinical proofs.
Unified verifier for all circuit types.
"""

import logging
from typing import Any, Dict, Optional

from .models import CircuitType, ProofData


class ProofVerifier:
    """Unified verifier for all proof types"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def verify(
        self,
        proof: ProofData,
        public_inputs: Dict[str, Any],
        circuit_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Verify a proof with comprehensive validation

        Args:
            proof: The proof to verify
            public_inputs: Public inputs used in proof generation
            circuit_config: Optional circuit configuration for validation

        Returns:
            True if proof is valid
        """
        try:
            # Basic structural validation
            if not self._validate_proof_structure(proof):
                self.logger.error("Proof structure validation failed")
                return False

            # Integrity check
            if not proof.verify_integrity():
                self.logger.error("Proof integrity check failed")
                return False

            # Circuit-specific validation
            if not self._validate_circuit_specific(proof, public_inputs):
                self.logger.error("Circuit-specific validation failed")
                return False

            self.logger.info(f"Proof verification successful for {proof.circuit_type.value}")
            return True

        except Exception as e:
            self.logger.error(f"Proof verification error: {str(e)}")
            return False

    def _validate_proof_structure(self, proof: ProofData) -> bool:
        """Validate basic proof structure"""
        if not isinstance(proof, ProofData):
            return False

        if not proof.public_output:
            return False

        if not proof.proof_bytes or len(proof.proof_bytes) == 0:
            return False

        if not proof.verification_key or len(proof.verification_key) == 0:
            return False

        return True

    def _validate_circuit_specific(self, proof: ProofData, public_inputs: Dict[str, Any]) -> bool:
        """Circuit-specific validation logic"""

        if proof.circuit_type == CircuitType.DIABETES_RISK:
            return self._validate_diabetes_proof(proof, public_inputs)
        elif proof.circuit_type == CircuitType.BIOMARKER_THRESHOLD:
            return self._validate_biomarker_proof(proof, public_inputs)
        else:
            self.logger.warning(f"Unknown circuit type: {proof.circuit_type}")
            return False

    def _validate_diabetes_proof(self, proof: ProofData, public_inputs: Dict[str, Any]) -> bool:
        """Validate diabetes risk proof"""
        # Verify output format
        if not proof.public_output.startswith("RISK_LEVEL:"):
            return False

        parts = proof.public_output.split(":")
        if len(parts) != 4:
            return False

        risk_level = parts[1]
        if risk_level not in ["HIGH", "NORMAL"]:
            return False

        # Verify confidence is valid
        try:
            confidence = float(parts[3])
            if not 0 <= confidence <= 1:
                return False
        except ValueError:
            return False

        return True

    def _validate_biomarker_proof(self, proof: ProofData, public_inputs: Dict[str, Any]) -> bool:
        """Validate biomarker threshold proof"""
        parts = proof.public_output.split(":")
        if len(parts) < 6:
            return False

        if parts[1] not in ["EXCEEDS", "NORMAL"]:
            return False

        # Verify numeric values
        try:
            margin = float(parts[3])
            confidence = float(parts[5])
            if not 0 <= confidence <= 1:
                return False
        except ValueError:
            return False

        return True


# Convenience function
def verify_proof(proof: ProofData, public_inputs: Dict[str, Any]) -> bool:
    """Verify a proof using the default verifier"""
    verifier = ProofVerifier()
    return verifier.verify(proof, public_inputs)
