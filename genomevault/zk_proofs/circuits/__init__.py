"""Package initialization for circuits."""

import hashlib
import json
import os
from datetime import datetime
from typing import Any

from .base_circuits import BaseCircuit
from .clinical_circuits import ClinicalCircuit

"""ZK proof circuits for genomic applications."""


class PRSProofCircuit(BaseCircuit):
    """Circuit for proving PRS (Polygenic Risk Score) is within a range."""

    def __init__(self):
        """Initialize instance."""
        super().__init__()
        self.name = "prs_range_proof"
        self.proof_size = 384  # Groth16 proof size

    def prove_prs_in_range(self, prs: float, min_val: float, max_val: float) -> Any:
        """Generate proof that PRS is in [min_val, max_val]."""
        # Validate inputs
        if not min_val <= prs <= max_val:
            raise ValueError(f"PRS {prs} not in range [{min_val}, {max_val}]")

        # In production, this would create actual ZK proof
        # For now, create a mock proof
        return self._create_mock_proof(
            {
                "statement": "prs_in_range",
                "public": {"min": min_val, "max": max_val},
                "private": {"prs": prs},
            }
        )

    def serialize_proof(self, proof: Any) -> bytes:
        """Serialize proof to bytes."""
        # In production, this would properly serialize the proof

        # Create deterministic but randomized proof
        proof_data = json.dumps(proof, sort_keys=True)
        proof_hash = hashlib.sha256(proof_data.encode()).digest()

        # Add randomness

        random_bytes = os.urandom(self.proof_size - 32)

        return proof_hash + random_bytes

    def deserialize_proof(self, data: bytes) -> Any:
        """Deserialize proof from bytes."""
        if len(data) != self.proof_size:
            raise ValueError(f"Invalid proof size: {len(data)} != {self.proof_size}")

        # For mock, just return a proof object
        return {"data": data, "size": len(data), "valid_format": True}

    def _create_mock_proof(self, data: dict) -> Any:
        """Create a mock proof object."""
        return {
            "circuit": self.name,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "is_valid": lambda: True,
            "serialize": lambda: self.serialize_proof(data),
        }


# Re-export for convenience
__all__ = ["BaseCircuit", "ClinicalCircuit", "PRSProofCircuit"]
