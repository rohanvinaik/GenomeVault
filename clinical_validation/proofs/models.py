"""
Data models for proofs and circuit configurations.
Single source of truth for proof-related data structures.
"""

from typing import Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json


class CircuitType(Enum):
    """Types of supported circuits"""

    DIABETES_RISK = "diabetes_risk"
    BIOMARKER_THRESHOLD = "biomarker_threshold"
    VARIANT_VERIFICATION = "variant_verification"
    POLYGENIC_RISK = "polygenic_risk"
    MULTI_OMICS = "multi_omics"


class ComparisonType(Enum):
    """Types of comparisons for biomarker circuits"""

    GREATER = "greater"
    LESS = "less"
    EQUAL = "equal"
    RANGE = "range"


@dataclass
class CircuitConfig:
    """Configuration for a ZK circuit"""

    name: str
    version: str = "1.0.0"
    constraints: int = 1000
    proof_size: int = 256
    supported_parameters: Dict[str, Any] = field(default_factory=dict)
    security_level: int = 128  # bits

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "circuit_name": self.name,
            "version": self.version,
            "constraints": self.constraints,
            "proof_size": self.proof_size,
            "supported_parameters": self.supported_parameters,
            "security_level": self.security_level,
        }


@dataclass
class ProofData:
    """Enhanced proof data structure with validation"""

    public_output: str
    proof_bytes: bytes
    verification_key: bytes
    circuit_type: CircuitType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate proof data"""
        if not self.proof_bytes:
            raise ValueError("Proof bytes cannot be empty")
        if not self.verification_key:
            raise ValueError("Verification key cannot be empty")

        # Add circuit type to metadata
        self.metadata["circuit_type"] = self.circuit_type.value

    def serialize(self) -> bytes:
        """Serialize proof to bytes with validation"""
        data = {
            "public_output": self.public_output,
            "proof_bytes": self.proof_bytes.hex(),
            "verification_key": self.verification_key.hex(),
            "circuit_type": self.circuit_type.value,
            "metadata": self.metadata,
        }
        return json.dumps(data, sort_keys=True).encode()

    @classmethod
    def deserialize(cls, data: bytes) -> "ProofData":
        """Deserialize proof from bytes"""
        obj = json.loads(data.decode())
        return cls(
            public_output=obj["public_output"],
            proof_bytes=bytes.fromhex(obj["proof_bytes"]),
            verification_key=bytes.fromhex(obj["verification_key"]),
            circuit_type=CircuitType(obj["circuit_type"]),
            metadata=obj.get("metadata", {}),
        )

    def verify_integrity(self) -> bool:
        """Verify proof integrity"""
        expected_hash = self.metadata.get("proof_hash")
        if not expected_hash:
            return True  # No hash to verify

        actual_hash = hashlib.sha256(self.proof_bytes).hexdigest()
        return actual_hash == expected_hash
