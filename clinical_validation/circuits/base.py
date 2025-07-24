"""
Base circuit implementation for all ZK circuits in GenomeVault.
Single source of truth for circuit architecture.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np

from ..proofs.models import CircuitConfig, ProofData


@runtime_checkable
class Circuit(Protocol):
    """Protocol for ZK circuits"""

    def setup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Setup circuit parameters"""
        ...

    def generate_witness(self, private_inputs: Dict, public_inputs: Dict) -> Dict:
        """Generate witness for proof"""
        ...

    def prove(self, witness: Dict, public_inputs: Dict) -> ProofData:
        """Generate proof"""
        ...

    def verify(self, proof: ProofData, public_inputs: Dict) -> bool:
        """Verify proof"""
        ...


class BaseCircuit(ABC):
    """Enhanced base class for ZK circuits with common functionality"""

    def __init__(self, config: CircuitConfig):
        """Magic method implementation."""
        self.config = config
        self._setup_complete = False
        self._verification_keys: Dict[str, bytes] = {}

    def setup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Setup circuit parameters"""
        # Validate parameters
        for param, value in params.items():
            if param in self.config.supported_parameters:
                expected_type = self.config.supported_parameters[param]
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Parameter {param} must be of type {expected_type}"
                    )

        self._setup_complete = True
        return self.config.to_dict()

    @abstractmethod
    def generate_witness(self, private_inputs: Dict, public_inputs: Dict) -> Dict:
        """Generate witness for proof - must be implemented by subclasses"""

    @abstractmethod
    def prove(self, witness: Dict, public_inputs: Dict) -> ProofData:
        """Generate proof - must be implemented by subclasses"""

    def verify(self, proof: ProofData, public_inputs: Dict) -> bool:
        """Common verification logic"""
        # Basic validation
        if not isinstance(proof, ProofData):
            return False

        if not proof.verify_integrity():
            return False

        # Verify public inputs match
        expected_hash = self._hash_inputs(public_inputs)
        if proof.metadata.get("public_inputs_hash") != expected_hash:
            return False

        # Delegate to specific verification
        return self._verify_proof_specific(proof, public_inputs)

    @abstractmethod
    def _verify_proof_specific(self, proof: ProofData, public_inputs: Dict) -> bool:
        """Circuit-specific verification logic"""

    def _generate_verification_key(self) -> bytes:
        """Generate verification key for the circuit"""
        # In production, this would be from trusted setup
        key_data = (
            f"{self.config.name}:{self.config.version}:{self.config.security_level}"
        )
        return hashlib.sha256(key_data.encode()).digest() + np.random.bytes(32)

    def _hash_inputs(self, inputs: Dict[str, Any]) -> str:
        """Hash public inputs for verification"""
        # Sort keys for deterministic hashing
        sorted_data = json.dumps(inputs, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()

    def _add_proof_metadata(
        self, proof: ProofData, public_inputs: Dict, witness: Dict
    ) -> None:
        """Add standard metadata to proof"""
        proof.metadata.update(
            {
                "circuit": self.config.name,
                "version": self.config.version,
                "timestamp": np.datetime64("now").astype(str),
                "public_inputs_hash": self._hash_inputs(public_inputs),
                "proof_hash": hashlib.sha256(proof.proof_bytes).hexdigest(),
                "constraints": self.config.constraints,
                "security_level": self.config.security_level,
            }
        )
