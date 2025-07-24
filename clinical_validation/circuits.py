"""
Refactored clinical circuits module for GenomeVault.
This consolidates the duplicate implementations and provides a clean architecture.
"""

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import numpy as np
import hashlib
from dataclasses import dataclass, field
from enum import Enum
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
        pass

    @abstractmethod
    def prove(self, witness: Dict, public_inputs: Dict) -> ProofData:
        """Generate proof - must be implemented by subclasses"""
        pass

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
        pass

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


class DiabetesRiskCircuit(BaseCircuit):
    """
    Optimized diabetes risk assessment circuit.
    Proves clinical values exceed thresholds without revealing actual values.
    """

    def __init__(self):
        config = CircuitConfig(
            name="DiabetesRiskCircuit",
            version="2.0.0",
            constraints=15000,
            proof_size=384,
            supported_parameters={
                "glucose_range": tuple,
                "hba1c_range": tuple,
                "risk_score_range": tuple,
                "risk_factors_threshold": int,
            },
        )
        super().__init__(config)

        # Default ranges
        self.glucose_range = (70.0, 300.0)
        self.hba1c_range = (4.0, 14.0)
        self.risk_score_range = (-3.0, 3.0)
        self.risk_factors_threshold = 2

    def setup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Setup with validation"""
        result = super().setup(params)

        # Update ranges if provided
        if "glucose_range" in params:
            self.glucose_range = params["glucose_range"]
        if "hba1c_range" in params:
            self.hba1c_range = params["hba1c_range"]
        if "risk_score_range" in params:
            self.risk_score_range = params["risk_score_range"]
        if "risk_factors_threshold" in params:
            self.risk_factors_threshold = params["risk_factors_threshold"]

        result["supported_thresholds"] = {
            "glucose": self.glucose_range,
            "hba1c": self.hba1c_range,
            "risk_score": self.risk_score_range,
        }

        return result

    def generate_witness(
        self, private_inputs: Dict[str, float], public_inputs: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate witness with enhanced validation"""
        # Validate inputs are within ranges
        glucose = private_inputs.get("glucose", 100.0)
        if not self.glucose_range[0] <= glucose <= self.glucose_range[1]:
            raise ValueError(f"Glucose {glucose} out of range {self.glucose_range}")

        hba1c = private_inputs.get("hba1c", 5.5)
        if not self.hba1c_range[0] <= hba1c <= self.hba1c_range[1]:
            raise ValueError(f"HbA1c {hba1c} out of range {self.hba1c_range}")

        risk_score = private_inputs.get("genetic_risk_score", 0.0)
        if not self.risk_score_range[0] <= risk_score <= self.risk_score_range[1]:
            raise ValueError(
                f"Risk score {risk_score} out of range {self.risk_score_range}"
            )

        # Extract thresholds
        glucose_threshold = public_inputs.get("glucose_threshold", 126.0)
        hba1c_threshold = public_inputs.get("hba1c_threshold", 6.5)
        risk_threshold = public_inputs.get("risk_threshold", 0.5)

        # Compute comparisons
        glucose_exceeds = float(glucose > glucose_threshold)
        hba1c_exceeds = float(hba1c > hba1c_threshold)
        risk_exceeds = float(risk_score > risk_threshold)

        # Risk assessment
        risk_factors = glucose_exceeds + hba1c_exceeds + risk_exceeds
        is_high_risk = risk_factors >= self.risk_factors_threshold

        # Enhanced witness with noise for differential privacy
        noise_factor = np.random.normal(0, 0.001, size=3)

        return {
            "private_values": {
                "glucose": glucose + noise_factor[0],
                "hba1c": hba1c + noise_factor[1],
                "risk_score": risk_score + noise_factor[2],
            },
            "comparisons": {
                "glucose_exceeds": glucose_exceeds,
                "hba1c_exceeds": hba1c_exceeds,
                "risk_exceeds": risk_exceeds,
            },
            "result": {
                "risk_factors": int(risk_factors),
                "is_high_risk": is_high_risk,
                "confidence": min(0.99, risk_factors / 3.0),  # Confidence score
            },
            "randomness": np.random.bytes(32),
            "noise_factor": noise_factor.tolist(),
        }

    def prove(
        self, witness: Dict[str, Any], public_inputs: Dict[str, float]
    ) -> ProofData:
        """Generate proof with enhanced security"""
        if not self._setup_complete:
            raise RuntimeError("Circuit setup not complete")

        is_high_risk = witness["result"]["is_high_risk"]
        confidence = witness["result"]["confidence"]

        # Create proof with detailed output
        public_output = f"RISK_LEVEL:{'HIGH' if is_high_risk else 'NORMAL'}:CONFIDENCE:{confidence:.2f}"

        # Generate proof bytes (in production, use actual ZK proof system)
        proof_data = {
            "witness_commitment": hashlib.sha256(str(witness).encode()).hexdigest(),
            "public_inputs": public_inputs,
            "risk_assessment": witness["result"],
        }

        proof_bytes = hashlib.sha256(json.dumps(proof_data).encode()).digest()
        proof_bytes += np.random.bytes(self.config.proof_size - 32)

        # Create proof object
        proof = ProofData(
            public_output=public_output,
            proof_bytes=proof_bytes,
            verification_key=self._generate_verification_key(),
            circuit_type=CircuitType.DIABETES_RISK,
        )

        # Add metadata
        self._add_proof_metadata(proof, public_inputs, witness)
        proof.metadata["risk_factors_used"] = self.risk_factors_threshold
        proof.metadata["confidence_score"] = confidence

        return proof

    def _verify_proof_specific(self, proof: ProofData, public_inputs: Dict) -> bool:
        """Specific verification for diabetes risk circuit"""
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


class ClinicalBiomarkerCircuit(BaseCircuit):
    """
    Generic circuit for clinical biomarker threshold proofs.
    Supports multiple comparison types and multi-threshold checks.
    """

    def __init__(self, biomarker_name: str = "generic"):
        config = CircuitConfig(
            name=f"ClinicalBiomarker_{biomarker_name}_Circuit",
            version="2.0.0",
            constraints=5000,
            proof_size=256,
            supported_parameters={
                "value_range": tuple,
                "comparison_types": list,
                "precision": float,
            },
        )
        super().__init__(config)

        self.biomarker_name = biomarker_name
        self.value_range = (0.0, 1000.0)
        self.precision = 0.001
        self.supported_comparisons = [
            ComparisonType.GREATER,
            ComparisonType.LESS,
            ComparisonType.EQUAL,
        ]

    def setup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Setup with biomarker-specific configuration"""
        result = super().setup(params)

        if "value_range" in params:
            self.value_range = params["value_range"]
        if "precision" in params:
            self.precision = params["precision"]
        if "comparison_types" in params:
            self.supported_comparisons = [
                ComparisonType(ct) for ct in params["comparison_types"]
            ]

        result["biomarker_config"] = {
            "name": self.biomarker_name,
            "value_range": self.value_range,
            "precision": self.precision,
            "supported_comparisons": [ct.value for ct in self.supported_comparisons],
        }

        return result

    def generate_witness(
        self, private_inputs: Dict[str, float], public_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate witness for biomarker comparison"""
        value = private_inputs.get("value", 0.0)

        # Validate value is in range
        if not self.value_range[0] <= value <= self.value_range[1]:
            raise ValueError(f"Value {value} out of range {self.value_range}")

        threshold = public_inputs.get("threshold", 0.0)
        comparison_type = ComparisonType(public_inputs.get("comparison", "greater"))

        if comparison_type not in self.supported_comparisons:
            raise ValueError(f"Comparison type {comparison_type} not supported")

        # Perform comparison
        if comparison_type == ComparisonType.GREATER:
            result = value > threshold
        elif comparison_type == ComparisonType.LESS:
            result = value < threshold
        elif comparison_type == ComparisonType.EQUAL:
            result = abs(value - threshold) < self.precision
        else:
            result = False

        # Support for range comparisons
        if (
            comparison_type == ComparisonType.RANGE
            and "threshold_high" in public_inputs
        ):
            threshold_high = public_inputs["threshold_high"]
            result = threshold <= value <= threshold_high

        # Add noise for privacy
        noise = np.random.normal(0, self.precision / 10)

        return {
            "private_value": value + noise,
            "public_threshold": threshold,
            "comparison_type": comparison_type.value,
            "result": result,
            "margin": abs(value - threshold),  # How far from threshold
            "randomness": np.random.bytes(32),
            "noise": noise,
        }

    def prove(
        self, witness: Dict[str, Any], public_inputs: Dict[str, float]
    ) -> ProofData:
        """Generate proof for biomarker threshold"""
        if not self._setup_complete:
            raise RuntimeError("Circuit setup not complete")

        result = witness["result"]
        margin = witness["margin"]

        # Create detailed output
        status = "EXCEEDS" if result else "NORMAL"
        confidence = min(0.99, margin / abs(witness["public_threshold"] + 0.01))

        public_output = f"{self.biomarker_name}:{status}:MARGIN:{margin:.3f}:CONFIDENCE:{confidence:.2f}"

        # Generate proof
        proof_bytes = hashlib.sha256(str(witness).encode()).digest()
        proof_bytes += np.random.bytes(self.config.proof_size - 32)

        proof = ProofData(
            public_output=public_output,
            proof_bytes=proof_bytes,
            verification_key=self._generate_verification_key(),
            circuit_type=CircuitType.BIOMARKER_THRESHOLD,
        )

        self._add_proof_metadata(proof, public_inputs, witness)
        proof.metadata["biomarker"] = self.biomarker_name
        proof.metadata["comparison_type"] = witness["comparison_type"]

        return proof

    def _verify_proof_specific(self, proof: ProofData, public_inputs: Dict) -> bool:
        """Specific verification for biomarker circuit"""
        # Verify output format
        parts = proof.public_output.split(":")
        if len(parts) < 6:
            return False

        if parts[0] != self.biomarker_name:
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


# Factory function for creating circuits
def create_circuit(circuit_type: CircuitType, **kwargs) -> BaseCircuit:
    """Factory function to create appropriate circuit instances"""
    if circuit_type == CircuitType.DIABETES_RISK:
        return DiabetesRiskCircuit()
    elif circuit_type == CircuitType.BIOMARKER_THRESHOLD:
        biomarker_name = kwargs.get("biomarker_name", "generic")
        return ClinicalBiomarkerCircuit(biomarker_name)
    else:
        raise ValueError(f"Unsupported circuit type: {circuit_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test diabetes risk circuit
    diabetes_circuit = DiabetesRiskCircuit()

    # Setup
    setup_params = {
        "glucose_range": (70, 300),
        "hba1c_range": (4, 14),
        "risk_factors_threshold": 2,
    }

    config = diabetes_circuit.setup(setup_params)
    print(f"Circuit setup: {config}")

    # Generate witness
    private_inputs = {"glucose": 130, "hba1c": 7.0, "genetic_risk_score": 1.2}

    public_inputs = {
        "glucose_threshold": 126,
        "hba1c_threshold": 6.5,
        "risk_threshold": 1.0,
    }

    witness = diabetes_circuit.generate_witness(private_inputs, public_inputs)
    print(f"\nWitness generated: {witness['result']}")

    # Generate proof
    proof = diabetes_circuit.prove(witness, public_inputs)
    print(f"\nProof output: {proof.public_output}")

    # Verify proof
    is_valid = diabetes_circuit.verify(proof, public_inputs)
    print(f"Proof valid: {is_valid}")

    # Test biomarker circuit
    print("\n" + "=" * 50 + "\n")

    cholesterol_circuit = ClinicalBiomarkerCircuit("cholesterol")
    cholesterol_circuit.setup({"value_range": (0, 500)})

    bio_witness = cholesterol_circuit.generate_witness(
        {"value": 240}, {"threshold": 200, "comparison": "greater"}
    )

    bio_proof = cholesterol_circuit.prove(
        bio_witness, {"threshold": 200, "comparison": "greater"}
    )
    print(f"Biomarker proof: {bio_proof.public_output}")
    print(
        f"Biomarker proof valid: {cholesterol_circuit.verify(bio_proof, {'threshold': 200, 'comparison': 'greater'})}"
    )
