"""Mock implementations for missing modules in tests."""

from typing import Any, Dict, List
from unittest.mock import MagicMock


class MockPRSCalculator:
    """Mock implementation of PRSCalculator for tests."""
    
    def __init__(self, snp_weights: Dict[str, float] = None):
        self.snp_weights = snp_weights or {}
        
    def calculate_score(self, variants: List[Dict[str, Any]]) -> float:
        """Calculate a mock PRS score."""
        return 0.5  # Mock score
        
    def validate_variants(self, variants: List[Dict[str, Any]]) -> bool:
        """Validate variants format."""
        return True


class MockClinicalBiomarkerCircuit:
    """Mock implementation of ClinicalBiomarkerCircuit."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        
    def generate_proof(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock proof."""
        return {
            "proof": "mock_proof_data",
            "public_inputs": inputs,
            "valid": True
        }
        
    def verify_proof(self, proof: Dict[str, Any]) -> bool:
        """Verify mock proof."""
        return proof.get("valid", False)


class MockDiabetesRiskCircuit:
    """Mock implementation of DiabetesRiskCircuit."""
    
    def __init__(self):
        pass
        
    def assess_risk(self, clinical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess diabetes risk."""
        return {
            "risk_score": 0.3,
            "risk_category": "low",
            "confidence": 0.85
        }


class MockProofData:
    """Mock implementation of ProofData."""
    
    def __init__(self, circuit_type: str, proof: str, public_inputs: Dict[str, Any]):
        self.circuit_type = circuit_type
        self.proof = proof
        self.public_inputs = public_inputs
        
    def to_json(self) -> str:
        """Convert to JSON."""
        return {
            "circuit_type": self.circuit_type,
            "proof": self.proof,
            "public_inputs": self.public_inputs
        }


def create_circuit(circuit_type: str, **kwargs) -> Any:
    """Factory function to create mock circuits."""
    if circuit_type == "clinical_biomarker":
        return MockClinicalBiomarkerCircuit(**kwargs)
    elif circuit_type == "diabetes_risk":
        return MockDiabetesRiskCircuit(**kwargs)
    else:
        return MagicMock()


def verify_proof(proof_data: MockProofData) -> bool:
    """Mock proof verification."""
    return True


class MockGovernanceLedger:
    """Mock implementation of governance ledger."""
    
    def __init__(self):
        self.entries = []
        
    def add_entry(self, entry: Dict[str, Any]) -> str:
        """Add entry to ledger."""
        entry_id = f"entry_{len(self.entries)}"
        self.entries.append({"id": entry_id, **entry})
        return entry_id
        
    def get_entry(self, entry_id: str) -> Dict[str, Any]:
        """Get entry from ledger."""
        for entry in self.entries:
            if entry["id"] == entry_id:
                return entry
        return None
        
    def verify_integrity(self) -> bool:
        """Verify ledger integrity."""
        return True