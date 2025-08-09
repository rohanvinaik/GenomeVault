"""
Unit tests for the refactored clinical circuits.
"""

import unittest

try:
    from tests.mocks import (
        MockClinicalBiomarkerCircuit as ClinicalBiomarkerCircuit,
        MockDiabetesRiskCircuit as DiabetesRiskCircuit,
        MockProofData as ProofData,
        create_circuit,
        verify_proof,
    )
    from enum import Enum
    
    class CircuitType(Enum):
        """Mock CircuitType enum."""
        CLINICAL_BIOMARKER = "clinical_biomarker"
        DIABETES_RISK = "diabetes_risk"

    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestRefactoredCircuits(unittest.TestCase):
    """Test suite for refactored clinical circuits"""

    def setUp(self):
        """Set up test fixtures"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")

    def test_diabetes_circuit_basic(self):
        """Test basic diabetes circuit functionality"""
        # Create and setup circuit
        circuit = DiabetesRiskCircuit()
        config = circuit.setup({"risk_factors_threshold": 2})

        self.assertEqual(config["circuit_name"], "DiabetesRiskCircuit")
        self.assertEqual(config["version"], "2.0.0")

        # Generate witness
        private_inputs = {"glucose": 130, "hba1c": 7.0, "genetic_risk_score": 1.2}
        public_inputs = {
            "glucose_threshold": 126,
            "hba1c_threshold": 6.5,
            "risk_threshold": 1.0,
        }

        witness = circuit.generate_witness(private_inputs, public_inputs)
        self.assertIn("result", witness)
        self.assertIn("is_high_risk", witness["result"])

        # Generate and verify proof
        proof = circuit.prove(witness, public_inputs)
        self.assertIsInstance(proof, ProofData)
        self.assertTrue(proof.public_output.startswith("RISK_LEVEL:"))

        # Verify proof
        is_valid = circuit.verify(proof, public_inputs)
        self.assertTrue(is_valid)

    def test_biomarker_circuit_basic(self):
        """Test basic biomarker circuit functionality"""
        # Create circuit using factory
        circuit = create_circuit(CircuitType.BIOMARKER_THRESHOLD, biomarker_name="test")
        circuit.setup({"value_range": (0, 500)})

        # Test witness generation
        witness = circuit.generate_witness(
            {"value": 240}, {"threshold": 200, "comparison": "greater"}
        )
        self.assertTrue(witness["result"])  # 240 > 200

        # Test proof generation and verification
        proof = circuit.prove(witness, {"threshold": 200, "comparison": "greater"})
        self.assertTrue(proof.public_output.startswith("test:"))

        is_valid = circuit.verify(proof, {"threshold": 200, "comparison": "greater"})
        self.assertTrue(is_valid)

    def test_proof_serialization(self):
        """Test proof serialization and deserialization"""
        circuit = DiabetesRiskCircuit()
        circuit.setup({})

        # Create a proof
        private_inputs = {"glucose": 140, "hba1c": 8.0, "genetic_risk_score": 1.5}
        public_inputs = {
            "glucose_threshold": 126,
            "hba1c_threshold": 6.5,
            "risk_threshold": 1.0,
        }

        witness = circuit.generate_witness(private_inputs, public_inputs)
        original_proof = circuit.prove(witness, public_inputs)

        # Serialize and deserialize
        serialized = original_proof.serialize()
        deserialized_proof = ProofData.deserialize(serialized)

        # Check that deserialized proof is equivalent
        self.assertEqual(original_proof.public_output, deserialized_proof.public_output)
        self.assertEqual(original_proof.circuit_type, deserialized_proof.circuit_type)

        # Verify deserialized proof works
        is_valid = circuit.verify(deserialized_proof, public_inputs)
        self.assertTrue(is_valid)

    def test_unified_verifier(self):
        """Test the unified verification function"""
        circuit = DiabetesRiskCircuit()
        circuit.setup({})

        private_inputs = {"glucose": 100, "hba1c": 5.0, "genetic_risk_score": 0.5}
        public_inputs = {
            "glucose_threshold": 126,
            "hba1c_threshold": 6.5,
            "risk_threshold": 1.0,
        }

        witness = circuit.generate_witness(private_inputs, public_inputs)
        proof = circuit.prove(witness, public_inputs)

        # Test unified verifier
        is_valid = verify_proof(proof, public_inputs)
        self.assertTrue(is_valid)

    def test_error_handling(self):
        """Test error handling and validation"""
        circuit = DiabetesRiskCircuit()
        circuit.setup({})

        # Test invalid glucose range
        with self.assertRaises(ValueError):
            circuit.generate_witness(
                {
                    "glucose": 500,
                    "hba1c": 7.0,
                    "genetic_risk_score": 0.5,
                },  # Invalid glucose
                {
                    "glucose_threshold": 126,
                    "hba1c_threshold": 6.5,
                    "risk_threshold": 1.0,
                },
            )

        # Test circuit not setup for proof generation
        circuit2 = ClinicalBiomarkerCircuit("test")
        witness = circuit2.generate_witness({"value": 100}, {"threshold": 50})

        with self.assertRaises(RuntimeError):
            circuit2.prove(witness, {"threshold": 50})


if __name__ == "__main__":
    unittest.main()
