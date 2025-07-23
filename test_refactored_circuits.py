"""
Comprehensive test for the refactored clinical circuits.
Validates that the new implementation works correctly.
"""

import sys
import os

# Add the genomevault path for imports
sys.path.insert(0, "/Users/rohanvinaik/genomevault")

from clinical_validation.circuits import (
    DiabetesRiskCircuit,
    ClinicalBiomarkerCircuit,
    create_circuit,
)
from clinical_validation.proofs import CircuitType, verify_proof


def test_diabetes_circuit():
    """Test the diabetes risk circuit"""
    print("Testing Diabetes Risk Circuit...")

    # Create circuit
    circuit = DiabetesRiskCircuit()

    # Setup
    setup_params = {"glucose_range": (70, 300), "hba1c_range": (4, 14), "risk_factors_threshold": 2}

    config = circuit.setup(setup_params)
    print(f"✓ Circuit setup: {config['circuit_name']} v{config['version']}")

    # Generate witness
    private_inputs = {"glucose": 130, "hba1c": 7.0, "genetic_risk_score": 1.2}

    public_inputs = {"glucose_threshold": 126, "hba1c_threshold": 6.5, "risk_threshold": 1.0}

    witness = circuit.generate_witness(private_inputs, public_inputs)
    print(f"✓ Witness generated: {witness['result']}")

    # Generate proof
    proof = circuit.prove(witness, public_inputs)
    print(f"✓ Proof generated: {proof.public_output}")

    # Verify proof
    is_valid = circuit.verify(proof, public_inputs)
    print(f"✓ Proof verification: {is_valid}")

    # Test unified verifier
    unified_valid = verify_proof(proof, public_inputs)
    print(f"✓ Unified verifier: {unified_valid}")

    return is_valid and unified_valid


def test_biomarker_circuit():
    """Test the biomarker circuit"""
    print("\nTesting Biomarker Circuit...")

    # Create circuit using factory
    circuit = create_circuit(CircuitType.BIOMARKER_THRESHOLD, biomarker_name="cholesterol")

    # Setup
    circuit.setup({"value_range": (0, 500)})
    print(f"✓ Circuit setup: {circuit.config.name}")

    # Generate witness
    bio_witness = circuit.generate_witness(
        {"value": 240}, {"threshold": 200, "comparison": "greater"}
    )
    print(f"✓ Witness generated: result={bio_witness['result']}")

    # Generate proof
    bio_proof = circuit.prove(bio_witness, {"threshold": 200, "comparison": "greater"})
    print(f"✓ Proof generated: {bio_proof.public_output}")

    # Verify proof
    is_valid = circuit.verify(bio_proof, {"threshold": 200, "comparison": "greater"})
    print(f"✓ Proof verification: {is_valid}")

    # Test unified verifier
    unified_valid = verify_proof(bio_proof, {"threshold": 200, "comparison": "greater"})
    print(f"✓ Unified verifier: {unified_valid}")

    return is_valid and unified_valid


def test_serialization():
    """Test proof serialization and deserialization"""
    print("\nTesting Proof Serialization...")

    circuit = DiabetesRiskCircuit()
    circuit.setup({})

    private_inputs = {"glucose": 140, "hba1c": 8.0, "genetic_risk_score": 1.5}
    public_inputs = {"glucose_threshold": 126, "hba1c_threshold": 6.5, "risk_threshold": 1.0}

    witness = circuit.generate_witness(private_inputs, public_inputs)
    original_proof = circuit.prove(witness, public_inputs)

    # Serialize
    serialized = original_proof.serialize()
    print(f"✓ Proof serialized: {len(serialized)} bytes")

    # Deserialize
    from clinical_validation.proofs import ProofData

    deserialized_proof = ProofData.deserialize(serialized)
    print(f"✓ Proof deserialized: {deserialized_proof.public_output}")

    # Verify deserialized proof works
    is_valid = circuit.verify(deserialized_proof, public_inputs)
    print(f"✓ Deserialized proof verification: {is_valid}")

    return is_valid


def test_error_handling():
    """Test error handling and validation"""
    print("\nTesting Error Handling...")

    circuit = DiabetesRiskCircuit()
    circuit.setup({})

    # Test invalid glucose range
    try:
        circuit.generate_witness(
            {"glucose": 500, "hba1c": 7.0},  # Invalid glucose
            {"glucose_threshold": 126, "hba1c_threshold": 6.5},
        )
        print("✗ Should have failed with invalid glucose")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught invalid glucose: {e}")

    # Test circuit not setup
    circuit2 = ClinicalBiomarkerCircuit("test")
    try:
        witness = circuit2.generate_witness({"value": 100}, {"threshold": 50})
        circuit2.prove(witness, {"threshold": 50})
        print("✗ Should have failed without setup")
        return False
    except RuntimeError as e:
        print(f"✓ Correctly caught uninitialized circuit: {e}")

    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("REFACTORED CLINICAL CIRCUITS COMPREHENSIVE TEST")
    print("=" * 60)

    tests = [
        ("Diabetes Circuit", test_diabetes_circuit),
        ("Biomarker Circuit", test_biomarker_circuit),
        ("Serialization", test_serialization),
        ("Error Handling", test_error_handling),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"\n{test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"\n{test_name}: ERROR - {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} {status}")
        if not result:
            all_passed = False

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
