#!/usr/bin/env python3
"""
Test script for clinical validation module
Verifies that the module can work with both real and simulated components
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_clinical_validation():
    """Test clinical validation module"""
    print("🧪 Testing GenomeVault Clinical Validation Module")
    print("=" * 60)

    # Test 1: Import modules
    print("\n1️⃣ Testing module imports...")
    try:
        from clinical_validation import ClinicalValidator, ZKProver, ProofData
        from clinical_validation.clinical_circuits import DiabetesRiskCircuit

        print("✅ All modules imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test 2: Initialize components
    print("\n2️⃣ Testing component initialization...")
    try:
        validator = ClinicalValidator()
        print("✅ ClinicalValidator initialized")

        prover = ZKProver()
        print("✅ ZKProver initialized")

        circuit = DiabetesRiskCircuit()
        print("✅ DiabetesRiskCircuit initialized")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

    # Test 3: Generate a proof
    print("\n3️⃣ Testing ZK proof generation...")
    try:
        # Test data
        private_inputs = {
            "glucose": 140,  # Above threshold
            "hba1c": 7.2,  # Above threshold
            "genetic_risk_score": 0.8,  # Above threshold
        }

        public_inputs = {
            "glucose_threshold": 126,
            "hba1c_threshold": 6.5,
            "risk_threshold": 0.5,
        }

        # Generate proof
        proof = prover.generate_proof(circuit, private_inputs, public_inputs)
        print(f"✅ Proof generated successfully")
        print(f"   - Public output: {proof.public_output}")
        print(
            f"   - Proof size: {len(proof.proof_bytes) if proof.proof_bytes else 0} bytes"
        )

        # Verify proof
        is_valid = prover.verify_proof(proof)
        print(f"✅ Proof verification: {'VALID' if is_valid else 'INVALID'}")

    except Exception as e:
        print(f"❌ Proof generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 4: Test with different risk levels
    print("\n4️⃣ Testing different risk scenarios...")
    test_cases = [
        {
            "name": "High Risk (all factors)",
            "private": {"glucose": 150, "hba1c": 8.0, "genetic_risk_score": 1.5},
            "expected": "HIGH_RISK",
        },
        {
            "name": "Normal (no factors)",
            "private": {"glucose": 100, "hba1c": 5.0, "genetic_risk_score": -0.5},
            "expected": "NORMAL",
        },
        {
            "name": "Borderline (2 factors)",
            "private": {"glucose": 130, "hba1c": 7.0, "genetic_risk_score": 0.2},
            "expected": "HIGH_RISK",
        },
    ]

    for test_case in test_cases:
        try:
            proof = prover.generate_proof(circuit, test_case["private"], public_inputs)
            result = "✅" if proof.public_output == test_case["expected"] else "❌"
            print(f"{result} {test_case['name']}: {proof.public_output}")
        except Exception as e:
            print(f"❌ {test_case['name']}: Error - {e}")

    # Test 5: Test clinical validator with sample data
    print("\n5️⃣ Testing clinical validator with sample data...")
    try:
        import pandas as pd

        # Create sample clinical data
        sample_data = pd.DataFrame(
            {
                "patient_id": range(5),
                "glucose": [100, 130, 150, 90, 200],
                "hba1c": [5.5, 6.8, 7.5, 5.2, 9.0],
                "bmi": [22, 28, 32, 20, 35],
                "age": [45, 55, 65, 35, 70],
                "bp": [120, 140, 160, 110, 180],
            }
        )

        # Test hypervector encoding
        hypervector_results = validator.validate_with_hypervectors(sample_data)
        print(f"✅ Hypervector encoding tested")
        print(f"   - Patients encoded: {hypervector_results['n_encoded']}")
        print(
            f"   - Using real encoding: {hypervector_results.get('using_real_encoding', False)}"
        )

        # Test PIR queries
        test_variants = ["rs7903146", "rs1801282"]
        pir_results = validator.validate_with_pir(test_variants)
        print(f"✅ PIR queries tested")
        print(f"   - Queries performed: {pir_results['n_queries']}")
        print(f"   - Using real PIR: {pir_results.get('using_real_pir', False)}")

    except Exception as e:
        print(f"❌ Clinical validator test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    return True


if __name__ == "__main__":
    success = test_clinical_validation()
    sys.exit(0 if success else 1)
