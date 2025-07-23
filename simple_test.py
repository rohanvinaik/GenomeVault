#!/usr/bin/env python3
"""Simple test script to validate the refactored implementation"""

import sys
import os

sys.path.insert(0, "/Users/rohanvinaik/genomevault")


def test_imports():
    """Test if all imports work"""
    try:
        from clinical_validation.circuits import DiabetesRiskCircuit

        print("✓ DiabetesRiskCircuit import successful")

        from clinical_validation.proofs import CircuitType

        print("✓ CircuitType import successful")

        # Quick functionality test
        circuit = DiabetesRiskCircuit()
        circuit.setup({})
        print("✓ Basic circuit creation successful")

        return True
    except Exception as e:
        print(f"✗ Import/setup error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    print(f"Test result: {'SUCCESS' if success else 'FAILED'}")
