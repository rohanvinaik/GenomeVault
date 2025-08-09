#!/usr/bin/env python3

"""Test script to verify all stub fixes are working."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all previously failing imports now work."""
    errors = []

    # Test 1: Import verifier module
    try:
        print("Testing verifier import...")
        from genomevault.zk_proofs import verifier

        print("✓ Verifier module imported successfully")
    except ImportError as e:
        errors.append(f"✗ Verifier import failed: {e}")

    # Test 2: Import circuit_manager
    try:
        print("Testing circuit_manager import...")
        from genomevault.zk_proofs import circuit_manager

        print("✓ Circuit manager imported successfully")
    except ImportError as e:
        errors.append(f"✗ Circuit manager import failed: {e}")

    # Test 3: Import base_circuits with all classes
    try:
        print("Testing base_circuits classes...")
        from genomevault.zk_proofs.circuits.base_circuits import (
            BaseCircuit,
            FieldElement,
            ComparisonCircuit,
            MerkleTreeCircuit,
            RangeProofCircuit,
        )

        print("✓ All base circuit classes imported successfully")
    except ImportError as e:
        errors.append(f"✗ Base circuits import failed: {e}")

    # Test 4: Import PIR core module
    try:
        print("Testing PIR core module...")
        from genomevault.pir.core import PIRClient, PIRServer

        print("✓ PIR core module imported successfully")
    except ImportError as e:
        errors.append(f"✗ PIR core import failed: {e}")

    # Test 5: Import diabetes circuits
    try:
        print("Testing diabetes circuits...")
        from genomevault.zk_proofs.circuits.biological.diabetes import (
            DiabetesRiskCircuit,
            GlucoseMonitoringCircuit,
        )

        print("✓ Diabetes circuits imported successfully")
    except ImportError as e:
        errors.append(f"✗ Diabetes circuits import failed: {e}")

    # Test 6: Import variant circuits
    try:
        print("Testing variant circuits...")
        from genomevault.zk_proofs.circuits.biological.variant import (
            PathwayEnrichmentCircuit,
            PharmacogenomicCircuit,
            PolygenenicRiskScoreCircuit,
            VariantPresenceCircuit,
        )

        print("✓ Variant circuits imported successfully")
    except ImportError as e:
        errors.append(f"✗ Variant circuits import failed: {e}")

    # Test 7: Test the log_operation decorator
    try:
        print("Testing log_operation decorator...")
        from genomevault.utils.logging import log_operation

        @log_operation
        def test_function():
            return "test"

        result = test_function()
        print("✓ log_operation decorator works correctly")
    except Exception as e:
        errors.append(f"✗ log_operation decorator failed: {e}")

    # Test 8: Import prover module
    try:
        print("Testing prover import...")
        from genomevault.zk_proofs.prover import Prover, Proof, Circuit

        print("✓ Prover module imported successfully")
    except ImportError as e:
        errors.append(f"✗ Prover import failed: {e}")

    return errors


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Stub Fixes")
    print("=" * 60)
    print()

    errors = test_imports()

    print()
    print("=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} errors found")
        print("=" * 60)
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        print("SUCCESS: All imports working correctly!")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
