#!/usr/bin/env python3
"""
Integration test for ZK proof generation and verification.

This script tests the updated RealZKEngine with snarkjs integration.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from genomevault.zk.real_engine import RealZKEngine


def test_sum64_proof():
    """Test sum64 circuit proof generation and verification."""
    print("=" * 60)
    print("Testing ZK Proof Generation and Verification")
    print("=" * 60)

    # Initialize engine
    print("\n1. Initializing RealZKEngine...")
    engine = RealZKEngine(str(project_root))

    # List available circuits
    print("\n2. Checking available circuits...")
    available = engine.list_available_circuits()
    print(f"   Available circuits: {available}")

    if not available:
        print("\n   ⚠️  No compiled circuits found!")
        print("   Run: ./scripts/build_circuits.sh")
        print("   to build the sum64 circuit first.")
        return False

    # Test sum64 if available
    if "sum64" in available:
        print("\n3. Testing sum64 circuit...")

        # Test case 1: Simple addition
        print("\n   Test Case 1: 15 + 27 = 42")
        inputs = {
            "a": 15,
            "b": 27,
            "c": 42,  # public output
        }

        print(f"   Inputs: a={inputs['a']}, b={inputs['b']}, c={inputs['c']}")

        # Generate proof
        print("   Generating proof...")
        proof = engine.generate_proof("sum64", inputs)

        if proof:
            print("   ✓ Proof generated successfully")
            print(f"   Public outputs: {proof.public}")
            print(f"   Circuit type: {proof.circuit_type}")

            # Check metadata
            if proof.metadata:
                if proof.metadata.get("fallback"):
                    print("   ⚠️  Using transcript fallback (snarkjs not available)")
                else:
                    print(f"   Engine: {proof.metadata.get('engine', 'unknown')}")
                    print(f"   Backend: {proof.metadata.get('backend', 'unknown')}")

            # Verify proof
            print("\n   Verifying proof...")
            is_valid = engine.verify_proof(proof.proof, proof.public, circuit_type="sum64")

            if is_valid:
                print("   ✓ Proof verified successfully!")
            else:
                print("   ✗ Proof verification failed")
                return False

            # Test case 2: Different values
            print("\n   Test Case 2: 100 + 250 = 350")
            inputs2 = {"a": 100, "b": 250, "c": 350}

            proof2 = engine.generate_proof("sum64", inputs2)
            if proof2:
                is_valid2 = engine.verify_proof(proof2.proof, proof2.public, circuit_type="sum64")

                if is_valid2:
                    print("   ✓ Second proof verified successfully!")
                else:
                    print("   ✗ Second proof verification failed")
                    return False

            # Test case 3: Invalid proof (wrong public output)
            print("\n   Test Case 3: Testing invalid proof")
            print("   Attempting to verify with wrong public output...")

            wrong_public = {"c": "999"}  # Wrong value
            is_invalid = engine.verify_proof(proof.proof, wrong_public, circuit_type="sum64")

            if not is_invalid:
                print("   ✓ Invalid proof correctly rejected!")
            else:
                print("   ✗ Invalid proof was incorrectly accepted")
                return False

            print("\n" + "=" * 60)
            print("✅ All tests passed successfully!")
            print("=" * 60)
            return True

        else:
            print("   ✗ Failed to generate proof")
            return False
    else:
        print("\n   ⚠️  sum64 circuit not available")
        print("   Run: ./scripts/build_circuits.sh")
        return False


def test_circuit_loading():
    """Test circuit artifact loading and caching."""
    print("\n" + "=" * 60)
    print("Testing Circuit Loading and Caching")
    print("=" * 60)

    engine = RealZKEngine(str(project_root))

    # Test loading sum64
    print("\n1. Loading sum64 circuit...")
    artifacts = engine.load_circuit("sum64")

    if artifacts:
        print("   ✓ Circuit loaded successfully")
        print(f"   WASM: {artifacts.wasm_path.name}")
        print(f"   ZKey: {artifacts.zkey_path.name}")
        print(f"   VKey: {artifacts.vkey_path.name}")
        print(f"   R1CS: {artifacts.r1cs_path.name}")

        # Test caching
        print("\n2. Testing cache (loading again)...")
        artifacts2 = engine.load_circuit("sum64")
        if artifacts2 is artifacts:
            print("   ✓ Circuit loaded from cache")
        else:
            print("   ⚠️  Circuit not cached properly")

        # Load verification key
        print("\n3. Loading verification key...")
        try:
            vkey = artifacts.verification_key
            print("   ✓ Verification key loaded")
            print(f"   Key type: {vkey.get('protocol', 'unknown')}")
            print(f"   Curve: {vkey.get('curve', 'unknown')}")
        except Exception as e:
            print(f"   ✗ Failed to load verification key: {e}")
            return False
    else:
        print("   ⚠️  Circuit not found")
        print("   Run: ./scripts/build_circuits.sh first")
        return False

    # Test non-existent circuit
    print("\n4. Testing non-existent circuit...")
    artifacts3 = engine.load_circuit("non_existent")
    if artifacts3 is None:
        print("   ✓ Correctly returned None for missing circuit")
    else:
        print("   ✗ Should have returned None")
        return False

    print("\n✅ Circuit loading tests passed!")
    return True


def main():
    """Run all integration tests."""
    print("\n🔬 GenomeVault ZK Integration Tests")
    print("=" * 60)

    # Check if we're in the right directory
    if not (project_root / "genomevault").exists():
        print("❌ Error: Must run from project root directory")
        sys.exit(1)

    # Run tests
    results = []

    # Test circuit loading
    results.append(("Circuit Loading", test_circuit_loading()))

    # Test proof generation/verification
    results.append(("Proof Generation", test_sum64_proof()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
