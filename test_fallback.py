#!/usr/bin/env python3
"""
Test ZK proof system with transcript fallback mode.

This tests the fallback mode that works without compiled circuits.
"""

import json
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from genomevault.zk.real_engine import RealZKEngine


def test_transcript_fallback():
    """Test proof generation with transcript fallback."""
    print("=" * 60)
    print("Testing ZK Proof System (Transcript Fallback Mode)")
    print("=" * 60)
    
    # Initialize engine
    print("\n1. Initializing RealZKEngine...")
    engine = RealZKEngine(str(project_root))
    
    print(f"   Toolchain available: {engine.toolchain_available}")
    print("   Using transcript fallback mode")
    
    # Test case 1: sum64 circuit simulation
    print("\n2. Testing sum64 circuit (transcript mode)...")
    print("   Input: a=15, b=27, c=42")
    
    inputs = {
        "a": 15,
        "b": 27,
        "c": 42
    }
    
    # Generate proof (will use transcript fallback)
    print("   Generating proof...")
    proof = engine.generate_proof("sum64", inputs)
    
    if proof:
        print(f"   ✓ Proof generated successfully")
        print(f"   Circuit type: {proof.circuit_type}")
        print(f"   Public outputs: {proof.public}")
        
        # Check if using fallback
        if proof.metadata and proof.metadata.get("fallback"):
            print("   ✓ Using transcript fallback as expected")
        
        # Check proof structure
        if proof.proof.get("engine") == "transcript":
            print(f"   ✓ Transcript engine confirmed")
            print(f"   Algorithm: {proof.proof.get('algorithm')}")
            print(f"   Claim: {proof.proof.get('claim')}")
            
            # Verify transcript has signature
            if "signature" in proof.proof:
                print(f"   ✓ Transcript is signed")
                print(f"   Signature: {proof.proof['signature'][:16]}...")
        
        # Verify proof
        print("\n3. Verifying proof...")
        is_valid = engine.verify_proof(
            proof.proof,
            proof.public,
            circuit_type="sum64"
        )
        
        if is_valid:
            print("   ✓ Proof verified successfully!")
        else:
            print("   ✗ Proof verification failed")
            return False
        
        # Test case 2: Invalid verification
        print("\n4. Testing invalid proof (wrong public output)...")
        wrong_public = {"c": "999"}
        
        # Create a new proof for correct values
        valid_proof = engine.generate_proof("sum64", inputs)
        
        # Try to verify with wrong public values
        is_invalid = engine.verify_proof(
            valid_proof.proof,
            wrong_public,
            circuit_type="sum64"
        )
        
        if not is_invalid:
            print("   ✓ Invalid proof correctly rejected!")
        else:
            print("   ✗ Invalid proof was incorrectly accepted")
            return False
        
        # Test case 3: Custom circuit type
        print("\n5. Testing custom circuit type...")
        custom_inputs = {
            "data": "test_value",
            "hash": "abc123",
            "_private": "secret"  # Private input (starts with _)
        }
        
        custom_proof = engine.generate_proof("custom_circuit", custom_inputs)
        
        if custom_proof:
            print(f"   ✓ Custom proof generated")
            print(f"   Public inputs: {custom_proof.public}")
            
            # Verify private input was not included in public
            if "_private" not in custom_proof.public:
                print("   ✓ Private inputs correctly excluded from public")
            
            # Verify custom proof
            is_valid_custom = engine.verify_proof(
                custom_proof.proof,
                custom_proof.public,
                circuit_type="custom_circuit"
            )
            
            if is_valid_custom:
                print("   ✓ Custom proof verified!")
            else:
                print("   ✗ Custom proof verification failed")
                return False
        
        print("\n" + "=" * 60)
        print("✅ All transcript fallback tests passed!")
        print("=" * 60)
        print("\nNote: To use real ZK proofs with snarkjs:")
        print("1. Install circom: https://docs.circom.io/getting-started/installation/")
        print("2. Run: ./scripts/build_circuits.sh")
        print("3. Run: python test_integration.py")
        
        return True
    else:
        print("   ✗ Failed to generate proof")
        return False


def main():
    """Run fallback tests."""
    print("\n🔬 GenomeVault ZK Fallback Mode Test")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not (project_root / "genomevault").exists():
        print("❌ Error: Must run from project root directory")
        sys.exit(1)
    
    # Run test
    success = test_transcript_fallback()
    
    if success:
        print("\n🎉 Fallback mode tests successful!")
        print("\nThe ZK engine is working correctly with transcript fallback.")
        print("Real ZK proofs will be available once circom is installed.")
        return 0
    else:
        print("\n❌ Fallback mode tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())