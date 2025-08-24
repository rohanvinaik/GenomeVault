#!/usr/bin/env python3
"""Test the improved Powers of Tau ceremony implementation."""

import sys

sys.path.insert(0, ".")

from genomevault.zk_proofs.backends.circom_backend import CircomBackend


def test_trusted_setup():
    """Test the Powers of Tau ceremony."""
    print("🔮 Testing Powers of Tau Ceremony")
    print("=" * 50)

    # Initialize backend
    backend = CircomBackend()

    # Check dependencies
    if not backend.check_dependencies():
        print("❌ Missing dependencies")
        return False

    print("✅ Dependencies available")

    # Test circuit compilation
    print("\n📋 Testing circuit compilation...")
    if not backend.compile_circuit("variant_presence"):
        print("❌ Circuit compilation failed")
        return False

    print("✅ Circuit compilation successful")

    # Test trusted setup
    print("\n🔐 Testing Powers of Tau ceremony...")
    try:
        success = backend.setup_trusted_setup("variant_presence", tau_power=12)
        if success:
            print("✅ Powers of Tau ceremony completed successfully!")

            # Check that files were created
            circuit = backend.circuits["variant_presence"]
            print("\n📁 Generated files:")
            print(f"   Proving key: {circuit.zkey_path.exists()} - {circuit.zkey_path}")
            print(f"   Verification key: {circuit.vkey_path.exists()} - {circuit.vkey_path}")

            return True
        else:
            print("❌ Powers of Tau ceremony failed")
            return False

    except Exception as e:
        print(f"❌ Error during ceremony: {e}")
        return False


if __name__ == "__main__":
    success = test_trusted_setup()

    print("\n" + "=" * 50)
    if success:
        print("🎉 TRUSTED SETUP TEST PASSED!")
    else:
        print("💥 TRUSTED SETUP TEST FAILED!")

    sys.exit(0 if success else 1)
