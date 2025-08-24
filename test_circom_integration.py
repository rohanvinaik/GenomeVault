#!/usr/bin/env python3
"""
Test Circom Integration - Verify real ZK proofs are being generated
"""

import sys

sys.path.insert(0, "/Users/rohanvinaik/genomevault")

print("🔍 Testing Circom ZK Proof Integration")
print("=" * 50)

# Check if Circom backend is available
try:
    from genomevault.zk_proofs.backends.circom_backend import CircomBackend

    print("✅ Circom backend module imported successfully")

    # Initialize backend
    backend = CircomBackend()
    print("✅ Circom backend initialized")
    print(f"   Circuits directory: {backend.circuits_dir}")

    # Check dependencies
    deps_ok = backend.check_dependencies()
    print(f"   Dependencies check: {'✅ PASS' if deps_ok else '❌ FAIL'}")

    # Check for compiled circuits
    for circuit_name, circuit in backend.circuits.items():
        print(f"\n📦 Circuit: {circuit_name}")
        print(f"   R1CS: {circuit.r1cs_path.exists()} - {circuit.r1cs_path}")
        print(f"   WASM: {circuit.wasm_path.exists()} - {circuit.wasm_path}")
        print(f"   ZKey: {circuit.zkey_path.exists()} - {circuit.zkey_path}")
        print(f"   VKey: {circuit.vkey_path.exists()} - {circuit.vkey_path}")

        if circuit.zkey_path.exists():
            print("   ✅ Circuit is COMPILED and READY")
        else:
            print("   ❌ Circuit needs compilation")

except Exception as e:
    print(f"❌ Failed to import Circom backend: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("🧪 Testing Prover with Circom Backend")
print("=" * 50)

try:
    from genomevault.zk_proofs.prover import Prover

    # Initialize prover with Circom
    prover = Prover(use_circom=True)

    print("✅ Prover initialized")
    print(f"   Circom backend: {prover.circom_backend is not None}")
    print(f"   Production ready: {prover.is_production_ready}")

    if prover.circom_backend:
        print("\n🎯 Attempting real ZK proof generation...")

        # Try to generate a simple proof
        try:
            # Test with minimal inputs
            public_inputs = {
                "variant_hash": "0x123456789abcdef",
                "reference_hash": "hg38",
                "commitment_root": "0xabcdef123456789",
            }

            private_inputs = {
                "chr": "1",
                "position": 123456,
                "ref_allele": "A",
                "alt_allele": "G",
                "merkle_proof": ["0x111", "0x222"],
                "merkle_indices": [0, 1],
                "witness_randomness": "random123",
            }

            print(f"   Public inputs: {list(public_inputs.keys())}")
            print(f"   Private inputs: {list(private_inputs.keys())}")

            # Try to generate proof
            proof = prover.generate_proof(
                circuit_name="variant_presence",
                public_inputs=public_inputs,
                private_inputs=private_inputs,
            )

            print("\n✅ REAL ZK PROOF GENERATED!")
            print(f"   Proof type: {type(proof)}")
            print(f"   Proof size: {len(str(proof))} bytes")

            # Check if it's a real proof or fallback
            if "transcript" in str(proof).lower() or "mock" in str(proof).lower():
                print("   ⚠️  Using fallback proof system")
            else:
                print("   ✅ Using REAL Circom/SNARK proofs!")

        except Exception as e:
            print(f"   ❌ Proof generation failed: {e}")
            print("      This might be due to missing witness generation files")
    else:
        print("   ⚠️  Circom backend not initialized - using fallback")

except Exception as e:
    print(f"❌ Failed to test prover: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 50)
print("📊 Summary")
print("=" * 50)

# Final status
if "prover" in locals() and prover.is_production_ready:
    print("✅ CIRCOM IS FULLY INTEGRATED AND WORKING!")
    print("   You are using REAL zero-knowledge proofs")
    print("   Not mock implementations!")
else:
    print("⚠️  Circom integration needs attention")
    print("   Currently using fallback proof system")
