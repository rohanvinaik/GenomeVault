#!/usr/bin/env python3
"""Test that the variant circuit import fix worked"""

print("Testing variant circuit import fix...")

try:
    from zk_proofs.circuits.biological.variant import VariantPresenceCircuit
    print("✅ Variant circuit import successful!")
    
    # Try to instantiate it
    circuit = VariantPresenceCircuit(merkle_depth=20)
    print(f"✅ Created variant circuit: {circuit.name}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")

print("\nTesting other biological circuits...")

try:
    from zk_proofs.circuits.biological.multi_omics import MultiOmicsCorrelationCircuit
    print("✅ Multi-omics circuit import successful!")
except ImportError as e:
    print(f"❌ Multi-omics import error: {e}")

print("\nDone!")
