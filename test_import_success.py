#!/usr/bin/env python3
"""Simple test script with clear output"""

print("\n" + "=" * 50)
print("TESTING VARIANT CIRCUIT IMPORT FIX")
print("=" * 50)

# The problematic import that we fixed
print("\nAttempting import...")
try:
    from zk_proofs.circuits.biological.variant import VariantPresenceCircuit

    print("✅ SUCCESS! Import works!")

    # Create an instance to prove it fully works
    circuit = VariantPresenceCircuit()
    print(f"✅ Created circuit: {circuit.name}")

    # The fix was changing the import from:
    # from .base_circuits import ...
    # to:
    # from ..base_circuits import ...

    print("\n✅ THE FIX WORKED! ✅")
    print("\nThe import path was corrected from:")
    print("  ❌ from .base_circuits import ...")
    print("to:")
    print("  ✅ from ..base_circuits import ...")

except ImportError as e:
    print(f"❌ IMPORT FAILED: {e}")
except Exception as e:
    print(f"❌ OTHER ERROR: {e}")

print("\n" + "=" * 50)
