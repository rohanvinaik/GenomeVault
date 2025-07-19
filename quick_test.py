#!/usr/bin/env python3
"""Quick test of the import fix"""

print("Testing import fix...")

# Test 1: Base import
try:
    from zk_proofs.circuits.biological.variant import VariantPresenceCircuit

    print("✅ SUCCESS: VariantPresenceCircuit imported")
    circuit = VariantPresenceCircuit()
    print(f"✅ Circuit created: {circuit.name}")
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")

# Test 2: Check what the error was about
print("\nChecking base_circuits location...")
import os

base_path = "/Users/rohanvinaik/genomevault/zk_proofs/circuits"
files = os.listdir(base_path)
print(f"Files in {base_path}:")
for f in files:
    print(f"  - {f}")

print("\nDONE!")
