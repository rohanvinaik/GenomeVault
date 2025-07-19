#!/usr/bin/env python3
"""Test import with minimal dependencies"""

# First, let's mock the missing modules
import sys
from unittest.mock import MagicMock

# Mock the missing dependencies
sys.modules["structlog"] = MagicMock()
sys.modules["yaml"] = MagicMock()
sys.modules["cryptography"] = MagicMock()
sys.modules["cryptography.fernet"] = MagicMock()
sys.modules["cryptography.hazmat"] = MagicMock()
sys.modules["cryptography.hazmat.primitives"] = MagicMock()
sys.modules["cryptography.hazmat.primitives.kdf"] = MagicMock()
sys.modules["cryptography.hazmat.primitives.kdf.pbkdf2"] = MagicMock()
sys.modules["cryptography.hazmat.backends"] = MagicMock()
sys.modules["cryptography.hazmat.primitives.asymmetric"] = MagicMock()
sys.modules["cryptography.hazmat.primitives.ciphers"] = MagicMock()
sys.modules["cryptography.hazmat.primitives.kdf.hkdf"] = MagicMock()
sys.modules["cryptography.hazmat.primitives.serialization"] = MagicMock()
sys.modules["nacl"] = MagicMock()
sys.modules["nacl.secret"] = MagicMock()
sys.modules["nacl.utils"] = MagicMock()
sys.modules["nacl.public"] = MagicMock()
sys.modules["nacl.signing"] = MagicMock()

print("=" * 60)
print("TESTING VARIANT CIRCUIT IMPORT WITH MOCKED DEPENDENCIES")
print("=" * 60)

# Now try the import
print("\nAttempting import...")
try:
    from zk_proofs.circuits.biological.variant import VariantPresenceCircuit

    print("✅ SUCCESS! Import works!")

    # Create an instance
    circuit = VariantPresenceCircuit(merkle_depth=20)
    print(f"✅ Created circuit: {circuit.name}")
    print(f"   - Type: {type(circuit)}")
    print(f"   - Constraints: {circuit.num_constraints}")
    print(f"   - Merkle depth: {circuit.merkle_depth}")

    print("\n✅ THE IMPORT FIX WORKED! ✅")
    print("\nThe fix was changing the import path from:")
    print("  ❌ from .base_circuits import ...")
    print("to:")
    print("  ✅ from ..base_circuits import ...")
    print("\nThis correctly imports from the parent circuits/ directory.")

except ImportError as e:
    print(f"❌ IMPORT FAILED: {e}")
    import traceback

    traceback.print_exc()
except Exception as e:
    print(f"❌ OTHER ERROR: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
