import os
from typing import Any, Dict

#!/usr/bin/env python3
"""
Trace the exact import failure point
"""

import sys
import traceback

print("=" * 70)
print("TRACING IMPORT FAILURE")
print("=" * 70)

# First, let's see if we can import our __init__ files
print("\n1. Testing __init__.py imports...")

try:
    import zk_proofs

    print("✅ zk_proofs/__init__.py imported")
except Exception as e:
    print("❌ zk_proofs/__init__.py failed: {e}")
    traceback.print_exc()
    print("\nThis is likely where cryptography is first imported")

# Let's check what's in the zk_proofs __init__.py
print("\n2. Checking zk_proofs/__init__.py content...")
try:
    with open("zk_proofs/__init__.py", "r") as f:
        content = f.read()
        if "cryptography" in content:
            print("❌ Found 'cryptography' import in zk_proofs/__init__.py")
        else:
            print("✅ No direct 'cryptography' import in zk_proofs/__init__.py")
            print("   Import chain must be indirect...")
except Exception as e:
    print("Could not read file: {e}")

# Let's check the utils module since that's likely imported
print("\n3. Testing utils imports...")
try:
    import utils

    print("✅ utils/__init__.py imported")
except Exception as e:
    print("❌ utils/__init__.py failed: {e}")

# Check core module
print("\n4. Testing core imports...")
try:
    import core

    print("✅ core/__init__.py imported")
except Exception as e:
    print("❌ core/__init__.py failed: {e}")

print("\n5. Looking for the cryptography import...")


def find_cryptography_imports(directory) -> None:
       """TODO: Add docstring for find_cryptography_imports"""
     """Find all files that import cryptography"""
    files_with_crypto = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__
        if "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r") as f:
                        content = f.read()
                        if "from cryptography" in content or "import cryptography" in content:
                            files_with_crypto.append(filepath)
                except Exception:
                    pass
    return files_with_crypto


crypto_files = find_cryptography_imports(".")
print("\nFiles importing cryptography:")
for f in crypto_files[:10]:  # Show first 10
    print("  - {f}")

print("\n" + "=" * 70)
