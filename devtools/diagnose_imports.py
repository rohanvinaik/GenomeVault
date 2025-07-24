#!/usr/bin/env python3
"""Quick test to identify the exact import issue"""

import traceback

print("🔍 GenomeVault Import Diagnostic")
print(" = " * 50)

# Test 1: Basic package structure
print("\n1. Testing basic package structure...")
try:
    pass

    print("✓ genomevault package exists")
except Exception:
    print("✗ genomevault package error: {e}")

# Test 2: Core config
print("\n2. Testing core.config...")
try:
    pass

    print("✓ core.config imports work")
except Exception:
    print("✗ core.config error: {e}")

# Test 3: Utils
print("\n3. Testing utils...")
try:
    pass

    print("✓ utils.logging works")
except Exception:
    print("✗ utils.logging error: {e}")

try:
    pass

    print("✓ utils.encryption works")
except Exception:
    print("✗ utils.encryption error: {e}")

# Test 4: Hypervector - step by step
print("\n4. Testing hypervector_transform step by step...")

# 4a: Can we import the package?
try:
    pass

    print("✓ hypervector_transform package imports")
except Exception:
    print("✗ hypervector_transform package error: {e}")
    traceback.print_exc()

# 4b: Can we import from binding directly?
try:
    pass

    print("✓ circular_bind imports from binding.py")
except Exception:
    print("✗ binding.py error: {e}")

# 4c: What about the __init__.py imports?
try:
    pass

    print("✓ circular_bind imports from __init__.py")
except Exception:
    print("✗ __init__.py re-export error: {e}")

# 4d: Check encoding
try:
    pass

    print("✓ HypervectorEncoder imports correctly")
except Exception:
    print("✗ encoding.py error: {e}")
    traceback.print_exc()

# Test 5: The specific import that was failing
print("\n5. Testing the specific failing import...")
try:
    pass

    print("✓ HypervectorEncoder imports from package")
except Exception:
    print("✗ Package-level import error: {e}")

print("\n" + " = " * 50)
print("Diagnostic complete!")
