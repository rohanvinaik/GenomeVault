#!/usr/bin/env python3
import logging

"""Quick test to identify the exact import issue"""

import sys
import traceback

print("🔍 GenomeVault Import Diagnostic")
print("=" * 50)

# Test 1: Basic package structure
print("\n1. Testing basic package structure...")
try:
    import genomevault

    print("✓ genomevault package exists")
except Exception as e:
    print("✗ genomevault package error: {e}")

# Test 2: Core config
print("\n2. Testing core.config...")
try:
    from core.config import Config, get_config

    print("✓ core.config imports work")
except Exception as e:
    print("✗ core.config error: {e}")

# Test 3: Utils
print("\n3. Testing utils...")
try:
    from utils.logging import get_logger

    print("✓ utils.logging works")
except Exception as e:
    print("✗ utils.logging error: {e}")

try:
    from utils.encryption import AESGCMCipher

    print("✓ utils.encryption works")
except Exception as e:
    print("✗ utils.encryption error: {e}")

# Test 4: Hypervector - step by step
print("\n4. Testing hypervector_transform step by step...")

# 4a: Can we import the package?
try:
    import hypervector_transform

    print("✓ hypervector_transform package imports")
except Exception as e:
    print("✗ hypervector_transform package error: {e}")
    traceback.print_exc()

# 4b: Can we import from binding directly?
try:
    from hypervector_transform.binding import circular_bind

    print("✓ circular_bind imports from binding.py")
except Exception as e:
    print("✗ binding.py error: {e}")

# 4c: What about the __init__.py imports?
try:
    from hypervector_transform import circular_bind

    print("✓ circular_bind imports from __init__.py")
except Exception as e:
    print("✗ __init__.py re-export error: {e}")

# 4d: Check encoding
try:
    from hypervector_transform.encoding import HypervectorEncoder

    print("✓ HypervectorEncoder imports correctly")
except Exception as e:
    print("✗ encoding.py error: {e}")
    traceback.print_exc()

# Test 5: The specific import that was failing
print("\n5. Testing the specific failing import...")
try:
    from hypervector_transform import HypervectorEncoder

    print("✓ HypervectorEncoder imports from package")
except Exception as e:
    print("✗ Package-level import error: {e}")

print("\n" + "=" * 50)
print("Diagnostic complete!")
