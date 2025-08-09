#!/usr/bin/env python3
"""
Quick test of the HYPERVECTOR_DIMENSIONS constant
"""

import os
import sys

# Change to genomevault directory
os.chdir("/Users/rohanvinaik/genomevault")

try:
    # Test 1: Import the constant
    from genomevault.core.constants import HYPERVECTOR_DIMENSIONS

    print(f"✓ HYPERVECTOR_DIMENSIONS = {HYPERVECTOR_DIMENSIONS}")

    # Test 2: Import the encoder class
    from genomevault.hypervector.encoding.unified_encoder import (
        UnifiedHypervectorEncoder,
    )

    print("✓ UnifiedHypervectorEncoder imports successfully")

    # Test 3: Verify the constant has the expected value
    assert HYPERVECTOR_DIMENSIONS == 10000, f"Expected 10000, got {HYPERVECTOR_DIMENSIONS}"
    print("✓ Constant has correct value (10000)")

    # Test 4: Test basic encoder instantiation
    encoder = UnifiedHypervectorEncoder(dimension=HYPERVECTOR_DIMENSIONS)
    print("✓ Encoder instantiated with constant successfully")
    print(f"  Encoder dimension: {encoder.dimension}")

    print("\n🎉 ALL CONSTANT TESTS PASSED!")
    print("The HYPERVECTOR_DIMENSIONS constant is working correctly.")

except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)
except AssertionError as e:
    print(f"❌ Assertion Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    sys.exit(1)
