#!/usr/bin/env python3
"""Simple benchmark runner that works"""

import os
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Minimal imports to test
print("Testing basic imports...")

try:
    import numpy as np

    print("✓ NumPy imported")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import torch

    print("✓ PyTorch imported")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    from genomevault.hypervector.encoding import GenomicEncoder, PackedGenomicEncoder

    print("✓ GenomeVault encoders imported")
except ImportError as e:
    print(f"✗ GenomeVault import failed: {e}")
    print("\nTrying to fix import path...")

    # Try direct import
    try:
        import genomevault

        print(f"GenomeVault found at: {genomevault.__file__}")
    except BaseException:
        print("GenomeVault package not found")
    sys.exit(1)

print("\n" + "=" * 50)
print("All imports successful! Ready to run benchmarks.")
print("=" * 50)

# Simple test
print("\nRunning simple test...")

try:
    # Create encoder
    encoder = GenomicEncoder(dimension=1000)
    print("✓ Created GenomicEncoder")

    # Test encoding
    hv = encoder.encode_variant("chr1", 12345, "A", "G")
    print(f"✓ Encoded variant, hypervector shape: {hv.shape}")

    print("\n✅ Basic functionality working!")

except Exception as e:
    print(f"\n❌ Error during test: {e}")
    import traceback

    traceback.print_exc()

print("\nTo run full benchmarks, ensure all files are properly formatted.")
