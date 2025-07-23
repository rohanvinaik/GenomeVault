#!/usr/bin/env python3
"""Quick diagnostic to see what's failing"""

import sys
import traceback

print("🔍 Quick Import Diagnostic")
print("=" * 50)

# Test 1: Can we import torch?
print("\n1. Testing torch import...")
try:
    import torch

    print("✓ Torch version: {torch.__version__}")
except ImportError as e:
    print("✗ Torch not installed: {e}")
    print("  Install with: pip install torch")

# Test 2: Test the specific imports that were failing
print("\n2. Testing specific failing imports...")

# PIR Client
print("\n  Testing PIR Client...")
try:
    sys.path.insert(0, "/Users/rohanvinaik/genomevault")
    from pir.client import PIRClient

    print("  ✓ PIR Client imported")
except Exception as e:
    print("  ✗ PIR Client failed: {e}")
    # Try to see what's in the pir directory
    import os

    pir_path = "/Users/rohanvinaik/genomevault/pir"
    if os.path.exists(pir_path):
        print("  PIR directory contents: {os.listdir(pir_path)}")
        client_path = os.path.join(pir_path, "client.py")
        if os.path.exists(client_path):
            print("  client.py exists, checking first few lines...")
            with open(client_path, "r") as f:
                lines = f.readlines()[:10]
                for i, line in enumerate(lines):
                    print("    {i+1}: {line.rstrip()}")

# ZK Prover
print("\n  Testing ZK Prover...")
try:
    from zk_proofs.prover import ZKProver

    print("  ✓ ZK Prover imported")
except Exception as e:
    print("  ✗ ZK Prover failed: {e}")
    traceback.print_exc()

# Test the config import pattern
print("\n3. Testing config import pattern...")
try:
    from utils.config import get_config

    config = get_config()
    print("  ✓ Config loaded: {type(config)}")
    print("  ✓ Environment: {config.environment}")
except Exception as e:
    print("  ✗ Config failed: {e}")
    traceback.print_exc()

# Test if we have the required dependencies
print("\n4. Checking required dependencies...")
deps = ["numpy", "pydantic", "cryptography", "torch"]
for dep in deps:
    try:
        __import__(dep)
        print("  ✓ {dep} installed")
    except ImportError:
        print("  ✗ {dep} NOT installed")

print("\n" + "=" * 50)
print("Diagnostic complete!")
