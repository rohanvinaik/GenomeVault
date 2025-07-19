#!/usr/bin/env python3
"""
Test the current state after reverting the optional imports
"""

print("=" * 70)
print("TESTING CURRENT STATE - POST REVERT")
print("=" * 70)

print("\n1. Testing basic Python imports...")
try:
    import os
    import sys
    print("✅ Standard library imports work")
except Exception as e:
    print(f"❌ Standard library error: {e}")

print("\n2. Testing the fixed import path...")
try:
    # This should work since we fixed the relative import
    import zk_proofs.circuits.biological.variant as variant_module
    print("✅ Module found at correct path")
except ImportError as e:
    print(f"❌ Import error: {e}")
    
print("\n3. Checking what's actually failing...")
try:
    from zk_proofs.circuits.biological.variant import VariantPresenceCircuit
    print("✅ Import successful!")
except ImportError as e:
    print(f"❌ Import failed with: {e}")
    # Let's trace the import chain
    print("\n   Tracing import chain:")
    try:
        import zk_proofs
        print("   ✅ zk_proofs package found")
    except:
        print("   ❌ zk_proofs package not found")
        
    try:
        import zk_proofs.circuits
        print("   ✅ zk_proofs.circuits found")
    except Exception as e:
        print(f"   ❌ zk_proofs.circuits failed: {e}")
        
    try:
        import zk_proofs.circuits.biological
        print("   ✅ zk_proofs.circuits.biological found")
    except Exception as e:
        print(f"   ❌ zk_proofs.circuits.biological failed: {e}")

print("\n4. Checking which dependencies are missing...")
dependencies = {
    'yaml': 'PyYAML',
    'cryptography': 'cryptography',
    'nacl': 'PyNaCl', 
    'structlog': 'structlog',
    'numpy': 'numpy',
    'torch': 'torch'
}

missing = []
for module, package in dependencies.items():
    try:
        __import__(module)
        print(f"   ✅ {package} is installed")
    except ImportError:
        print(f"   ❌ {package} is NOT installed")
        missing.append(package)

if missing:
    print(f"\n5. Missing packages: {', '.join(missing)}")
    print("\nTo install missing packages:")
    print(f"   pip install {' '.join(missing)}")
else:
    print("\n5. All dependencies are installed!")

print("\n" + "=" * 70)
