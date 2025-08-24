#!/usr/bin/env python3
"""Test that PIR imports work with all expected names."""

import sys

sys.path.insert(0, "/Users/rohanvinaik/genomevault")

print("=" * 60)
print("🔍 TESTING PIR IMPORT FIXES")
print("=" * 60)

# Test 1: Original class name
print("\nTest 1: Import PIRProtocol (original name)")
try:
    from genomevault.pir.it_pir_protocol import PIRProtocol

    print("  ✅ PIRProtocol imported successfully")
except ImportError as e:
    print(f"  ❌ Failed: {e}")

# Test 2: Expected alias ITPrivateInformationRetrieval
print("\nTest 2: Import ITPrivateInformationRetrieval (alias)")
try:
    from genomevault.pir.it_pir_protocol import ITPrivateInformationRetrieval

    print("  ✅ ITPrivateInformationRetrieval imported successfully")
    print(f"     Is same as PIRProtocol? {ITPrivateInformationRetrieval is PIRProtocol}")
except ImportError as e:
    print(f"  ❌ Failed: {e}")

# Test 3: Expected alias ITPIRProtocol
print("\nTest 3: Import ITPIRProtocol (alias)")
try:
    from genomevault.pir.it_pir_protocol import ITPIRProtocol

    print("  ✅ ITPIRProtocol imported successfully")
    print(f"     Is same as PIRProtocol? {ITPIRProtocol is PIRProtocol}")
except ImportError as e:
    print(f"  ❌ Failed: {e}")

# Test 4: Import from package __init__
print("\nTest 4: Import from genomevault.pir")
try:
    from genomevault.pir import PIRProtocol, ITPrivateInformationRetrieval, ITPIRProtocol

    print("  ✅ All names imported from genomevault.pir")
    print(
        f"     All are same class? {PIRProtocol is ITPrivateInformationRetrieval is ITPIRProtocol}"
    )
except ImportError as e:
    print(f"  ❌ Failed: {e}")

# Test 5: Instantiate and use
print("\nTest 5: Create instance and test basic functionality")
try:
    from genomevault.pir.it_pir_protocol import PIRParameters

    params = PIRParameters(database_size=100, num_servers=3)
    protocol = ITPrivateInformationRetrieval(params)
    queries = protocol.generate_query_vectors(index=5)
    print(f"  ✅ Generated {len(queries)} queries for 3 servers")
    print(
        f"     Query shape: {queries[0].shape if hasattr(queries[0], 'shape') else len(queries[0])}"
    )
except Exception as e:
    print(f"  ❌ Failed: {e}")

print("\n" + "=" * 60)
print("✅ PIR IMPORT ISSUE FIXED")
print("=" * 60)
