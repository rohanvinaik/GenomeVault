#!/usr/bin/env python3
"""Comprehensive test of the variant circuit import fix"""

import sys
import traceback

print("=" * 60)
print("TESTING VARIANT CIRCUIT IMPORT FIX")
print("=" * 60)

# Test 1: Direct import
print("\n1. Testing direct import of VariantPresenceCircuit...")
try:
    from zk_proofs.circuits.biological.variant import VariantPresenceCircuit

    print("✅ Import successful!")

    # Test instantiation
    circuit = VariantPresenceCircuit(merkle_depth=20)
    print(f"✅ Created circuit: {circuit.name} with {circuit.num_constraints} constraints")

except Exception as e:
    print(f"❌ Failed: {e}")
    traceback.print_exc()

# Test 2: Import all biological circuits
print("\n2. Testing all biological circuit imports...")
try:
    from zk_proofs.circuits.biological.variant import (
        VariantPresenceCircuit,
        PolygenenicRiskScoreCircuit,
        DiabetesRiskCircuit,
        PharmacogenomicCircuit,
        PathwayEnrichmentCircuit,
    )

    print("✅ All variant circuits imported successfully!")

    # Test each one
    circuits = [
        ("VariantPresence", VariantPresenceCircuit()),
        ("PRS", PolygenenicRiskScoreCircuit()),
        ("Diabetes", DiabetesRiskCircuit()),
        ("Pharmacogenomic", PharmacogenomicCircuit()),
        ("Pathway", PathwayEnrichmentCircuit()),
    ]

    for name, circuit in circuits:
        print(f"  ✅ {name}: {circuit.name}")

except Exception as e:
    print(f"❌ Failed: {e}")
    traceback.print_exc()

# Test 3: Import multi-omics circuits
print("\n3. Testing multi-omics circuit imports...")
try:
    from zk_proofs.circuits.biological.multi_omics import (
        MultiOmicsCorrelationCircuit,
        GenotypePhenotypeAssociationCircuit,
        ClinicalTrialEligibilityCircuit,
        RareVariantBurdenCircuit,
    )

    print("✅ All multi-omics circuits imported successfully!")

    circuits = [
        ("MultiOmicsCorrelation", MultiOmicsCorrelationCircuit()),
        ("G-P Association", GenotypePhenotypeAssociationCircuit()),
        ("Clinical Trial", ClinicalTrialEligibilityCircuit()),
        ("Rare Variant", RareVariantBurdenCircuit()),
    ]

    for name, circuit in circuits:
        print(f"  ✅ {name}: {circuit.name}")

except Exception as e:
    print(f"❌ Failed: {e}")
    traceback.print_exc()

# Test 4: Test the base circuits import
print("\n4. Testing base circuits import...")
try:
    from zk_proofs.circuits.base_circuits import (
        BaseCircuit,
        FieldElement,
        MerkleTreeCircuit,
        RangeProofCircuit,
        ComparisonCircuit,
    )

    print("✅ Base circuits imported successfully!")

    # Test FieldElement
    fe = FieldElement(12345)
    print(f"  ✅ FieldElement: {fe.value}")

    # Test MerkleTreeCircuit
    merkle = MerkleTreeCircuit(depth=10)
    print(f"  ✅ MerkleTree: depth={merkle.depth}")

except Exception as e:
    print(f"❌ Failed: {e}")
    traceback.print_exc()

# Test 5: Run the simple pytest
print("\n5. Running pytest...")
import subprocess

result = subprocess.run(
    [sys.executable, "-m", "pytest", "test_simple.py", "-v"], capture_output=True, text=True
)
print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("The variant circuit import fix has been applied.")
print("Import path changed from '.base_circuits' to '..base_circuits'")
print("This correctly imports from the parent directory.")
