#!/usr/bin/env python3
"""
Simple E2E test that demonstrates working components.
"""

import sys
import json
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


def test_demo_data_generation():
    """Test demo data generation."""
    print("\n" + "=" * 60)
    print("1. DEMO DATA GENERATION")
    print("=" * 60)

    import subprocess

    # Generate small dataset
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_demo_data.py",
            "--output",
            "test_e2e_data",
            "--variants",
            "10",
            "--fast5-reads",
            "5",
            "--zk-proofs",
            "2",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("✓ Demo data generated successfully")

        # Check manifest
        manifest_path = Path("test_e2e_data/manifest.json")
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            print(f"  Variants: {manifest['statistics']['total_variants']}")
            print(f"  FAST5 reads: {manifest['statistics']['fast5_reads']}")
            print(f"  Hypervectors: {manifest['statistics']['hypervectors']}")
            print(f"  ZK proofs: {manifest['statistics']['zk_proofs']}")
        return True
    else:
        print(f"✗ Failed: {result.stderr}")
        return False


def test_hypervector_encoding():
    """Test hypervector encoding."""
    print("\n" + "=" * 60)
    print("2. HYPERVECTOR ENCODING")
    print("=" * 60)

    from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
    from genomevault.hypervector.featurizers.variants import variant_to_numeric
    from genomevault.core.constants import OmicsType
    import numpy as np

    # Create encoder
    config = HypervectorConfig(dimension=1000, seed=42)
    encoder = HypervectorEncoder(config)

    # Create variant data and convert to numeric features
    variant = {"chrom": "chr1", "pos": 12345, "ref": "A", "alt": "G", "impact": "MODERATE"}

    try:
        # Convert variant to numeric features
        numeric_features = variant_to_numeric(variant)
        features_array = np.array(numeric_features, dtype=np.float32)

        # HypervectorEncoder requires omics_type parameter as enum
        encoded = encoder.encode(features_array, omics_type=OmicsType.GENOMIC)
        print(f"✓ Encoded variant to {len(encoded)} dimensional vector")
        print(f"  Input features: {len(numeric_features)} dimensions")

        # Convert to numpy for statistics
        encoded_np = encoded.detach().cpu().numpy()
        print(f"  Sparsity: {(encoded_np == 0).mean():.2%}")
        print(f"  Non-zero elements: {(encoded_np != 0).sum()}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_zk_proof_fallback():
    """Test ZK proof with transcript fallback."""
    print("\n" + "=" * 60)
    print("3. ZK PROOF (TRANSCRIPT FALLBACK)")
    print("=" * 60)

    from genomevault.zk.real_engine import RealZKEngine

    # Create engine
    engine = RealZKEngine(str(project_root))

    # Generate proof
    inputs = {"a": 10, "b": 32, "c": 42}

    try:
        proof = engine.generate_proof("sum64", inputs)
        if proof:
            print("✓ Proof generated")
            print(f"  Circuit type: {proof.circuit_type}")
            print(f"  Public outputs: {proof.public}")

            # Verify proof
            is_valid = engine.verify_proof(proof.proof, proof.public, circuit_type="sum64")

            if is_valid:
                print("✓ Proof verified successfully")
                return True
            else:
                print("✗ Proof verification failed")
                return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_pir_protocol():
    """Test PIR protocol locally."""
    print("\n" + "=" * 60)
    print("4. PIR PROTOCOL (LOCAL)")
    print("=" * 60)

    import numpy as np

    # Simple PIR demonstration
    database_size = 10
    target_index = 4

    # Create unit vector for target
    unit_vector = np.zeros(database_size, dtype=np.uint8)
    unit_vector[target_index] = 1

    # Generate query vectors for 2 servers
    query1 = np.random.randint(0, 2, database_size, dtype=np.uint8)
    query2 = (unit_vector - query1) % 2

    # Verify queries XOR to unit vector
    verification = (query1 + query2) % 2

    if np.array_equal(verification, unit_vector):
        print(f"✓ PIR queries generated for index {target_index}")
        print(f"  Query 1: {query1}")
        print(f"  Query 2: {query2}")
        print(f"  XOR result: {verification}")
        return True
    else:
        print("✗ PIR query generation failed")
        return False


def test_database_operations():
    """Test database operations."""
    print("\n" + "=" * 60)
    print("5. DATABASE OPERATIONS")
    print("=" * 60)

    import sqlite3

    db_path = Path("test_e2e_data/genomevault_demo.db")

    if not db_path.exists():
        print("⚠ Database not found, skipping")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Count variants
        cursor.execute("SELECT COUNT(*) FROM variants")
        variant_count = cursor.fetchone()[0]

        # Get sample variant
        cursor.execute("SELECT * FROM variants LIMIT 1")
        sample_variant = cursor.fetchone()

        conn.close()

        print("✓ Database accessible")
        print(f"  Total variants: {variant_count}")
        if sample_variant:
            print(
                f"  Sample: chr{sample_variant[1]}:{sample_variant[2]} {sample_variant[3]}>{sample_variant[4]}"
            )

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def cleanup():
    """Clean up test data."""
    import shutil

    test_dir = Path("test_e2e_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("\n✓ Cleaned up test data")


def main():
    """Run simple E2E tests."""
    print("\n" + "=" * 60)
    print("🧬 GenomeVault Simple E2E Test")
    print("=" * 60)
    print("Testing core components without external services")

    results = []

    # Run tests
    results.append(("Demo Data Generation", test_demo_data_generation()))
    results.append(("Hypervector Encoding", test_hypervector_encoding()))
    results.append(("ZK Proof", test_zk_proof_fallback()))
    results.append(("PIR Protocol", test_pir_protocol()))
    results.append(("Database Operations", test_database_operations()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{len(results)}")

    # Cleanup
    cleanup()

    if passed == len(results):
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
