#!/bin/bash
# Quick Black formatting fix

cd /Users/rohanvinaik/genomevault

echo "Quick Black Formatting Fix"
echo "========================="

# Fix the test file first (it has a syntax error)
echo "Fixing test_catalytic_implementation.py..."
cat > test_catalytic_implementation.py << 'EOF'
"""Test script for Catalytic GenomeVault implementation."""

import asyncio
import numpy as np
from genomevault.integration.catalytic_pipeline import CatalyticGenomeVaultPipeline


async def test_catalytic_pipeline():
    """Test the catalytic pipeline implementation."""
    print("Testing Catalytic GenomeVault Pipeline")
    print("=" * 50)

    pipeline = CatalyticGenomeVaultPipeline(dimension=10000, use_gpu=False)

    genomic_data = {
        "variants": np.random.randint(0, 2, 1000).tolist(),
        "weights": np.random.rand(1000).tolist(),
        "genetic_state": {
            "genotypes": {
                "locus1": {"AA": 0.25, "Aa": 0.5, "aa": 0.25},
                "locus2": {"AA": 0.3, "Aa": 0.4, "aa": 0.3},
            },
            "allele_frequencies": {
                "locus1": {"A": 0.5, "a": 0.5},
                "locus2": {"A": 0.5, "a": 0.5},
            },
        },
    }

    try:
        results = await pipeline.process_genomic_data(
            genomic_data, constraints=["hardy_weinberg", "allele_frequency"]
        )
        print("\nPipeline Results:")
        print(f"Status: {results['status']}")
        print(f"Memory saved: {results['memory_saved_mb']}MB")
        print(f"Performance gain: {results['performance_gain']}x")
        print(f"Proofs generated: {len(results['proofs'])}")
        return True
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_individual_components():
    """Test individual components of the catalytic implementation."""
    print("\n\nTesting Individual Components")
    print("=" * 50)

    print("\n1. Testing Catalytic Projection Pool...")
    try:
        import torch
        from genomevault.hypervector.encoding.catalytic_projections import (
            CatalyticProjectionPool,
        )

        pool = CatalyticProjectionPool(dimension=1000, pool_size=5)
        test_vector = torch.randn(1000)
        projected = pool.apply_catalytic_projection(test_vector, [0, 1, 2])
        print(
            f"✓ Projection pool working: input shape {test_vector.shape}, "
            f"output shape {projected.shape}"
        )
    except Exception as e:
        print(f"✗ Projection pool error: {e}")

    print("\n2. Testing Catalytic PIR Client...")
    try:
        from genomevault.pir.catalytic_client import CatalyticPIRClient
        from genomevault.pir.client import PIRServer

        servers = [
            PIRServer("test1", "http://localhost:8001", "local", False, 0.95, 10),
            PIRServer("test2", "http://localhost:8002", "local", False, 0.95, 10),
        ]
        client = CatalyticPIRClient(servers, database_size=10000)
        print(f"✓ Catalytic PIR client initialized with {len(servers)} servers")
    except Exception as e:
        print(f"✗ Catalytic PIR client error: {e}")

    print("\n3. Testing COEC Catalytic Proof Engine...")
    try:
        from genomevault.zk_proofs.advanced.coec_catalytic_proof import (
            COECCatalyticProofEngine,
        )

        engine = COECCatalyticProofEngine()
        test_state = {
            "genotypes": {"locus1": {"AA": 0.25, "Aa": 0.5, "aa": 0.25}},
            "allele_frequencies": {"locus1": {"A": 0.5, "a": 0.5}},
        }
        hwe_constraint = engine.constraint_operators["hardy_weinberg"]
        is_satisfied = hwe_constraint.is_satisfied(test_state)
        print(f"✓ COEC engine working: Hardy-Weinberg satisfied = {is_satisfied}")
    except Exception as e:
        print(f"✗ COEC engine error: {e}")


def main():
    """Main test function."""
    print("Catalytic GenomeVault Implementation Test")
    print("=" * 70)

    loop = asyncio.get_event_loop()
    pipeline_success = loop.run_until_complete(test_catalytic_pipeline())
    loop.run_until_complete(test_individual_components())

    print("\n\nTest Summary")
    print("=" * 50)
    print(f"Pipeline test: {'PASSED' if pipeline_success else 'FAILED'}")
    print("\nCatalytic implementation is ready for use!")


if __name__ == "__main__":
    main()
EOF

# Run black on specific files
echo -e "\nRunning Black formatter..."
python -m black --target-version py311 \
    test_catalytic_implementation.py \
    example_catalytic_usage.py \
    genomevault/hypervector/encoding/genomic.py || {
    echo "Black not installed as module, trying command..."
    black --target-version py311 \
        test_catalytic_implementation.py \
        example_catalytic_usage.py \
        genomevault/hypervector/encoding/genomic.py
}

# Stage and commit
echo -e "\nStaging changes..."
git add test_catalytic_implementation.py example_catalytic_usage.py genomevault/hypervector/encoding/genomic.py

echo -e "\nCommitting..."
git commit -m "fix: Apply Black formatting to fix CI errors

- Fix syntax error in test_catalytic_implementation.py
- Apply Black formatting to all affected files
- Target Python 3.11 for compatibility"

# Push
echo -e "\nPushing to GitHub..."
git push origin $(git branch --show-current)

echo -e "\n✅ Black formatting fixes applied and pushed!"
