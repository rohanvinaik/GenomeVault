"""
HDC Error Tuning with PIR Batching - Complete Example
Demonstrates the full pipeline from accuracy dial to batched PIR queries
"""

import asyncio
import time
from typing import Dict, List

import numpy as np
import torch

from genomevault.hypervector.error_handling import ErrorBudgetAllocator
from genomevault.pir.client import BatchedPIRQueryBuilder, GenomicQuery, PIRClient, QueryType


async def main():
    """
    Demonstrate the complete error-tuned query pipeline
    """
    print("=== HDC Error Tuning with PIR Batching Demo ===\n")

    # Step 1: User specifies accuracy requirements
    print("1. User Accuracy Requirements:")
    epsilon = 0.01  # 1% relative error
    delta_exp = 20  # 1 in 2^20 failure probability
    print(f"   - Allowed error: ±{epsilon*100}%")
    print(f"   - Confidence: 1 in {2**delta_exp:,} chance of failure")
    print(f"   - ECC enabled: Yes (3-block XOR parity)")

    # Step 2: System plans error budget
    print("\n2. Error Budget Planning:")
    allocator = ErrorBudgetAllocator(dim_cap=150000)
    budget = allocator.plan_budget(
        epsilon=epsilon,
        delta_exp=delta_exp,
        ecc_enabled=True,
        repeat_cap=None,  # AUTO mode
    )

    print(f"   - Dimension: {budget.dimension:,}")
    print(f"   - Parity groups: {budget.parity_g}")
    print(f"   - Repeats needed: {budget.repeats}")
    print(f"   - Estimated latency: {allocator.estimate_latency(budget)}ms")
    print(f"   - Estimated bandwidth: {allocator.estimate_bandwidth(budget)}MB")

    # Step 3: Initialize PIR infrastructure (mock)
    print("\n3. Initializing PIR Infrastructure:")

    # Mock PIR servers
    servers = [
        "pir-server-1.genomevault.io:50051",
        "pir-server-2.genomevault.io:50052",
        "pir-server-3.genomevault.io:50053",
        "pir-server-4.genomevault.io:50054",
        "pir-server-5.genomevault.io:50055",
    ]

    database_size = 10_000_000  # 10M variants

    # Create mock PIR client
    pir_client = MockPIRClient(servers, database_size)

    # Mock index mapping (in production, loaded from index files)
    index_mapping = create_mock_index_mapping()

    # Create batched query builder
    query_builder = BatchedPIRQueryBuilder(pir_client, index_mapping)

    print(f"   - Connected to {len(servers)} PIR servers")
    print(f"   - Database size: {database_size:,} entries")
    print(f"   - Privacy guarantee: {pir_client.calculate_privacy_guarantee(3):.2e}")

    # Step 4: Build and execute query
    print("\n4. Executing Privacy-Preserving Query:")

    # User wants to query BRCA1 variant
    genomic_query = GenomicQuery(
        query_type=QueryType.VARIANT_LOOKUP,
        parameters={
            "chromosome": "chr17",
            "position": 43106487,  # BRCA1 pathogenic variant
            "ref_allele": "A",
            "alt_allele": "G",
        },
    )

    print(
        f"   Query: {genomic_query.parameters['chromosome']}:{genomic_query.parameters['position']} "
    )
    print(
        f"         {genomic_query.parameters['ref_allele']}>{genomic_query.parameters['alt_allele']}"
    )

    # Execute with progress tracking
    start_time = time.time()

    # Simulate streaming execution with progress
    print("\n   Executing batched PIR queries:")
    batched_query = query_builder.build_repeat_batch(budget, genomic_query)

    results = []
    async for idx, result in query_builder.execute_streaming_batch(batched_query):
        results.append(result)
        progress = (idx + 1) / budget.repeats * 100
        print(
            f"   [{progress:3.0f}%] Completed repeat {idx + 1}/{budget.repeats}",
            end="\r",
        )

    print()  # New line after progress

    # Step 5: Aggregate results
    print("\n5. Aggregating Results:")

    # Calculate median and error
    values = [r["allele_frequency"] for r in results]
    median_freq = np.median(values)
    median_error = np.median(np.abs(np.array(values) - median_freq))

    print(f"   - Individual results: {values[:5]}... (showing first 5)")
    print(f"   - Median frequency: {median_freq:.4f}")
    print(f"   - Median absolute deviation: {median_error:.6f}")
    print(f"   - Error within bound: {median_error <= epsilon} ✓")

    # Step 6: Generate proof (mock)
    print("\n6. Zero-Knowledge Proof Generation:")
    proof_metadata = {
        "query_type": "variant_lookup",
        "repeats": budget.repeats,
        "median_result": median_freq,
        "error_bound": epsilon,
        "error_achieved": median_error,
        "timestamp": time.time(),
    }

    proof_hash = generate_mock_proof(proof_metadata)
    print(f"   - Proof generated: ipfs://{proof_hash}")
    print(f"   - Proof verifies: median error ≤ ε ✓")

    # Step 7: Final results
    elapsed_time = (time.time() - start_time) * 1000

    print("\n7. Query Complete:")
    print(f"   - Result: Allele frequency = {median_freq:.4f} ± {epsilon*100}%")
    print(f"   - Confidence: {budget.confidence}")
    print(f"   - Total time: {elapsed_time:.0f}ms")
    print(f"   - PIR queries used: {budget.repeats}")

    # Demonstrate different accuracy levels
    print("\n=== Accuracy vs Performance Trade-offs ===")
    print("\nEpsilon | Delta    | Dimension | Repeats | Latency | Bandwidth")
    print("--------|----------|-----------|---------|---------|----------")

    for eps in [0.1, 0.05, 0.01, 0.005]:
        for delta_e in [10, 15, 20]:
            b = allocator.plan_budget(eps, delta_e, ecc_enabled=True)
            lat = allocator.estimate_latency(b)
            bw = allocator.estimate_bandwidth(b)
            print(
                f"{eps:7.3f} | 2^-{delta_e:<5} | {b.dimension:>9,} | {b.repeats:>7} | {lat:>6}ms | {bw:>7.1f}MB"
            )


class MockPIRClient(PIRClient):
    """Mock PIR client for demonstration"""

    def __init__(self, servers: list[str], database_size: int):
        # Don't call super().__init__ to avoid real connection
        self.server_urls = servers
        self.database_size = database_size
        self.threshold = 2

    async def execute_query(self, query):
        """Simulate PIR query execution"""
        # Simulate network delay
        await asyncio.sleep(0.01 + np.random.exponential(0.005))

        # Return mock genomic data
        true_frequency = 0.0123  # True allele frequency
        noise = np.random.normal(0, 0.0005)  # Add realistic noise

        return {
            "variant_id": "rs80357906",
            "gene": "BRCA1",
            "allele_frequency": true_frequency + noise,
            "clinical_significance": "Pathogenic",
            "consequence": "missense_variant",
        }

    def create_query(self, db_index: int, seed: int = None):
        """Create mock query"""
        from genomevault.pir.client.pir_client import PIRQuery

        return PIRQuery(indices=[db_index], seed=seed, metadata={"mock": True})

    def decode_response(self, response_data, response_type="genomic"):
        """Decode mock response"""
        return response_data

    def calculate_privacy_guarantee(self, num_servers: int) -> float:
        """Calculate privacy guarantee"""
        honesty_prob = 0.95
        return (1 - honesty_prob) ** num_servers


def create_mock_index_mapping() -> dict[str, dict]:
    """Create mock index mapping for demo"""
    return {
        "variants": {
            "chr17:43106487:A:G": 8234567,  # BRCA1 variant
            "chr13:32914437:T:C": 5123456,  # BRCA2 variant
            # ... more variants
        },
        "positions": {
            "chr17:43106487": [8234567],
            "chr13:32914437": [5123456],
        },
        "genes": {
            "BRCA1": {
                "chromosome": "chr17",
                "start": 43044295,
                "end": 43125483,
            },
            "BRCA2": {
                "chromosome": "chr13",
                "start": 32889611,
                "end": 32973805,
            },
        },
    }


def generate_mock_proof(metadata: dict) -> str:
    """Generate mock proof hash"""
    import hashlib

    proof_str = str(metadata)
    return hashlib.sha256(proof_str.encode()).hexdigest()[:32]


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
