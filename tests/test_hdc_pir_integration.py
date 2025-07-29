"""
Test for HDC Error Handling with PIR Batch Integration
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch

from genomevault.hypervector.error_handling import (
    AdaptiveHDCEncoder,
    ECCEncoderMixin,
    ErrorBudgetAllocator,
)
from genomevault.pir.client import BatchedPIRQueryBuilder, GenomicQuery, PIRClient, QueryType


class TestBatchedPIRIntegration:
    """Test the integration between error budget and PIR batching"""

    def test_error_budget_allocation(self):
        """Test error budget allocation logic"""
        allocator = ErrorBudgetAllocator(dim_cap=100000)

        # Test basic allocation
        budget = allocator.plan_budget(epsilon=0.01, delta_exp=15, ecc_enabled=True)

        assert budget.dimension > 0
        assert budget.dimension <= 100000
        assert budget.repeats > 0
        assert budget.epsilon == 0.01
        assert budget.delta_exp == 15
        assert budget.ecc_enabled

        # Test without ECC
        budget_no_ecc = allocator.plan_budget(epsilon=0.01, delta_exp=15, ecc_enabled=False)

        # Without ECC, should need more dimension or repeats
        assert budget_no_ecc.dimension > budget.dimension or budget_no_ecc.repeats > budget.repeats

    def test_ecc_encoder(self):
        """Test ECC encoding and decoding"""
        encoder = ECCEncoderMixin(base_dimension=1000, parity_g=3)

        # Test with binary vector
        binary_vec = torch.randint(0, 2, (1000,)).bool()
        encoded = encoder.encode_with_ecc(binary_vec.float())

        # Check dimension expansion
        expected_dim = int(1000 * 4 / 3)  # g+1/g expansion
        assert len(encoded) == encoder.expanded_dimension

        # Test decoding without errors
        decoded, errors = encoder.decode_with_ecc(encoded)
        assert len(decoded) == 1000
        assert errors == 0

        # Test with injected error
        encoded_with_error = encoded.clone()
        encoded_with_error[10] = -encoded_with_error[10]  # Flip a bit

        decoded_with_error, errors = encoder.decode_with_ecc(encoded_with_error)
        assert errors > 0  # Should detect error

    @pytest.mark.asyncio
    async def test_batched_query_builder(self):
        """Test batched PIR query builder"""
        # Mock PIR client
        mock_client = Mock(spec=PIRClient)
        mock_client.create_query = Mock(
            side_effect=lambda idx, seed=None: Mock(indices=[idx], seed=seed)
        )
        mock_client.execute_query = AsyncMock(return_value={"value": 42.0})
        mock_client.decode_response = Mock(return_value=42.0)

        # Mock index mapping
        index_mapping = {
            "variants": {"chr1:100000:A:G": 123},
            "positions": {},
            "genes": {},
        }

        builder = BatchedPIRQueryBuilder(mock_client, index_mapping)

        # Create test query
        query = GenomicQuery(
            query_type=QueryType.VARIANT_LOOKUP,
            parameters={
                "chromosome": "chr1",
                "position": 100000,
                "ref_allele": "A",
                "alt_allele": "G",
            },
        )

        # Create budget
        allocator = ErrorBudgetAllocator()
        budget = allocator.plan_budget(epsilon=0.01, delta_exp=10)

        # Build batch
        batch = builder.build_repeat_batch(budget, query)

        assert len(batch.queries) == budget.repeats
        assert batch.error_threshold == 0.01

        # Test that each query has a different seed
        seeds = [q.seed for q in batch.queries]
        assert len(set(seeds)) == len(seeds)  # All unique

    @pytest.mark.asyncio
    async def test_streaming_execution(self):
        """Test streaming batch execution"""
        # Mock PIR client
        mock_client = Mock(spec=PIRClient)
        mock_client.execute_query = AsyncMock(
            side_effect=lambda q: {"value": 42.0 + np.random.normal(0, 0.1)}  # Add some noise
        )
        mock_client.decode_response = Mock(side_effect=lambda r, t: r["value"])

        index_mapping = {
            "variants": {"chr1:100000:A:G": 123},
            "positions": {},
            "genes": {},
        }

        builder = BatchedPIRQueryBuilder(mock_client, index_mapping)

        query = GenomicQuery(
            query_type=QueryType.VARIANT_LOOKUP,
            parameters={
                "chromosome": "chr1",
                "position": 100000,
                "ref_allele": "A",
                "alt_allele": "G",
            },
        )

        allocator = ErrorBudgetAllocator()
        budget = allocator.plan_budget(epsilon=0.01, delta_exp=10, repeat_cap=5)

        batch = builder.build_repeat_batch(budget, query)

        # Execute with streaming
        results_received = []
        async for idx, result in builder.execute_streaming_batch(batch):
            results_received.append((idx, result))

        assert len(results_received) == 5

        # Check results are in expected range
        values = [r for _, r in results_received]
        median_val = np.median(values)
        assert abs(median_val - 42.0) < 0.5  # Should be close to true value

    def test_adaptive_hdc_encoder(self):
        """Test adaptive HDC encoder with error budget"""
        encoder = AdaptiveHDCEncoder(dimension=10000)

        # Mock variants
        variants = [
            {"chromosome": "chr1", "position": 100000, "ref": "A", "alt": "G"},
            {"chromosome": "chr2", "position": 200000, "ref": "C", "alt": "T"},
        ]

        # Create budget
        allocator = ErrorBudgetAllocator()
        budget = allocator.plan_budget(epsilon=0.01, delta_exp=15, repeat_cap=3)

        # Encode with budget
        encoded_vec, metadata = encoder.encode_with_budget(variants, budget)

        assert encoded_vec is not None
        assert metadata["budget"] == budget
        assert len(metadata["proofs"]) == budget.repeats
        assert metadata["error_within_bound"] is not None

        # Check that dimension adjustment worked
        if budget.dimension != 10000:
            assert encoder.dimension == budget.dimension

    @pytest.mark.asyncio
    async def test_early_termination(self):
        """Test early termination when error is low"""
        mock_client = Mock(spec=PIRClient)

        # First few results are very consistent
        result_values = [42.0, 42.01, 41.99, 42.0, 42.0, 50.0, 60.0, 70.0]
        call_count = 0

        async def mock_execute(q):
            nonlocal call_count
            if call_count < len(result_values):
                val = result_values[call_count]
                call_count += 1
                return {"value": val}
            return {"value": 100.0}

        mock_client.execute_query = mock_execute
        mock_client.decode_response = Mock(side_effect=lambda r, t: r["value"])

        index_mapping = {
            "variants": {"chr1:100000:A:G": 123},
            "positions": {},
            "genes": {},
        }

        builder = BatchedPIRQueryBuilder(mock_client, index_mapping)

        query = GenomicQuery(
            query_type=QueryType.VARIANT_LOOKUP,
            parameters={
                "chromosome": "chr1",
                "position": 100000,
                "ref_allele": "A",
                "alt_allele": "G",
            },
        )

        # Large budget but should terminate early
        allocator = ErrorBudgetAllocator()
        budget = allocator.plan_budget(epsilon=0.1, delta_exp=10, repeat_cap=10)

        batch = builder.build_repeat_batch(budget, query)

        # Execute - should terminate early
        results = []
        async for idx, result in builder.execute_streaming_batch(batch):
            results.append(result)

        # Should have terminated before all 10 repeats
        assert len(results) < 10
        assert len(results) >= 6  # At least 60% as per early termination logic


if __name__ == "__main__":
    # Run basic tests
    test = TestBatchedPIRIntegration()

    print("Testing error budget allocation...")
    test.test_error_budget_allocation()
    print("✓ Error budget allocation works correctly")

    print("\nTesting ECC encoder...")
    test.test_ecc_encoder()
    print("✓ ECC encoding/decoding works correctly")

    print("\nTesting adaptive HDC encoder...")
    test.test_adaptive_hdc_encoder()
    print("✓ Adaptive HDC encoder works correctly")

    print("\nRunning async tests...")
    asyncio.run(test.test_batched_query_builder())
    print("✓ Batched query builder works correctly")

    asyncio.run(test.test_streaming_execution())
    print("✓ Streaming execution works correctly")

    asyncio.run(test.test_early_termination())
    print("✓ Early termination works correctly")

    print("\n✅ All tests passed!")
