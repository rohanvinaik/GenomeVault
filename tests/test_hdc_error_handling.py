from typing import Any, Dict

"""
Test cases for HDC error handling with uncertainty tuning
"""

import pytest
import torch

from genomevault.hypervector.error_handling import (
    AdaptiveHDCEncoder,
    ECCEncoderMixin,
    ErrorBudget,
    ErrorBudgetAllocator,
)


class TestErrorBudgetAllocator:
    """Test error budget allocation"""


    def test_basic_allocation(self) -> None:
    """Test basic budget allocation"""
        allocator = ErrorBudgetAllocator()
        budget = allocator.plan_budget(epsilon=0.01, delta_exp=15)

        assert budget.dimension > 0
        assert budget.repeats > 0
        assert budget.epsilon == 0.01
        assert budget.delta_exp == 15


    def test_ecc_impact(self) -> None:
    """Test that ECC reduces required dimension"""
        allocator = ErrorBudgetAllocator()

        budget_with_ecc = allocator.plan_budget(epsilon=0.01, delta_exp=15, ecc_enabled=True)
        budget_without_ecc = allocator.plan_budget(epsilon=0.01, delta_exp=15, ecc_enabled=False)

        # ECC should allow smaller dimensions for same error
        assert budget_with_ecc.dimension <= budget_without_ecc.dimension


    def test_dimension_capping(self) -> None:
    """Test dimension capping with repeat compensation"""
        allocator = ErrorBudgetAllocator(dim_cap=50000)

        # Request very high accuracy (would need huge dimension)
        budget = allocator.plan_budget(epsilon=0.0001, delta_exp=30)

        assert budget.dimension == 50000  # Should be capped
        assert budget.repeats > 100  # Should compensate with more repeats


    def test_latency_estimation(self) -> None:
    """Test latency estimation"""
        allocator = ErrorBudgetAllocator()
        budget = allocator.plan_budget(epsilon=0.01, delta_exp=15)

        latency = allocator.estimate_latency(budget)
        assert latency > 0
        assert latency < 5000  # Should be under 5 seconds


    def test_bandwidth_estimation(self) -> None:
    """Test bandwidth estimation"""
        allocator = ErrorBudgetAllocator()
        budget = allocator.plan_budget(epsilon=0.01, delta_exp=15)

        bandwidth = allocator.estimate_bandwidth(budget)
        assert bandwidth > 0
        assert bandwidth < 100  # Should be under 100MB


class TestECCEncoder:
    """Test error correcting code functionality"""


    def test_encode_decode(self) -> None:
    """Test basic encode/decode cycle"""
        encoder = ECCEncoderMixin(base_dimension=1000, parity_g=3)

        # Create test vector
        original = torch.randn(1000)

        # Encode
        encoded = encoder.encode_with_ecc(original)
        assert len(encoded) > len(original)  # Should be expanded

        # Decode
        decoded, errors = encoder.decode_with_ecc(encoded)
        assert len(decoded) == len(original)
        assert errors == 0  # No errors in clean encode/decode

        # Verify similarity
        similarity = torch.cosine_similarity(original, decoded, dim=0)
        assert similarity > 0.99


    def test_error_correction(self) -> None:
    """Test error detection capability"""
        encoder = ECCEncoderMixin(base_dimension=100, parity_g=3)

        original = torch.randn(100)
        encoded = encoder.encode_with_ecc(original)

        # Introduce errors
        corrupted = encoded.clone()
        corrupted[10] += 5.0  # Large error

        # Decode should detect error
        decoded, errors = encoder.decode_with_ecc(corrupted)
        assert errors > 0  # Should detect the error


    def test_dimension_handling(self) -> None:
    """Test handling of different dimensions"""
        encoder = ECCEncoderMixin(base_dimension=123, parity_g=4)

        # Test with non-divisible dimension
        original = torch.randn(123)
        encoded = encoder.encode_with_ecc(original)
        decoded, _ = encoder.decode_with_ecc(encoded)

        assert len(decoded) == 123
        similarity = torch.cosine_similarity(original, decoded, dim=0)
        assert similarity > 0.99


class TestAdaptiveEncoder:
    """Test adaptive HDC encoder with error handling"""


    def test_encode_with_budget(self) -> None:
    """Test encoding with error budget"""
        encoder = AdaptiveHDCEncoder(dimension=10000)

        # Create test variants
        variants = [
            {"chromosome": "chr1", "position": 100000, "ref": "A", "alt": "G", "type": "SNP"},
            {"chromosome": "chr2", "position": 200000, "ref": "C", "alt": "T", "type": "SNP"},
        ]

        # Create budget
        budget = ErrorBudget(
            dimension=10000, parity_g=3, repeats=5, epsilon=0.01, delta_exp=15, ecc_enabled=True
        )

        # Encode
        encoded_vector, metadata = encoder.encode_with_budget(variants, budget)

        assert encoded_vector is not None
        assert metadata["error_within_bound"]
        assert len(metadata["proofs"]) == budget.repeats


    def test_median_aggregation(self) -> None:
    """Test that median aggregation reduces variance"""
        encoder = AdaptiveHDCEncoder(dimension=1000)

        variants = [
            {"chromosome": "chr1", "position": 100000, "ref": "A", "alt": "G", "type": "SNP"}
        ]

        # Single repeat
        budget_single = ErrorBudget(
            dimension=1000, parity_g=3, repeats=1, epsilon=0.01, delta_exp=10, ecc_enabled=False
        )

        # Multiple repeats
        budget_multi = ErrorBudget(
            dimension=1000, parity_g=3, repeats=20, epsilon=0.01, delta_exp=10, ecc_enabled=False
        )

        _, meta_single = encoder.encode_with_budget(variants, budget_single)
        _, meta_multi = encoder.encode_with_budget(variants, budget_multi)

        # Multiple repeats should have lower error
        assert meta_multi["median_error"] < meta_single["median_error"]


    def test_dimension_adaptation(self) -> None:
    """Test dimension adaptation"""
        encoder = AdaptiveHDCEncoder(dimension=5000)

        variants = [
            {"chromosome": "chr1", "position": 100000, "ref": "A", "alt": "G", "type": "SNP"}
        ]

        # Budget with different dimension
        budget = ErrorBudget(
            dimension=10000, parity_g=3, repeats=1, epsilon=0.01, delta_exp=10, ecc_enabled=False
        )

        encoded_vector, _ = encoder.encode_with_budget(variants, budget)

        # Should adapt to budget dimension
        assert encoder.dimension == 10000


class TestErrorBudgetClass:
    """Test ErrorBudget dataclass"""


    def test_properties(self) -> None:
    """Test ErrorBudget properties"""
        budget = ErrorBudget(
            dimension=10000, parity_g=3, repeats=10, epsilon=0.01, delta_exp=20, ecc_enabled=True
        )

        assert budget.delta == 2**-20
        assert budget.confidence == "1 in 1048576"


    def test_confidence_levels(self) -> None:
    """Test different confidence levels"""
        budgets = [
            ErrorBudget(10000, 3, 10, 0.01, 10, True),
            ErrorBudget(10000, 3, 10, 0.01, 15, True),
            ErrorBudget(10000, 3, 10, 0.01, 20, True),
            ErrorBudget(10000, 3, 10, 0.01, 25, True),
        ]

        confidences = [b.confidence for b in budgets]
        expected = ["1 in 1024", "1 in 32768", "1 in 1048576", "1 in 33554432"]

        assert confidences == expected

@pytest.mark.parametrize(
    "epsilon,delta_exp,expected_dim_range",
    [
        (0.1, 10, (1000, 5000)),  # Low accuracy
        (0.01, 15, (10000, 50000)),  # Medium accuracy
        (0.001, 20, (50000, 200000)),  # High accuracy
    ],
)

    def test_dimension_scaling(epsilon, delta_exp, expected_dim_range) -> None:
    """Test dimension scaling with different parameters"""
    allocator = ErrorBudgetAllocator()
    budget = allocator.plan_budget(epsilon=epsilon, delta_exp=delta_exp)

    assert expected_dim_range[0] <= budget.dimension <= expected_dim_range[1]

@pytest.mark.parametrize("parity_g", [2, 3, 4, 5])

    def test_ecc_parity_groups(parity_g) -> None:
    """Test different parity group sizes"""
    encoder = ECCEncoderMixin(base_dimension=1000, parity_g=parity_g)

    original = torch.randn(1000)
    encoded = encoder.encode_with_ecc(original)

    # Check expansion factor
    expected_expansion = (parity_g + 1) / parity_g
    actual_expansion = len(encoded) / len(original)

    assert abs(actual_expansion - expected_expansion) < 0.1
