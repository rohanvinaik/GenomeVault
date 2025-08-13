import pytest

import numpy as np

from genomevault.local_processing.compression import (

    CompressionTier,
    HDCCompressor,
    SNPCompressor,
    TieredCompressor,
)


class TestCompressionTiers:
    """Test suite for multi-tier compression framework"""

    @pytest.fixture
    def sample_snp_data(self):
        """Generate realistic SNP test data"""
        # Mini tier: ~5,000 most-studied SNPs
        mini_snps = {
            "rs{i}": np.random.choice(["AA", "AT", "TT"], p=[0.25, 0.5, 0.25]) for i in range(5000)
        }

        # Clinical tier: ACMG + PharmGKB variants (~120k)
        clinical_snps = {
            "rs{i}": np.random.choice(
                ["AA", "AT", "TT", "GG", "GC", "CC"],
                p=[0.15, 0.25, 0.15, 0.15, 0.20, 0.10],
            )
            for i in range(120000)
        }

        return {"mini": mini_snps, "clinical": clinical_snps}

    @pytest.fixture
    def sample_hypervector(self):
        """Generate test hypervector data"""
        return {
            "base": np.random.randn(10000),  # 10k dimensions
            "mid": np.random.randn(15000),  # 15k dimensions
            "high": np.random.randn(20000),  # 20k dimensions
        }

    def test_mini_tier_compression_size(self, sample_snp_data):
        """Verify mini tier achieves ~25KB compression"""
        compressor = SNPCompressor(tier=CompressionTier.MINI)
        compressed = compressor.compress(sample_snp_data["mini"])

        # Verify size is within expected range (allowing 10% variance)
        size_kb = len(compressed) / 1024
        assert 22.5 <= size_kb <= 27.5, "Mini tier size {size_kb}KB outside expected 25KB ±10%"

        # Verify lossless compression
        decompressed = compressor.decompress(compressed)
        assert decompressed == sample_snp_data["mini"]

    def test_clinical_tier_compression_size(self, sample_snp_data):
        """Verify clinical tier achieves ~300KB compression"""
        compressor = SNPCompressor(tier=CompressionTier.CLINICAL)
        compressed = compressor.compress(sample_snp_data["clinical"])

        # Verify size is within expected range
        size_kb = len(compressed) / 1024
        assert 270 <= size_kb <= 330, "Clinical tier size {size_kb}KB outside expected 300KB ±10%"

        # Verify key ACMG variants preserved
        decompressed = compressor.decompress(compressed)
        assert len(decompressed) >= 59  # ACMG minimum actionable genes

    def test_hdc_compression_ratio(self, sample_hypervector):
        """Test HDC compression achieves 100-200KB per modality"""
        compressor = HDCCompressor()

        for level, vector in sample_hypervector.items():
            compressed = compressor.compress(vector)
            size_kb = len(compressed) / 1024

            assert (
                100 <= size_kb <= 200
            ), "HDC {level} compression {size_kb}KB outside 100-200KB range"

            # Test reconstruction error is minimal
            decompressed = compressor.decompress(compressed)
            mse = np.mean((vector - decompressed) ** 2)
            assert mse < 1e-6, "Reconstruction error too high: {mse}"

    def test_combined_storage_calculation(self, sample_snp_data, sample_hypervector):
        """Verify S_client = ∑modalities Size_tier formula"""
        tiered = TieredCompressor()

        # Add mini genomics + clinical pharmacogenomics
        tiered.add_modality("genomics", CompressionTier.MINI, sample_snp_data["mini"])
        tiered.add_modality(
            "pharmacogenomics", CompressionTier.CLINICAL, sample_snp_data["clinical"]
        )

        total_size = tiered.get_total_size()
        expected = 25 + 300  # KB

        assert (
            abs(total_size - expected) < expected * 0.1
        ), "Combined size {total_size}KB differs from expected {expected}KB"

    @pytest.mark.parametrize(
        "tier,expected_features,expected_size",
        [
            (CompressionTier.MINI, 5000, 25),
            (CompressionTier.CLINICAL, 120000, 300),
            (CompressionTier.FULL_HDC, "10k-20k dims", 150),
        ],
    )
    def test_tier_specifications(self, tier, expected_features, expected_size):
        """Parameterized test for all tier specifications"""
        compressor = (
            SNPCompressor(tier=tier) if tier != CompressionTier.FULL_HDC else HDCCompressor()
        )

        # Verify tier metadata
        assert compressor.get_feature_count() == expected_features or str(expected_features) in str(
            compressor.get_feature_description()
        )
        assert abs(compressor.estimate_size() - expected_size) < expected_size * 0.15

    def test_compression_determinism(self, sample_snp_data):
        """Ensure compression is deterministic for same input"""
        compressor = SNPCompressor(tier=CompressionTier.MINI)

        compressed1 = compressor.compress(sample_snp_data["mini"])
        compressed2 = compressor.compress(sample_snp_data["mini"])

        assert compressed1 == compressed2, "Compression not deterministic"

    def test_invalid_tier_handling(self):
        """Test error handling for invalid compression tiers"""
        with pytest.raises(ValueError, match="Invalid compression tier"):
            SNPCompressor(tier="INVALID_TIER")

    def test_empty_data_compression(self):
        """Test compression of empty datasets"""
        compressor = SNPCompressor(tier=CompressionTier.MINI)
        compressed = compressor.compress({})

        assert len(compressed) < 1024  # Should be minimal overhead
        assert compressor.decompress(compressed) == {}
