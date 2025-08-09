"""
Tests for hypervector encoding with golden vectors, stability tests, and benchmarks.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from genomevault.core.exceptions import ProjectionError
from genomevault.hypervector.encoding.orthogonal_projection import OrthogonalProjection
from genomevault.hypervector.encoding.sparse_projection import SparseRandomProjection
from genomevault.hypervector.encoding.unified_encoder import (
    UnifiedHypervectorEncoder,
    create_encoder,
)
from genomevault.hypervector.operations.binding import bundle

# Constants for magic number elimination (PLR2004)
SIM_LOWER = 0.5
SIM_UPPER = 0.95
COMPONENT_SIM_MIN = 0.2
SEQ_SIM_LOWER = 0.1
SEQ_SIM_UPPER = 0.7
THROUGHPUT_TARGET = 1000
CORRELATION_MIN = 0.7


class TestUnifiedHypervectorEncoder:
    """Tests for the unified hypervector encoder."""

    def test_dimension_validation(self):
        """Test that only valid dimensions are accepted."""
        # Valid dimensions
        for dim in [10000, 15000, 20000]:
            encoder = UnifiedHypervectorEncoder(dimension=dim)
            assert encoder.dimension == dim

        # Invalid dimensions
        with pytest.raises(ProjectionError) as exc_info:
            UnifiedHypervectorEncoder(dimension=5000)
        assert "10000, 15000, or 20000" in str(exc_info.value)

    def test_projection_types(self):
        """Test both sparse and orthogonal projection types."""
        # Sparse projection
        sparse_encoder = UnifiedHypervectorEncoder(
            dimension=10000, projection_type="sparse", sparse_density=0.1
        )
        assert sparse_encoder.projection_type == "sparse"
        assert isinstance(sparse_encoder.projection, SparseRandomProjection)

        # Orthogonal projection
        ortho_encoder = UnifiedHypervectorEncoder(dimension=10000, projection_type="orthogonal")
        assert ortho_encoder.projection_type == "orthogonal"
        assert isinstance(ortho_encoder.projection, OrthogonalProjection)

        # Invalid projection type
        with pytest.raises(ProjectionError):
            UnifiedHypervectorEncoder(projection_type="invalid")

    def test_deterministic_encoding(self):
        """Test that encoding is deterministic with same seed."""
        # Create two encoders with same seed
        encoder1 = create_encoder(dimension=10000, seed=42)
        encoder2 = create_encoder(dimension=10000, seed=42)

        # Fit both
        encoder1.fit(100)
        encoder2.fit(100)

        # Encode same variant
        vec1 = encoder1.encode_variant("chr1", 12345, "A", "G", "SNP")
        vec2 = encoder2.encode_variant("chr1", 12345, "A", "G", "SNP")

        # Should be identical
        np.testing.assert_array_almost_equal(vec1, vec2)

    def test_different_seeds_produce_different_vectors(self):
        """Test that different seeds produce different encodings."""
        encoder1 = create_encoder(dimension=10000, seed=42)
        encoder2 = create_encoder(dimension=10000, seed=43)

        encoder1.fit(100)
        encoder2.fit(100)

        vec1 = encoder1.encode_variant("chr1", 12345, "A", "G", "SNP")
        vec2 = encoder2.encode_variant("chr1", 12345, "A", "G", "SNP")

        # Should be different
        similarity = encoder1.similarity(vec1, vec2)
        assert similarity < SIM_UPPER  # Should not be too similar

    def test_encode_genomic_features(self):
        """Test encoding of genomic feature matrices."""
        encoder = create_encoder(dimension=10000)
        n_samples, n_features = 5, 50

        # Generate random features
        features = np.random.randn(n_samples, n_features)

        # Fit and encode
        encoder.fit(n_features)
        hypervectors = encoder.encode_genomic_features(features)

        # Check shape
        assert hypervectors.shape == (n_samples, 10000)

        # Check normalization
        norms = np.linalg.norm(hypervectors, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(n_samples))

    def test_encode_variant(self):
        """Test encoding of individual variants."""
        encoder = create_encoder(dimension=10000)

        # Test different variant types
        variants = [
            ("chr1", 12345, "A", "G", "SNP"),
            ("chr2", 67890, "ATG", "A", "DEL"),
            ("chrX", 11111, "C", "CGG", "INS"),
            ("chrY", 22222, "ACGT", "TGCA", "INV"),
        ]

        vectors = []
        for chrom, pos, ref, alt, var_type in variants:
            vec = encoder.encode_variant(chrom, pos, ref, alt, var_type)
            assert vec.shape == (10000,)
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-6
            vectors.append(vec)

        # Check that different variants produce different vectors
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarity = encoder.similarity(vectors[i], vectors[j])
                assert similarity < SIM_UPPER - 0.15  # Should be distinguishable

    def test_encode_sequence(self):
        """Test DNA sequence encoding."""
        encoder = create_encoder(dimension=10000)

        # Test sequences
        sequences = [
            "ATCG",
            "GGCCTTAA",
            "ATATATATATAT",
            "",  # Empty sequence
        ]

        for seq in sequences:
            vec = encoder.encode_sequence(seq)
            assert vec.shape == (10000,)

            if seq:  # Non-empty sequences should be normalized
                assert abs(np.linalg.norm(vec) - 1.0) < 1e-6
            else:  # Empty sequence should be zero vector
                assert np.allclose(vec, 0)

    def test_cross_modal_binding(self):
        """Test cross-modal binding between genomic and clinical data."""
        encoder = create_encoder(dimension=10000)

        # Create mock genomic and clinical vectors
        genomic_vec = np.random.randn(10000)
        genomic_vec /= np.linalg.norm(genomic_vec)

        clinical_vec = np.random.randn(10000)
        clinical_vec /= np.linalg.norm(clinical_vec)

        # Test default weights
        combined = encoder.cross_modal_binding(genomic_vec, clinical_vec)
        assert combined.shape == (10000,)
        assert abs(np.linalg.norm(combined) - 1.0) < 1e-6

        # Test custom weights
        weights = {"genomic": 0.7, "clinical": 0.3}
        combined_weighted = encoder.cross_modal_binding(genomic_vec, clinical_vec, weights)
        assert combined_weighted.shape == (10000,)
        assert abs(np.linalg.norm(combined_weighted) - 1.0) < 1e-6

        # Combined vectors should be different with different weights
        similarity = encoder.similarity(combined, combined_weighted)
        assert SIM_LOWER < similarity < SIM_UPPER  # Related but not identical

    def test_decode_components(self):
        """Test component decoding from bundled vectors."""
        encoder = create_encoder(dimension=10000)
        encoder.fit(100)

        # Create a bundled vector from known components
        vec_a = encoder._base_vectors["A"]
        vec_snp = encoder._base_vectors["SNP"]
        vec_chr1 = encoder._base_vectors["chr1"]

        bundled = bundle([vec_a, vec_snp, vec_chr1])

        # Decode components
        components = encoder.decode_components(bundled, threshold=COMPONENT_SIM_MIN)

        # Should find all three components
        component_names = [name for name, _ in components]
        assert "A" in component_names
        assert "SNP" in component_names
        assert "chr1" in component_names

        # Similarities should be reasonable
        for name, similarity in components:
            assert COMPONENT_SIM_MIN <= similarity <= 1.0


class TestGoldenVectors:
    """Tests with pre-computed golden vectors for regression testing."""

    @pytest.fixture
    def golden_vectors(self):
        """Generate golden vectors for comparison."""
        encoder = create_encoder(dimension=10000, seed=12345)
        encoder.fit(100)

        golden = {
            "snp_chr1_12345": encoder.encode_variant("chr1", 12345, "A", "G", "SNP"),
            "del_chr2_67890": encoder.encode_variant("chr2", 67890, "ATG", "A", "DEL"),
            "sequence_atcg": encoder.encode_sequence("ATCG"),
            "sequence_ggcc": encoder.encode_sequence("GGCC"),
        }

        return golden

    def test_golden_vector_stability(self, golden_vectors):
        """Test that golden vectors remain stable across runs."""
        # Create new encoder with same seed
        encoder = create_encoder(dimension=10000, seed=12345)
        encoder.fit(100)

        # Re-generate vectors
        new_vectors = {
            "snp_chr1_12345": encoder.encode_variant("chr1", 12345, "A", "G", "SNP"),
            "del_chr2_67890": encoder.encode_variant("chr2", 67890, "ATG", "A", "DEL"),
            "sequence_atcg": encoder.encode_sequence("ATCG"),
            "sequence_ggcc": encoder.encode_sequence("GGCC"),
        }

        # Compare with golden vectors
        for key in golden_vectors:
            np.testing.assert_array_almost_equal(
                golden_vectors[key],
                new_vectors[key],
                decimal=10,
                err_msg=f"Golden vector mismatch for {key}",
            )

    def test_golden_vector_relationships(self, golden_vectors):
        """Test expected relationships between golden vectors."""
        encoder = create_encoder(dimension=10000)

        # SNPs on same chromosome should be more similar than different chromosomes
        snp_chr1 = golden_vectors["snp_chr1_12345"]
        del_chr2 = golden_vectors["del_chr2_67890"]

        # Create another chr1 variant
        encoder.fit(100)
        encoder._base_vectors = create_encoder(dimension=10000, seed=12345)._base_vectors
        snp_chr1_other = encoder.encode_variant("chr1", 99999, "C", "T", "SNP")

        sim_same_chr = encoder.similarity(snp_chr1, snp_chr1_other)
        sim_diff_chr = encoder.similarity(snp_chr1, del_chr2)

        # Same chromosome variants should be more similar
        assert sim_same_chr > sim_diff_chr

        # Sequences with shared bases should be somewhat similar
        seq_atcg = golden_vectors["sequence_atcg"]
        seq_ggcc = golden_vectors["sequence_ggcc"]
        sim_sequences = encoder.similarity(seq_atcg, seq_ggcc)

        # Should have some similarity due to shared C/G
        assert SEQ_SIM_LOWER < sim_sequences < SEQ_SIM_UPPER


class TestPerformanceBenchmarks:
    """Performance benchmarks for hypervector operations."""

    @pytest.mark.parametrize("dimension", [10000, 15000, 20000])
    @pytest.mark.parametrize("projection_type", ["sparse", "orthogonal"])
    def test_encoding_performance(self, dimension, projection_type):
        """Benchmark encoding performance for different dimensions and projections."""
        encoder = create_encoder(
            dimension=dimension, projection_type=projection_type, sparse_density=0.1
        )

        # Prepare data
        n_variants = 10000
        n_features = 100
        encoder.fit(n_features)

        # Benchmark variant encoding
        start_time = time.time()
        for i in range(n_variants):
            encoder.encode_variant(f"chr{(i % 22) + 1}", i * 1000, "A", "G", "SNP")
        encoding_time = time.time() - start_time

        # Calculate throughput
        variants_per_second = n_variants / encoding_time
        ms_per_variant = (encoding_time / n_variants) * 1000

        print(f"\nDimension: {dimension}, Projection: {projection_type}")
        print(f"Encoded {n_variants} variants in {encoding_time:.2f} seconds")
        print(f"Throughput: {variants_per_second:.0f} variants/second")
        print(f"Latency: {ms_per_variant:.3f} ms/variant")

        # Basic performance assertion (adjust based on hardware)
        # On 8-core CPU, should encode at least 1000 variants/second
        assert (
            variants_per_second > THROUGHPUT_TARGET
        ), f"Performance too low: {variants_per_second:.0f} variants/s"

    def test_similarity_preservation(self):
        """Test that cosine similarity is preserved across projections."""
        n_features = 100
        n_samples = 100

        # Generate random feature matrix
        np.random.seed(42)
        features = np.random.randn(n_samples, n_features)

        # Compute original similarities
        original_similarities = []
        for i in range(10):
            for j in range(i + 1, 10):
                sim = np.dot(features[i], features[j]) / (
                    np.linalg.norm(features[i]) * np.linalg.norm(features[j])
                )
                original_similarities.append(sim)

        # Test different projections
        for projection_type in ["sparse", "orthogonal"]:
            encoder = create_encoder(dimension=10000, projection_type=projection_type)
            encoder.fit(n_features)

            # Project features
            projected = encoder.encode_genomic_features(features)

            # Compute projected similarities
            projected_similarities = []
            for i in range(10):
                for j in range(i + 1, 10):
                    sim = encoder.similarity(projected[i], projected[j])
                    projected_similarities.append(sim)

            # Check similarity preservation
            original_similarities = np.array(original_similarities)
            projected_similarities = np.array(projected_similarities)

            # Compute correlation
            correlation = np.corrcoef(original_similarities, projected_similarities)[0, 1]

            print(f"\n{projection_type} projection similarity preservation:")
            print(f"Correlation: {correlation:.3f}")

            # Should preserve similarity structure reasonably well
            assert correlation > CORRELATION_MIN, f"Poor similarity preservation: {correlation:.3f}"

    def test_memory_efficiency(self):
        """Test memory usage of hypervector operations."""
        import sys

        dimensions = [10000, 15000, 20000]

        for dim in dimensions:
            encoder = create_encoder(dimension=dim)
            encoder.fit(100)

            # Create some vectors
            vec1 = encoder.encode_variant("chr1", 12345, "A", "G", "SNP")
            vec2 = encoder.encode_sequence("ATCGATCGATCG")

            # Check memory usage
            vec_memory = sys.getsizeof(vec1)
            expected_memory = dim * 8  # 8 bytes per float64

            print(f"\nDimension {dim}:")
            print(f"Vector memory: {vec_memory} bytes")
            print(f"Expected: ~{expected_memory} bytes")
            print(f"Overhead: {vec_memory - expected_memory} bytes")

            # Memory should be close to expected (some overhead is normal)
            assert vec_memory < expected_memory * 1.5, f"Excessive memory usage: {vec_memory} bytes"


if __name__ == "__main__":
    # Run performance benchmarks
    benchmarks = TestPerformanceBenchmarks()

    print("Running performance benchmarks...")
    for dim in [10000, 15000, 20000]:
        for proj in ["sparse", "orthogonal"]:
            benchmarks.test_encoding_performance(dim, proj)

    benchmarks.test_similarity_preservation()
    benchmarks.test_memory_efficiency()
