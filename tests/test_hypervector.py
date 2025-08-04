from genomevault.observability.logging import configure_logging

logger = configure_logging()
# tests/test_hdc_hypervector.py
import numpy as np
import pytest

from genomevault.hypervector_transform.binding import BindingOperations
from genomevault.hypervector_transform.encoding import HypervectorEncoder
from genomevault.hypervector_transform.holographic import \
    HolographicRepresentation


class TestHypervectorEngine:
    """Test suite for hyperdimensional computing operations"""

    @pytest.fixture
    def encoder_10k(self):
        """10,000-dimensional base encoder"""
        return HypervectorEncoder(dimensions=10000, resolution="base")

    @pytest.fixture
    def encoder_15k(self):
        """15,000-dimensional mid-level encoder"""
        return HypervectorEncoder(dimensions=15000, resolution="mid")

    @pytest.fixture
    def encoder_20k(self):
        """20,000-dimensional high-level encoder"""
        return HypervectorEncoder(dimensions=20000, resolution="high")

    @pytest.fixture
    def sample_genomic_features(self):
        """Generate realistic genomic feature vectors"""
        return {
            "variants": np.random.randn(1000),  # 1000 variant features
            "expression": np.random.randn(500),  # 500 gene expression values
            "methylation": np.random.rand(300),  # 300 methylation sites
            "clinical": np.random.randint(0, 5, size=50),  # 50 clinical features
        }

    def test_multi_resolution_encoding(
        self, encoder_10k, encoder_15k, encoder_20k, sample_genomic_features
    ):
        """Test hierarchical encoding at different resolutions"""
        features = sample_genomic_features["variants"]

        # Encode at each resolution
        base_vec = encoder_10k.encode(features)
        mid_vec = encoder_15k.encode(features)
        high_vec = encoder_20k.encode(features)

        # Verify dimensions
        assert base_vec.shape == (10000,)
        assert mid_vec.shape == (15000,)
        assert high_vec.shape == (20000,)

        # Verify information preservation increases with dimensions
        base_info = encoder_10k.information_content(base_vec)
        mid_info = encoder_15k.information_content(mid_vec)
        high_info = encoder_20k.information_content(high_vec)

        assert base_info < mid_info < high_info

    def test_similarity_preservation(self, encoder_10k, sample_genomic_features):
        """Verify similarity preservation property: E[cos(y1,y2)] = cos(x1,x2)"""
        # Create two similar feature vectors
        features1 = sample_genomic_features["variants"]
        features2 = features1 + np.random.randn(1000) * 0.1  # Small perturbation

        # Original similarity
        orig_sim = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2)
        )

        # Encode and compute hypervector similarity
        vec1 = encoder_10k.encode(features1)
        vec2 = encoder_10k.encode(features2)
        hyper_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        # Should preserve similarity within tolerance
        assert (
            abs(orig_sim - hyper_sim) < 0.1
        ), "Similarity not preserved: {orig_sim:.3f} vs {hyper_sim:.3f}"

    def test_binding_operations(self, encoder_10k, sample_genomic_features):
        """Test hypervector binding operations"""
        binding = BindingOperations()

        # Encode different modalities
        var_vec = encoder_10k.encode(sample_genomic_features["variants"])
        expr_vec = encoder_10k.encode(sample_genomic_features["expression"])

        # Test element-wise multiplication binding
        bound_elem = binding.element_wise_bind(var_vec, expr_vec)
        assert bound_elem.shape == var_vec.shape

        # Test circular convolution binding
        bound_conv = binding.circular_convolution(var_vec, expr_vec)
        assert bound_conv.shape == var_vec.shape

        # Verify binding is reversible
        unbound = binding.unbind(bound_elem, expr_vec)
        similarity = np.dot(unbound, var_vec) / (
            np.linalg.norm(unbound) * np.linalg.norm(var_vec)
        )
        assert similarity > 0.8, "Binding should be approximately reversible"

    def test_holographic_representation(self, encoder_10k, sample_genomic_features):
        """Test holographic distributed representation"""
        holographic = HolographicRepresentation(dimensions=10000)

        # Create multi-modal representation
        modalities = {
            "genomic": encoder_10k.encode(sample_genomic_features["variants"]),
            "transcriptomic": encoder_10k.encode(sample_genomic_features["expression"]),
            "epigenetic": encoder_10k.encode(sample_genomic_features["methylation"]),
        }

        # Generate holographic representation
        holo_vec = holographic.create_representation(modalities)

        # Verify distributed property - information spread across vector
        entropy = -np.sum(np.abs(holo_vec) * np.log(np.abs(holo_vec) + 1e-10))
        assert entropy > 1000, "Holographic representation should be well-distributed"

        # Verify individual modalities can be approximately recovered
        for modality, original in modalities.items():
            recovered = holographic.recover_modality(holo_vec, modality)
            similarity = np.corrcoef(original, recovered)[0, 1]
            assert similarity > 0.7, "Failed to recover {modality}"

    def test_privacy_preservation(self, encoder_10k):
        """Test computational infeasibility of reconstruction"""
        # Generate random features
        original = np.random.randn(1000)
        encoded = encoder_10k.encode(original)

        # Attempt naive reconstruction (should fail)
        # Simulate adversary without projection matrix knowledge
        random_matrix = np.random.randn(1000, 10000)
        attempted_reconstruction = random_matrix @ encoded

        # Verify reconstruction fails
        similarity = np.corrcoef(original, attempted_reconstruction)[0, 1]
        assert abs(similarity) < 0.1, "Reconstruction should fail without key"

    @pytest.mark.parametrize(
        "dimensions,expected_memory_kb",
        [
            (10000, 78),  # 10k * 8 bytes / 1024
            (15000, 117),  # 15k * 8 bytes / 1024
            (20000, 156),  # 20k * 8 bytes / 1024
        ],
    )
    def test_memory_footprint(self, dimensions, expected_memory_kb):
        """Verify memory usage matches specifications"""
        encoder = HypervectorEncoder(dimensions=dimensions)
        vector = encoder.encode(np.random.randn(1000))

        # Calculate actual memory usage
        memory_kb = vector.nbytes / 1024

        # Allow 5% variance for overhead
        assert abs(memory_kb - expected_memory_kb) < expected_memory_kb * 0.05

    def test_cross_modal_binding(self, encoder_10k, sample_genomic_features):
        """Test binding across different biological modalities"""
        binding = BindingOperations()

        # Encode all modalities
        vectors = {
            key: encoder_10k.encode(features)
            for key, features in sample_genomic_features.items()
        }

        # Perform hierarchical binding
        genomic_clinical = binding.element_wise_bind(
            vectors["variants"], vectors["clinical"]
        )
        expression_methylation = binding.circular_convolution(
            vectors["expression"], vectors["methylation"]
        )

        # Final cross-modal binding
        final = binding.element_wise_bind(genomic_clinical, expression_methylation)

        # Verify final vector maintains relationships
        assert final.shape == (10000,)
        assert np.std(final) > 0.1  # Non-trivial variation
        assert np.abs(np.mean(final)) < 0.1  # Approximately zero-mean

    def test_encoding_determinism(self, encoder_10k):
        """Ensure encoding is deterministic for same input"""
        features = np.random.randn(1000)

        vec1 = encoder_10k.encode(features)
        vec2 = encoder_10k.encode(features)

        assert np.array_equal(vec1, vec2), "Encoding must be deterministic"

    def test_performance_benchmark(self, encoder_10k, sample_genomic_features):
        """Benchmark encoding performance"""
        import time

        features = sample_genomic_features["variants"]

        # Warm-up
        _ = encoder_10k.encode(features)

        # Time encoding
        start = time.time()
        for _ in range(100):
            _ = encoder_10k.encode(features)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / 100) * 1000
        assert avg_time_ms < 50, "Encoding too slow: {avg_time_ms:.1f}ms (target <50ms)"
