"""
Unit tests for hypervector encoding module
"""

import unittest

import numpy as np
import torch

from genomevault.core.constants import OmicsType
from genomevault.hypervector_transform.binding import (
    BindingType,
    CrossModalBinder,
    HypervectorBinder,
    PositionalBinder,
)
from genomevault.hypervector_transform.encoding import (
    HypervectorConfig,
    HypervectorEncoder,
    ProjectionType,
)
from genomevault.hypervector_transform.holographic import (
    HolographicEncoder,
)
from genomevault.hypervector_transform.mapping import (
    BiologicalSimilarityMapper,
    ManifoldPreservingMapper,
    SimilarityPreservingMapper,
)


class TestHypervectorEncoder(unittest.TestCase):
    """Test hypervector encoding functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.encoder = HypervectorEncoder()
        self.test_features = torch.randn(100)
        self.test_data = {
            "variants": {"snps": [1, 2, 3], "indels": [4, 5], "cnvs": [6]},
            "quality_metrics": {
                "mean_coverage": 30.5,
                "uniformity": 0.95,
                "gc_content": 0.45,
            },
        }

    def test_encoder_initialization(self):
        """Test encoder initialization"""
        self.assertEqual(self.encoder.config.dimension, 10000)
        self.assertEqual(
            self.encoder.config.projection_type, ProjectionType.SPARSE_RANDOM
        )
        self.assertTrue(self.encoder.config.normalize)

    def test_basic_encoding(self):
        """Test basic encoding functionality"""
        hypervector = self.encoder.encode(self.test_features, OmicsType.GENOMIC)

        self.assertEqual(hypervector.shape[0], 10000)
        self.assertAlmostEqual(torch.norm(hypervector).item(), 1.0, places=5)

    def test_multiresolution_encoding(self):
        """Test multi-resolution encoding"""
        multi_vectors = self.encoder.encode_multiresolution(
            self.test_features, OmicsType.GENOMIC
        )

        self.assertIn("base", multi_vectors)
        self.assertIn("mid", multi_vectors)
        self.assertIn("high", multi_vectors)

        self.assertEqual(multi_vectors["base"].shape[0], 10000)
        self.assertEqual(multi_vectors["mid"].shape[0], 15000)
        self.assertEqual(multi_vectors["high"].shape[0], 20000)

    def test_dict_feature_extraction(self):
        """Test feature extraction from dictionary"""
        hypervector = self.encoder.encode(self.test_data, OmicsType.GENOMIC)

        self.assertEqual(hypervector.shape[0], 10000)

    def test_similarity_computation(self):
        """Test similarity computation"""
        hv1 = self.encoder.encode(torch.randn(100), OmicsType.GENOMIC)
        hv2 = self.encoder.encode(torch.randn(100), OmicsType.GENOMIC)

        cosine_sim = self.encoder.similarity(hv1, hv2, "cosine")
        self.assertTrue(-1 <= cosine_sim <= 1)

        euclidean_sim = self.encoder.similarity(hv1, hv2, "euclidean")
        self.assertTrue(euclidean_sim <= 0)

        hamming_sim = self.encoder.similarity(hv1, hv2, "hamming")
        self.assertTrue(0 <= hamming_sim <= 1)

    def test_projection_types(self):
        """Test different projection types"""
        for proj_type in [
            ProjectionType.RANDOM_GAUSSIAN,
            ProjectionType.SPARSE_RANDOM,
            ProjectionType.ORTHOGONAL,
        ]:
            config = HypervectorConfig(projection_type=proj_type)
            encoder = HypervectorEncoder(config)

            hv = encoder.encode(self.test_features, OmicsType.GENOMIC)
            self.assertEqual(hv.shape[0], 10000)

    def test_quantization(self):
        """Test hypervector quantization"""
        config = HypervectorConfig(quantize=True, quantization_bits=4)
        encoder = HypervectorEncoder(config)

        hv = encoder.encode(self.test_features, OmicsType.GENOMIC)
        unique_values = torch.unique(hv).shape[0]

        # Should have limited unique values due to quantization
        self.assertLessEqual(unique_values, 2**4)


class TestHypervectorBinder(unittest.TestCase):
    """Test hypervector binding operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.dimension = 1000
        self.binder = HypervectorBinder(self.dimension)
        self.v1 = torch.randn(self.dimension)
        self.v2 = torch.randn(self.dimension)
        self.v3 = torch.randn(self.dimension)

    def test_multiply_binding(self):
        """Test element-wise multiplication binding"""
        bound = self.binder.bind([self.v1, self.v2], BindingType.MULTIPLY)
        self.assertEqual(bound.shape[0], self.dimension)

        # Test unbinding
        recovered = self.binder.unbind(bound, [self.v2], BindingType.MULTIPLY)
        similarity = torch.nn.functional.cosine_similarity(
            self.v1.unsqueeze(0), recovered.unsqueeze(0)
        ).item()
        self.assertGreater(similarity, 0.9)

    def test_circular_binding(self):
        """Test circular convolution binding"""
        bound = self.binder.bind([self.v1, self.v2], BindingType.CIRCULAR)
        self.assertEqual(bound.shape[0], self.dimension)

        # Test unbinding
        recovered = self.binder.unbind(bound, [self.v2], BindingType.CIRCULAR)
        similarity = torch.nn.functional.cosine_similarity(
            self.v1.unsqueeze(0), recovered.unsqueeze(0)
        ).item()
        self.assertGreater(similarity, 0.8)

    def test_permutation_binding(self):
        """Test permutation-based binding"""
        bound = self.binder.bind([self.v1, self.v2], BindingType.PERMUTATION)
        self.assertEqual(bound.shape[0], self.dimension)

    def test_xor_binding(self):
        """Test XOR binding for binary vectors"""
        # Create binary vectors
        b1 = torch.sign(self.v1)
        b2 = torch.sign(self.v2)

        bound = self.binder.bind([b1, b2], BindingType.XOR)
        self.assertEqual(bound.shape[0], self.dimension)

        # XOR is self-inverse
        recovered = self.binder.unbind(bound, [b2], BindingType.XOR)
        similarity = (b1 == recovered).float().mean().item()
        self.assertGreater(similarity, 0.9)

    def test_bundling(self):
        """Test bundling operation"""
        bundle = self.binder.bundle([self.v1, self.v2, self.v3])
        self.assertEqual(bundle.shape[0], self.dimension)

        # Bundle should have positive similarity with all components
        for v in [self.v1, self.v2, self.v3]:
            sim = torch.nn.functional.cosine_similarity(
                bundle.unsqueeze(0), v.unsqueeze(0)
            ).item()
            self.assertGreater(sim, 0)

    def test_protection(self):
        """Test vector protection with key"""
        key = torch.randn(self.dimension)

        protected = self.binder.protect(self.v1, key)
        recovered = self.binder.unprotect(protected, key)

        similarity = torch.nn.functional.cosine_similarity(
            self.v1.unsqueeze(0), recovered.unsqueeze(0)
        ).item()
        self.assertGreater(similarity, 0.9)


class TestPositionalBinder(unittest.TestCase):
    """Test positional binding"""

    def setUp(self):
        """Set up test fixtures"""
        self.dimension = 1000
        self.binder = PositionalBinder(self.dimension)
        self.vectors = [torch.randn(self.dimension) for _ in range(5)]

    def test_position_binding(self):
        """Test binding with position"""
        bound = self.binder.bind_with_position(self.vectors[0], 100)
        self.assertEqual(bound.shape[0], self.dimension)

    def test_sequence_binding(self):
        """Test sequence binding"""
        sequence_bound = self.binder.bind_sequence(self.vectors)
        self.assertEqual(sequence_bound.shape[0], self.dimension)


class TestCrossModalBinder(unittest.TestCase):
    """Test cross-modal binding"""

    def setUp(self):
        """Set up test fixtures"""
        self.dimension = 1000
        self.binder = CrossModalBinder(self.dimension)
        self.modalities = {
            "genomic": torch.randn(self.dimension),
            "transcriptomic": torch.randn(self.dimension),
            "epigenomic": torch.randn(self.dimension),
        }

    def test_modality_binding(self):
        """Test binding multiple modalities"""
        results = self.binder.bind_modalities(self.modalities)

        # Check combined representation exists
        self.assertIn("combined", results)
        self.assertEqual(results["combined"].shape[0], self.dimension)

        # Check pairwise combinations
        self.assertIn("genomic_transcriptomic", results)
        self.assertIn("genomic_epigenomic", results)
        self.assertIn("transcriptomic_epigenomic", results)


class TestHolographicEncoder(unittest.TestCase):
    """Test holographic encoding"""

    def setUp(self):
        """Set up test fixtures"""
        self.dimension = 1000
        self.encoder = HolographicEncoder(self.dimension)

    def test_structure_encoding(self):
        """Test encoding hierarchical structures"""
        structure = {
            "gene": "BRCA1",
            "expression": 5.2,
            "conditions": {"tissue": "breast", "treatment": "none"},
        }

        hologram = self.encoder.encode_structure(structure)
        self.assertEqual(hologram.root.shape[0], self.dimension)
        self.assertIn("gene", hologram.components)
        self.assertIn("conditions_nested", hologram.components)

    def test_query(self):
        """Test querying holographic representation"""
        structure = {"a": torch.randn(self.dimension), "b": torch.randn(self.dimension)}
        hologram = self.encoder.encode_structure(structure)

        # Query for component
        recovered_a = self.encoder.query(hologram.root, "a")
        self.assertEqual(recovered_a.shape[0], self.dimension)

    def test_genomic_variant_encoding(self):
        """Test encoding genomic variants"""
        variant_hv = self.encoder.encode_genomic_variant(
            "chr1", 12345, "A", "G", {"effect": "missense"}
        )
        self.assertEqual(variant_hv.shape[0], self.dimension)

    def test_memory_trace(self):
        """Test creating memory traces"""
        items = [
            {"id": 1, "value": "A"},
            {"id": 2, "value": "B"},
            {"id": 3, "value": "C"},
        ]

        memory = self.encoder.create_memory_trace(items)
        self.assertEqual(memory.shape[0], self.dimension)

    def test_similarity_preserving_hash(self):
        """Test similarity-preserving hash"""
        v1 = torch.randn(self.dimension)
        v2 = v1 + torch.randn(self.dimension) * 0.1  # Similar vector
        v3 = torch.randn(self.dimension)  # Different vector

        hash1 = self.encoder.similarity_preserving_hash(v1)
        hash2 = self.encoder.similarity_preserving_hash(v2)
        hash3 = self.encoder.similarity_preserving_hash(v3)

        # Similar vectors should have similar hashes
        self.assertEqual(len(hash1), 16)  # 64 bits = 16 hex chars

        # Count matching bits
        def count_matching_bits(h1, h2):
            b1 = bin(int(h1, 16))[2:].zfill(64)
            b2 = bin(int(h2, 16))[2:].zfill(64)
            return sum(c1 == c2 for c1, c2 in zip(b1, b2))

        match_12 = count_matching_bits(hash1, hash2)
        match_13 = count_matching_bits(hash1, hash3)

        # Similar vectors should have more matching bits
        self.assertGreater(match_12, match_13)


class TestSimilarityMapping(unittest.TestCase):
    """Test similarity-preserving mappings"""

    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 100
        self.output_dim = 1000
        self.n_samples = 50
        self.data = torch.randn(self.n_samples, self.input_dim)

    def test_basic_mapping(self):
        """Test basic similarity preservation"""
        mapper = SimilarityPreservingMapper(self.input_dim, self.output_dim)
        transformed = mapper.fit_transform(self.data)

        self.assertEqual(transformed.shape, (self.n_samples, self.output_dim))

        # Check similarity preservation
        orig_sim = mapper._compute_similarities(self.data)
        trans_sim = mapper._compute_similarities(transformed)

        # Average similarity difference should be small
        sim_diff = torch.abs(orig_sim - trans_sim).mean().item()
        self.assertLess(sim_diff, 0.5)

    def test_biological_mapper(self):
        """Test biological similarity mapper"""
        mapper = BiologicalSimilarityMapper(
            self.input_dim, self.output_dim, OmicsType.GENOMIC
        )

        # Test variant similarity
        v1 = torch.rand(100)
        v2 = torch.rand(100)
        sim = mapper.compute_biological_similarity(v1, v2, "variant")
        self.assertTrue(0 <= sim <= 1)

    def test_manifold_mapper(self):
        """Test manifold-preserving mapper"""
        # Create data with manifold structure
        t = torch.linspace(0, 4 * np.pi, self.n_samples)
        manifold_data = torch.stack([torch.sin(t), torch.cos(t), t / 10], dim=1)

        mapper = ManifoldPreservingMapper(3, 10, n_neighbors=5)
        embedded = mapper.fit_transform(manifold_data)

        self.assertEqual(embedded.shape, (self.n_samples, 10))

        # Check that nearby points remain nearby
        orig_dist = torch.cdist(manifold_data[:5], manifold_data[:5])
        embed_dist = torch.cdist(embedded[:5], embedded[:5])

        # Correlation between distances should be high
        orig_flat = orig_dist.flatten()
        embed_flat = embed_dist.flatten()
        correlation = torch.corrcoef(torch.stack([orig_flat, embed_flat]))[0, 1].item()

        self.assertGreater(correlation, 0.5)


if __name__ == "__main__":
    unittest.main()
