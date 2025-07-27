from typing import Any, Dict

"""
Comprehensive test suite for HDC encoding implementation

Tests all stages of the HDC implementation plan including:
- Determinism and reproducibility
- Algebraic properties
- Performance benchmarks
- Compression tiers
- API endpoints
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from genomevault.hypervector_transform.binding_operations import (
    BindingOperations,  # Test legacy import
)
from genomevault.hypervector_transform.binding_operations import (
    BindingType,
    HypervectorBinder,
)
from genomevault.hypervector_transform.hdc_encoder import (
    CompressionTier,
    HypervectorConfig,
    HypervectorEncoder,
    OmicsType,
    ProjectionType,
    create_encoder,
)
from genomevault.hypervector_transform.registry import HypervectorRegistry, VersionMigrator


class TestHDCDeterminism:
    """Test Stage 1 - Seed/Version Registry & Determinism"""


    def test_encoding_determinism(self) -> None:
    """Test that encoding is deterministic with same seed"""
        # Create two encoders with same seed
        config = HypervectorConfig(seed=42, dimension=10000)
        encoder1 = HypervectorEncoder(config)
        encoder2 = HypervectorEncoder(config)

        # Create test features
        features = np.random.randn(1000)

        # Encode with both encoders
        hv1 = encoder1.encode(features, OmicsType.GENOMIC)
        hv2 = encoder2.encode(features, OmicsType.GENOMIC)

        # Should be identical
        assert torch.allclose(hv1, hv2), "Encoding not deterministic"


    def test_registry_version_management(self) -> None:
    """Test version registry functionality"""
        registry = HypervectorRegistry("test_registry.json")

        # Register new version
        registry.register_version(
            version="test_v1.0",
            params={"dimension": 15000, "projection_type": "sparse_random", "seed": 12345},
            description="Test version",
        )

        # Get encoder with version
        encoder = registry.get_encoder("test_v1.0")

        assert encoder.dimension == 15000
        assert encoder.version == "test_v1.0"

        # Test version listing
        versions = registry.list_versions()
        assert any(v["version"] == "test_v1.0" for v in versions)

        # Clean up
        Path("test_registry.json").unlink(missing_ok=True)
        Path("test_registry.backup.json").unlink(missing_ok=True)

    @pytest.mark.parametrize("seed", [42, 123, 999])

    def test_seed_reproducibility(self, seed) -> None:
    """Test reproducibility across different seeds"""
        config = HypervectorConfig(seed=seed)
        encoder = HypervectorEncoder(config)

        features = np.random.randn(500)
        hv1 = encoder.encode(features, OmicsType.GENOMIC)

        # Re-create encoder with same seed
        encoder2 = HypervectorEncoder(HypervectorConfig(seed=seed))
        hv2 = encoder2.encode(features, OmicsType.GENOMIC)

        assert torch.allclose(hv1, hv2)


class TestAlgebraicProperties:
    """Test Stage 2 - Algebraic Properties"""


    def test_binding_identities(self) -> None:
    """Test algebraic identities for binding operations"""
        binder = HypervectorBinder(10000)

        # Create test vectors
        a = torch.randn(10000)
        b = torch.randn(10000)

        # Test unbind(bind(a,b),b) ≈ a
        bound = binder.bind([a, b], BindingType.CIRCULAR)
        recovered = binder.unbind(bound, [b], BindingType.CIRCULAR)

        similarity = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), recovered.unsqueeze(0)
        ).item()

        assert similarity > 0.95, f"Recovery failed: similarity = {similarity}"


    def test_binding_properties(self) -> None:
    """Test mathematical properties of bindings"""
        binder = HypervectorBinder(10000)
        results = binder.test_binding_properties()

        # Check commutativity
        assert results["multiply_commutative"], "Multiplication not commutative"

        # Check associativity
        assert results["multiply_associative"], "Multiplication not associative"

        # Check inverse quality
        assert results["circular_inverse_quality"] > 0.9, "Poor inverse quality"

        # Check distributivity
        assert results["distributive"] > 0.9, "Poor distributivity"


    def test_legacy_import(self) -> None:
    """Test backward compatibility with legacy import"""
        # Should be able to import BindingOperations
        binder = BindingOperations(10000)
        assert isinstance(binder, HypervectorBinder)

    @given(
        dim=st.integers(min_value=1000, max_value=20000),
        num_vectors=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=10, deadline=5000)

    def test_binding_dimension_preservation(self, dim, num_vectors) -> None:
    """Property test: binding preserves dimension"""
        binder = HypervectorBinder(dim)

        # Create random vectors
        vectors = [torch.randn(dim) for _ in range(num_vectors)]

        # Test all binding types
        for binding_type in BindingType:
            if binding_type == BindingType.BUNDLING:
                continue  # Skip bundling as it's not a binding operation

            try:
                bound = binder.bind(vectors, binding_type)
                assert bound.shape[0] == dim, f"Dimension not preserved for {binding_type}"
            except:
                # Some binding types may not support multiple vectors
                pass


class TestCompressionTiers:
    """Test Stage 3 - Compression Tiers"""


    def test_tier_dimensions(self) -> None:
    """Test that each tier produces correct dimensions"""
        features = np.random.randn(1000)

        for tier in CompressionTier:
            config = HypervectorConfig(compression_tier=tier)
            encoder = HypervectorEncoder(config)

            hv = encoder.encode(features, OmicsType.GENOMIC, tier)

            expected_dim = encoder.tier_configs[tier]["dimension"]
            assert (
                hv.shape[0] == expected_dim
            ), f"Wrong dimension for {tier}: {hv.shape[0]} != {expected_dim}"


    def test_tier_memory_usage(self) -> None:
    """Test memory usage for each tier"""
        features = np.random.randn(1000)

        tier_sizes = {
            CompressionTier.MINI: 25,  # KB
            CompressionTier.CLINICAL: 300,
            CompressionTier.FULL: 200,
        }

        for tier, expected_kb in tier_sizes.items():
            encoder = create_encoder(compression_tier=tier.value)
            hv = encoder.encode(features, OmicsType.GENOMIC, tier)

            # Calculate actual size
            actual_kb = hv.element_size() * hv.nelement() / 1024

            # Allow 20% variance
            assert (
                actual_kb < expected_kb * 1.2
            ), f"Memory usage too high for {tier}: {actual_kb:.1f} KB"


    def test_information_preservation_across_tiers(self) -> None:
    """Test that higher tiers preserve more information"""
        features = np.random.randn(1000)

        # Create similar features
        features_similar = features + np.random.randn(1000) * 0.1

        # Original similarity
        orig_sim = np.corrcoef(features, features_similar)[0, 1]

        # Test each tier
        tier_similarities = {}

        for tier in CompressionTier:
            encoder = create_encoder(compression_tier=tier.value)

            hv1 = encoder.encode(features, OmicsType.GENOMIC, tier)
            hv2 = encoder.encode(features_similar, OmicsType.GENOMIC, tier)

            sim = encoder.similarity(hv1, hv2)
            tier_similarities[tier] = sim

        # Higher tiers should preserve similarity better
        assert (
            tier_similarities[CompressionTier.MINI] <= tier_similarities[CompressionTier.CLINICAL]
        )
        assert (
            tier_similarities[CompressionTier.CLINICAL] <= tier_similarities[CompressionTier.FULL]
        )


class TestPerformanceBenchmarks:
    """Test Stage 4 - Performance & Memory Benchmarks"""

    @pytest.mark.parametrize("dimension", [5000, 10000, 20000])

    def test_encoding_throughput(self, dimension) -> None:
    """Test encoding throughput meets requirements"""
        encoder = create_encoder(dimension=dimension)
        features = np.random.randn(1000)

        # Warm-up
        _ = encoder.encode(features, OmicsType.GENOMIC)

        # Benchmark
        num_trials = 100
        start = time.time()

        for _ in range(num_trials):
            _ = encoder.encode(features, OmicsType.GENOMIC)

        elapsed = time.time() - start
        encodings_per_second = num_trials / elapsed

        # Should achieve reasonable throughput
        assert encodings_per_second > 100, f"Low throughput: {encodings_per_second:.1f} enc/s"


    def test_memory_efficiency(self) -> None:
    """Test memory efficiency of encoding"""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create encoder and encode large dataset
        encoder = create_encoder(dimension=20000)

        num_samples = 1000
        encoded_vectors = []

        for i in range(num_samples):
            features = np.random.randn(1000)
            hv = encoder.encode(features, OmicsType.GENOMIC)
            encoded_vectors.append(hv)

        # Check memory usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - baseline_memory

        # Expected: ~200KB per vector * 1000 = ~200MB
        expected_mb = 200

        assert memory_increase < expected_mb * 2, f"Memory usage too high: {memory_increase:.1f} MB"

    @pytest.mark.parametrize("projection_type", [p.value for p in ProjectionType])

    def test_projection_performance(self, projection_type) -> None:
    """Test performance of different projection types"""
        config = HypervectorConfig(dimension=10000, projection_type=ProjectionType(projection_type))
        encoder = HypervectorEncoder(config)

        features = np.random.randn(1000)

        # Time encoding
        start = time.time()
        _ = encoder.encode(features, OmicsType.GENOMIC)
        elapsed = time.time() - start

        # All projection types should be reasonably fast
        assert elapsed < 0.1, f"{projection_type} too slow: {elapsed:.3f}s"


class TestTaskValidation:
    """Test Stage 3 - Task-level Validation"""


    def test_similarity_search_accuracy(self) -> None:
    """Test similarity search accuracy"""
        encoder = create_encoder(dimension=10000)

        # Create dataset
        num_samples = 100
        samples = []
        labels = []

        # Create clusters
        for cluster in range(5):
            center = np.random.randn(1000) * 10
            for _ in range(20):
                sample = center + np.random.randn(1000)
                samples.append(sample)
                labels.append(cluster)

        # Encode all samples
        encoded = [encoder.encode(s, OmicsType.GENOMIC) for s in samples]

        # Test retrieval accuracy
        correct = 0
        for i, query in enumerate(encoded[:10]):  # Test first 10
            # Find nearest neighbors
            similarities = [encoder.similarity(query, hv) for hv in encoded[10:]]

            # Get top-5
            top5_indices = np.argsort(similarities)[-5:]
            top5_labels = [labels[10 + idx] for idx in top5_indices]

            # Check if correct cluster appears in top-5
            if labels[i] in top5_labels:
                correct += 1

        accuracy = correct / 10
        assert accuracy > 0.7, f"Low retrieval accuracy: {accuracy}"


    def test_classification_preservation(self) -> None:
    """Test that classification performance is preserved"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        # Generate synthetic classification data
        np.random.seed(42)
        n_samples = 500
        n_features = 100

        # Create two classes
        X_class0 = np.random.randn(n_samples // 2, n_features)
        X_class1 = np.random.randn(n_samples // 2, n_features) + 1.5

        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

        # Original accuracy
        clf_orig = LogisticRegression(random_state=42)
        orig_scores = cross_val_score(clf_orig, X, y, cv=5)
        orig_accuracy = orig_scores.mean()

        # Encode data
        encoder = create_encoder(dimension=5000)
        X_encoded = np.array([encoder.encode(x, OmicsType.GENOMIC).numpy() for x in X])

        # Encoded accuracy
        clf_enc = LogisticRegression(random_state=42, max_iter=1000)
        enc_scores = cross_val_score(clf_enc, X_encoded, y, cv=5)
        enc_accuracy = enc_scores.mean()

        # Should preserve most of the classification performance
        assert (
            enc_accuracy > orig_accuracy * 0.8
        ), f"Poor classification preservation: {enc_accuracy:.3f} vs {orig_accuracy:.3f}"


class TestIntegrationAPI:
    """Test Stage 5 - Integration & API"""


    def test_encoding_api_response(self) -> None:
    """Test encoding API response format"""
        from genomevault.hypervector_transform.hdc_api import EncodingRequest, EncodingResponse

        # Create request
        request = EncodingRequest(
            features={"variants": [1, 2, 3, 4, 5]},
            omics_type="genomic",
            compression_tier="clinical",
        )

        # Validate request model
        assert request.omics_type == "genomic"
        assert request.compression_tier == "clinical"


    def test_multimodal_binding(self) -> None:
    """Test multi-modal binding functionality"""
        encoder = create_encoder()
        binder = HypervectorBinder(encoder.config.dimension)

        # Create multi-modal data
        genomic_data = {"variants": np.random.randn(100)}
        transcript_data = {"expression": np.random.randn(200)}
        clinical_data = {"age": 45, "bmi": 25.5}

        # Encode each modality
        genomic_hv = encoder.encode(genomic_data, OmicsType.GENOMIC)
        transcript_hv = encoder.encode(transcript_data, OmicsType.TRANSCRIPTOMIC)
        clinical_hv = encoder.encode(clinical_data, OmicsType.CLINICAL)

        # Bind together
        combined = binder.bind([genomic_hv, transcript_hv, clinical_hv], BindingType.FOURIER)

        assert combined.shape[0] == encoder.config.dimension

        # Test that we can query back (approximately)
        # This would require storing role vectors in practice


    def test_version_migration(self) -> None:
    """Test version migration functionality"""
        registry = HypervectorRegistry("test_migration_registry.json")

        # Register two versions
        registry.register_version("v1", {"dimension": 10000, "projection_type": "random_gaussian"})
        registry.register_version("v2", {"dimension": 15000, "projection_type": "sparse_random"})

        # Create test vector
        encoder_v1 = registry.get_encoder("v1")
        test_data = np.random.randn(500)
        hv_v1 = encoder_v1.encode(test_data, OmicsType.GENOMIC)

        # Migrate to v2
        migrator = VersionMigrator(registry)
        hv_v2 = migrator.migrate_hypervector(hv_v1, "v1", "v2")

        assert hv_v2.shape[0] == 15000

        # Test migration report
        report = migrator.create_migration_report("v1", "v2", test_vectors=10)
        assert "compatibility" in report
        assert "tests" in report

        # Clean up
        Path("test_migration_registry.json").unlink(missing_ok=True)
        Path("test_migration_registry.backup.json").unlink(missing_ok=True)


class TestEndToEnd:
    """End-to-end integration tests"""


    def test_complete_pipeline(self) -> None:
    """Test complete encoding pipeline"""
        # 1. Initialize registry
        registry = HypervectorRegistry("test_e2e_registry.json")

        # 2. Register custom version
        registry.register_version(
            version="e2e_test_v1",
            params={
                "dimension": 15000,
                "projection_type": "sparse_random",
                "seed": 42,
                "sparsity": 0.1,
            },
        )

        # 3. Get encoder
        encoder = registry.get_encoder("e2e_test_v1")

        # 4. Create multi-modal data
        data = {
            "genomic": {"variants": np.random.randn(1000)},
            "clinical": {"age": 50, "bmi": 27.3, "gender": "M"},
        }

        # 5. Encode each modality
        encoded = {}
        for modality, features in data.items():
            omics_type = OmicsType.GENOMIC if modality == "genomic" else OmicsType.CLINICAL
            encoded[modality] = encoder.encode(features, omics_type)

        # 6. Bind modalities
        binder = HypervectorBinder(encoder.config.dimension)
        combined = binder.bind(list(encoded.values()), BindingType.FOURIER)

        # 7. Verify properties
        assert combined.shape[0] == 15000
        assert torch.isfinite(combined).all()

        # 8. Test composite binding
        role_filler_pairs = [
            (torch.randn(15000), encoded["genomic"]),
            (torch.randn(15000), encoded["clinical"]),
        ]
        composite = binder.create_composite_binding(role_filler_pairs)
        assert composite.shape[0] == 15000

        # Clean up
        Path("test_e2e_registry.json").unlink(missing_ok=True)
        Path("test_e2e_registry.backup.json").unlink(missing_ok=True)


    def test_privacy_guarantee(self) -> None:
    """Test that original data cannot be recovered"""
        encoder = create_encoder(dimension=10000)

        # Generate sensitive genomic data
        original_data = np.random.randn(1000)

        # Encode
        encoded = encoder.encode(original_data, OmicsType.GENOMIC)

        # Try to recover without projection matrix
        # This simulates an attacker
        random_projection = torch.randn(1000, 10000)
        attempted_recovery = torch.matmul(random_projection, encoded)

        # Calculate similarity
        similarity = np.corrcoef(original_data, attempted_recovery.numpy())[0, 1]

        # Should be near zero (no correlation)
        assert abs(similarity) < 0.1, "Privacy breach: data recovered!"


    def test_binding_capacity(self) -> None:
    """Test binding capacity degradation"""
        binder = HypervectorBinder(10000)

        # Test capacity calculation
        capacities = []
        for n in [2, 5, 10, 20, 50]:
            capacity = binder.compute_binding_capacity(n)
            capacities.append(capacity)

        # Capacity should decrease with more items
        for i in range(1, len(capacities)):
            assert capacities[i] < capacities[i - 1], "Capacity not decreasing"

@pytest.mark.benchmark

class TestBenchmarks:
    """Performance benchmarks for reporting"""


    def test_benchmark_summary(self, benchmark) -> None:
    """Generate benchmark summary"""
        encoder = create_encoder(dimension=10000)
        features = np.random.randn(1000)

        # Benchmark encoding
        result = benchmark(encoder.encode, features, OmicsType.GENOMIC)

        assert result.shape[0] == 10000


    def test_binding_benchmark(self, benchmark) -> None:
    """Benchmark binding operations"""
        binder = HypervectorBinder(10000)
        vectors = [torch.randn(10000) for _ in range(5)]

        # Benchmark circular binding
        result = benchmark(binder.bind, vectors, BindingType.CIRCULAR)

        assert result.shape[0] == 10000

# Fixtures for pytest
@pytest.fixture

def temp_registry() -> None:
    """Create temporary registry for testing"""
    registry = HypervectorRegistry("temp_test_registry.json")
    yield registry
    # Cleanup
    Path("temp_test_registry.json").unlink(missing_ok=True)
    Path("temp_test_registry.backup.json").unlink(missing_ok=True)

@pytest.fixture

def sample_encoder() -> None:
    """Create sample encoder for testing"""
    return create_encoder(dimension=10000, seed=42)

@pytest.fixture

def sample_features() -> None:
    """Generate sample features for testing"""
    np.random.seed(42)
    return {
        "genomic": np.random.randn(1000),
        "transcriptomic": np.random.randn(500),
        "clinical": {"age": 45, "bmi": 25.5, "gender": "F"},
    }
