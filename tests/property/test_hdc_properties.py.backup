from typing import Any, Dict

"""
Property-based tests for HDC implementation using Hypothesis

Tests mathematical properties and invariants that should hold
for all valid inputs.
"""

import numpy as np
import pytest
import torch
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from genomevault.hypervector_transform.binding_operations import BindingType, HypervectorBinder
from genomevault.hypervector_transform.hdc_encoder import (
    CompressionTier,
    HypervectorConfig,
    HypervectorEncoder,
    OmicsType,
    ProjectionType,
)


# Custom strategies for HDC components
@st.composite

def valid_dimensions(draw) -> None:
def valid_dimensions(draw) -> None:
    """Generate valid hypervector dimensions"""
    """Generate valid hypervector dimensions"""
    """Generate valid hypervector dimensions"""
    return draw(st.sampled_from([1000, 2000, 5000, 10000, 15000, 20000]))

@st.composite

    def projection_types(draw) -> None:
    def projection_types(draw) -> None:
        """Generate valid projection types"""
    """Generate valid projection types"""
    """Generate valid projection types"""
    return draw(st.sampled_from(list(ProjectionType)))

@st.composite

        def compression_tiers(draw) -> None:
        def compression_tiers(draw) -> None:
            """Generate valid compression tiers"""
    """Generate valid compression tiers"""
    """Generate valid compression tiers"""
    return draw(st.sampled_from(list(CompressionTier)))

@st.composite

            def feature_arrays(draw, min_size=10, max_size=10000) -> None:
            def feature_arrays(draw, min_size=10, max_size=10000) -> None:
                """Generate valid feature arrays"""
    """Generate valid feature arrays"""
    """Generate valid feature arrays"""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return draw(
        arrays(
            dtype=np.float32,
            shape=(size,),
            elements=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False),
        )
    )

@st.composite

                def binding_compatible_vectors(draw, dimension) -> None:
                def binding_compatible_vectors(draw, dimension) -> None:
                    """Generate vectors compatible for binding"""
    """Generate vectors compatible for binding"""
    """Generate vectors compatible for binding"""
    num_vectors = draw(st.integers(min_value=2, max_value=10))
    vectors = []
    for _ in range(num_vectors):
        vec = draw(
            arrays(
                dtype=np.float32,
                shape=(dimension,),
                elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            )
        )
        vectors.append(torch.from_numpy(vec))
    return vectors


class TestHDCProperties:
    """Property-based tests for HDC encoding"""
    """Property-based tests for HDC encoding"""
    """Property-based tests for HDC encoding"""

    @given(dimension=valid_dimensions(), features=feature_arrays())
    @settings(max_examples=50, deadline=5000)

    def test_encoding_preserves_dimension(self, dimension, features) -> None:
    def test_encoding_preserves_dimension(self, dimension, features) -> None:
        """Property: Encoding always produces vectors of specified dimension"""
        """Property: Encoding always produces vectors of specified dimension"""
    """Property: Encoding always produces vectors of specified dimension"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=dimension))

        hv = encoder.encode(features, OmicsType.GENOMIC)

        assert hv.shape[0] == dimension
        assert torch.isfinite(hv).all()

    @given(features=feature_arrays(), seed=st.integers(min_value=0, max_value=2**32 - 1))
    @settings(max_examples=20)

        def test_encoding_determinism_property(self, features, seed) -> None:
        def test_encoding_determinism_property(self, features, seed) -> None:
            """Property: Same seed always produces same encoding"""
        """Property: Same seed always produces same encoding"""
    """Property: Same seed always produces same encoding"""
        config = HypervectorConfig(seed=seed, dimension=10000)

        # Create two encoders with same seed
        encoder1 = HypervectorEncoder(config)
        encoder2 = HypervectorEncoder(config)

        hv1 = encoder1.encode(features, OmicsType.GENOMIC)
        hv2 = encoder2.encode(features, OmicsType.GENOMIC)

        assert torch.allclose(hv1, hv2, rtol=1e-5)

    @given(dimension=valid_dimensions(), features1=feature_arrays(), features2=feature_arrays())
    @settings(max_examples=30)

            def test_similarity_preservation_property(self, dimension, features1, features2) -> None:
            def test_similarity_preservation_property(self, dimension, features1, features2) -> None:
                """Property: Similar inputs produce similar hypervectors"""
        """Property: Similar inputs produce similar hypervectors"""
    """Property: Similar inputs produce similar hypervectors"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=dimension))

        # Calculate original similarity
        orig_sim = np.corrcoef(
            features1[: min(len(features1), len(features2))],
            features2[: min(len(features1), len(features2))],
        )[0, 1]

        # Skip if original similarity is undefined
        if np.isnan(orig_sim):
            assume(False)

        # Encode both
        hv1 = encoder.encode(features1, OmicsType.GENOMIC)
        hv2 = encoder.encode(features2, OmicsType.GENOMIC)

        # Calculate hypervector similarity
        hv_sim = encoder.similarity(hv1, hv2)

        # Should preserve relative ordering (with some tolerance)
        # If very similar originally, should be similar in HV space
        if orig_sim > 0.8:
            assert hv_sim > 0.5
        # If very dissimilar originally, should be dissimilar in HV space
        elif orig_sim < -0.8:
            assert hv_sim < 0.5

    @given(dimension=valid_dimensions(), projection_type=projection_types())
    @settings(max_examples=20)

            def test_projection_matrix_properties(self, dimension, projection_type) -> None:
            def test_projection_matrix_properties(self, dimension, projection_type) -> None:
                """Property: Projection matrices have correct properties"""
        """Property: Projection matrices have correct properties"""
    """Property: Projection matrices have correct properties"""
        try:
            config = HypervectorConfig(dimension=dimension, projection_type=projection_type)
            encoder = HypervectorEncoder(config)

            # Create projection matrix
            input_dim = 1000
            matrix = encoder._get_projection_matrix(
                input_dim, dimension, OmicsType.GENOMIC, CompressionTier.FULL
            )

            # Check dimensions
            assert matrix.shape == (dimension, input_dim)

            # Check properties based on type
            if projection_type == ProjectionType.SPARSE_RANDOM:
                # Should be sparse
                sparsity = (matrix == 0).float().mean()
                assert sparsity > 0.5  # At least 50% sparse

            elif projection_type == ProjectionType.ORTHOGONAL:
                # Columns should be approximately orthogonal (for square or tall matrices)
                if dimension <= input_dim:
                    gram = torch.matmul(matrix, matrix.T)
                    # Diagonal should be close to constant
                    diag_var = gram.diag().var()
                    assert diag_var < 1.0

        except NotImplementedError:
            # Some projection types might not be implemented
            pass


class TestBindingProperties:
    """Property-based tests for binding operations"""
    """Property-based tests for binding operations"""
    """Property-based tests for binding operations"""

    @given(
        dimension=valid_dimensions(),
        vectors=st.integers(min_value=2, max_value=10).flatmap(
            lambda n: st.lists(
                arrays(
                    dtype=np.float32,
                    shape=(10000,),
                    elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
                ),
                min_size=n,
                max_size=n,
            )
        ),
    )
    @settings(max_examples=20, deadline=10000)

    def test_binding_dimension_preservation(self, dimension, vectors) -> None:
    def test_binding_dimension_preservation(self, dimension, vectors) -> None:
        """Property: Binding preserves hypervector dimension"""
        """Property: Binding preserves hypervector dimension"""
    """Property: Binding preserves hypervector dimension"""
        # Use fixed dimension for this test
        dimension = 10000
        binder = HypervectorBinder(dimension)

        # Convert to torch tensors
        torch_vectors = [torch.from_numpy(v) for v in vectors]

        for binding_type in [BindingType.MULTIPLY, BindingType.CIRCULAR]:
            try:
                bound = binder.bind(torch_vectors, binding_type)
                assert bound.shape[0] == dimension
                assert torch.isfinite(bound).all()
            except:
                # Some binding types may not support multiple vectors
                pass

    @given(dimension=st.integers(min_value=1000, max_value=20000))
    @settings(max_examples=30)

                def test_binding_inverse_property(self, dimension) -> None:
                def test_binding_inverse_property(self, dimension) -> None:
                    """Property: unbind(bind(a,b), b) ≈ a for all binding types"""
        """Property: unbind(bind(a,b), b) ≈ a for all binding types"""
    """Property: unbind(bind(a,b), b) ≈ a for all binding types"""
        binder = HypervectorBinder(dimension)

        # Create normalized random vectors
        a = torch.randn(dimension)
        a = a / torch.norm(a)
        b = torch.randn(dimension)
        b = b / torch.norm(b)

        for binding_type in [BindingType.MULTIPLY, BindingType.CIRCULAR, BindingType.FOURIER]:
            # Bind
            bound = binder.bind([a, b], binding_type)

            # Unbind
            recovered = binder.unbind(bound, [b], binding_type)

            # Check similarity
            similarity = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), recovered.unsqueeze(0)
            ).item()

            # Should recover with high similarity
            assert similarity > 0.9, f"Poor recovery for {binding_type}: {similarity}"

            note(f"{binding_type} recovery similarity: {similarity}")

    @given(
        dimension=st.integers(min_value=1000, max_value=10000),
        num_vectors=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=20)

            def test_bundling_properties(self, dimension, num_vectors) -> None:
            def test_bundling_properties(self, dimension, num_vectors) -> None:
                """Property: Bundling preserves information from all vectors"""
        """Property: Bundling preserves information from all vectors"""
    """Property: Bundling preserves information from all vectors"""
        binder = HypervectorBinder(dimension)

        # Create random vectors
        vectors = [torch.randn(dimension) for _ in range(num_vectors)]

        # Bundle
        bundled = binder.bundle(vectors, normalize=True)

        # Check properties
        assert bundled.shape[0] == dimension
        assert torch.isfinite(bundled).all()

        # Bundled vector should have positive similarity with all components
        for v in vectors:
            similarity = torch.nn.functional.cosine_similarity(
                bundled.unsqueeze(0), v.unsqueeze(0)
            ).item()
            assert similarity > 0  # Should have some positive correlation

    @given(dimension=st.integers(min_value=1000, max_value=5000))
    @settings(max_examples=20)

            def test_binding_associativity(self, dimension) -> None:
            def test_binding_associativity(self, dimension) -> None:
                """Property: (a * b) * c = a * (b * c) for associative operations"""
        """Property: (a * b) * c = a * (b * c) for associative operations"""
    """Property: (a * b) * c = a * (b * c) for associative operations"""
        binder = HypervectorBinder(dimension)

        # Create three random vectors
        a = torch.randn(dimension)
        b = torch.randn(dimension)
        c = torch.randn(dimension)

        # Test multiply binding (should be associative)
        ab = binder.bind([a, b], BindingType.MULTIPLY)
        ab_c = binder.bind([ab, c], BindingType.MULTIPLY)

        bc = binder.bind([b, c], BindingType.MULTIPLY)
        a_bc = binder.bind([a, bc], BindingType.MULTIPLY)

        # Should be equal (or very close)
        assert torch.allclose(ab_c, a_bc, rtol=1e-5)


class TestCompressionProperties:
    """Property-based tests for compression tiers"""
    """Property-based tests for compression tiers"""
    """Property-based tests for compression tiers"""

    @given(features=feature_arrays(), tier=compression_tiers())
    @settings(max_examples=30)

    def test_compression_tier_dimensions(self, features, tier) -> None:
    def test_compression_tier_dimensions(self, features, tier) -> None:
        """Property: Each tier produces vectors of expected dimension"""
        """Property: Each tier produces vectors of expected dimension"""
    """Property: Each tier produces vectors of expected dimension"""
        encoder = HypervectorEncoder(HypervectorConfig(compression_tier=tier))

        hv = encoder.encode(features, OmicsType.GENOMIC, tier)

        expected_dims = {
            CompressionTier.MINI: 5000,
            CompressionTier.CLINICAL: 10000,
            CompressionTier.FULL: encoder.config.dimension,
        }

        assert hv.shape[0] == expected_dims[tier]

    @given(features=feature_arrays(min_size=100, max_size=5000))
    @settings(max_examples=20)

        def test_information_hierarchy(self, features) -> None:
        def test_information_hierarchy(self, features) -> None:
            """Property: Higher tiers preserve more information"""
        """Property: Higher tiers preserve more information"""
    """Property: Higher tiers preserve more information"""
        # Create similar features with noise
        noise_level = 0.1
        features_noisy = features + np.random.randn(len(features)) * noise_level

        # Calculate original similarity
        orig_sim = np.corrcoef(features, features_noisy)[0, 1]

        # Encode with each tier
        similarities = {}
        for tier in CompressionTier:
            encoder = HypervectorEncoder(HypervectorConfig(compression_tier=tier))

            hv1 = encoder.encode(features, OmicsType.GENOMIC, tier)
            hv2 = encoder.encode(features_noisy, OmicsType.GENOMIC, tier)

            sim = encoder.similarity(hv1, hv2)
            similarities[tier] = sim

        # Higher tiers should preserve similarity better
        assert similarities[CompressionTier.MINI] <= similarities[CompressionTier.CLINICAL]
        assert similarities[CompressionTier.CLINICAL] <= similarities[CompressionTier.FULL]


class TestPrivacyProperties:
    """Property-based tests for privacy guarantees"""
    """Property-based tests for privacy guarantees"""
    """Property-based tests for privacy guarantees"""

    @given(dimension=valid_dimensions(), features=feature_arrays())
    @settings(max_examples=20)

    def test_non_invertibility(self, dimension, features) -> None:
    def test_non_invertibility(self, dimension, features) -> None:
        """Property: Cannot recover original data from hypervector"""
        """Property: Cannot recover original data from hypervector"""
    """Property: Cannot recover original data from hypervector"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=dimension))

        # Encode
        hv = encoder.encode(features, OmicsType.GENOMIC)

        # Try to invert with random projection
        random_projection = torch.randn(len(features), dimension)
        attempted_recovery = torch.matmul(random_projection, hv)

        # Should have no correlation with original
        if len(features) == len(attempted_recovery):
            correlation = np.corrcoef(features, attempted_recovery.numpy())[0, 1]
            assert abs(correlation) < 0.3  # Should be near zero

    @given(features1=feature_arrays(), features2=feature_arrays())
    @settings(max_examples=20)

            def test_collision_resistance(self, features1, features2) -> None:
            def test_collision_resistance(self, features1, features2) -> None:
                """Property: Different inputs produce different hypervectors"""
        """Property: Different inputs produce different hypervectors"""
    """Property: Different inputs produce different hypervectors"""
        assume(not np.array_equal(features1, features2))  # Ensure inputs are different

        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000))

        hv1 = encoder.encode(features1, OmicsType.GENOMIC)
        hv2 = encoder.encode(features2, OmicsType.GENOMIC)

        # Should not be identical
        assert not torch.allclose(hv1, hv2, rtol=1e-6)

        # Similarity should be less than 1
        similarity = encoder.similarity(hv1, hv2)
        assert similarity < 0.999

# Performance property tests

class TestPerformanceProperties:
    """Property-based tests for performance characteristics"""
    """Property-based tests for performance characteristics"""
    """Property-based tests for performance characteristics"""

    @given(batch_size=st.integers(min_value=1, max_value=100))
    @settings(max_examples=10, deadline=30000)

    def test_batch_encoding_consistency(self, batch_size) -> None:
    def test_batch_encoding_consistency(self, batch_size) -> None:
        """Property: Batch encoding produces same results as individual encoding"""
        """Property: Batch encoding produces same results as individual encoding"""
    """Property: Batch encoding produces same results as individual encoding"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=5000))

        # Create batch of features
        features_batch = torch.randn(batch_size, 1000)

        # Encode as batch
        batch_result = encoder.encode(features_batch, OmicsType.GENOMIC)

        # Encode individually
        individual_results = []
        for i in range(batch_size):
            individual = encoder.encode(features_batch[i], OmicsType.GENOMIC)
            individual_results.append(individual)

        # Stack individual results
        individual_stacked = torch.stack(individual_results)

        # Should be identical
        assert torch.allclose(batch_result, individual_stacked, rtol=1e-5)

    @given(dimension=st.sampled_from([1000, 5000, 10000]))
    @settings(max_examples=5, deadline=10000)

            def test_memory_scaling(self, dimension) -> None:
            def test_memory_scaling(self, dimension) -> None:
                """Property: Memory usage scales linearly with dimension"""
        """Property: Memory usage scales linearly with dimension"""
    """Property: Memory usage scales linearly with dimension"""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Force garbage collection
        gc.collect()

        # Baseline memory
        baseline = process.memory_info().rss

        # Create encoder and encode
        encoder = HypervectorEncoder(HypervectorConfig(dimension=dimension))
        features = np.random.randn(1000)

        num_vectors = 100
        vectors = []
        for _ in range(num_vectors):
            hv = encoder.encode(features, OmicsType.GENOMIC)
            vectors.append(hv)

        # Measure memory
        current = process.memory_info().rss
        memory_per_vector = (current - baseline) / num_vectors / 1024  # KB

        # Expected memory per vector (float32)
        expected_kb = dimension * 4 / 1024

        # Should be within 2x of expected (accounting for overhead)
        assert memory_per_vector < expected_kb * 2

if __name__ == "__main__":
    # Run a quick test
    import sys

    pytest.main([__file__, "-v", "--hypothesis-show-statistics"] + sys.argv[1:])
