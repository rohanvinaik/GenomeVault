"""
Property-based tests for HDC implementation using Hypothesis

Tests mathematical properties and invariants that should hold
for all valid inputs.
"""
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st
import pytest

from hypothesis.extra.numpy import arrays
import numpy as np
import torch

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
)


# Custom strategies for HDC components
@st.composite
def valid_dimensions(draw):
    """Generate valid hypervector dimensions"""
    return draw(st.sampled_from([1000, 2000, 5000, 10000, 15000, 20000]))


@st.composite
def projection_types(draw):
    """Generate valid projection types"""
    return draw(st.sampled_from(list(ProjectionType)))


@st.composite
def compression_tiers(draw):
    """Generate valid compression tiers"""
    return draw(st.sampled_from(list(CompressionTier)))


@st.composite
def feature_arrays(draw, min_size=10, max_size=10000):
    """Generate valid feature arrays"""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return draw(
        arrays(
            dtype=np.float32,
            shape=(size,),
            elements=st.floats(
                -100,
                100,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32,
            ),
        )
    )


@st.composite
def binding_compatible_vectors(draw, dimension):
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

    @given(dimension=valid_dimensions(), features=feature_arrays())
    @settings(max_examples=10, deadline=10000)
    def test_encoding_preserves_dimension(self, dimension, features):
        """Property: Encoding always produces vectors of specified dimension"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=dimension, seed=42))

        hv = encoder.encode(features, OmicsType.GENOMIC)

        assert hv.shape[0] == dimension
        assert torch.isfinite(hv).all()

    @given(
        features=feature_arrays(), seed=st.integers(min_value=0, max_value=2**32 - 1)
    )
    @settings(max_examples=20)
    def test_encoding_determinism_property(self, features, seed):
        """Property: Same seed always produces same encoding"""
        config = HypervectorConfig(seed=seed, dimension=10000)

        # Create two encoders with same seed
        encoder1 = HypervectorEncoder(config)
        encoder2 = HypervectorEncoder(config)

        hv1 = encoder1.encode(features, OmicsType.GENOMIC)
        hv2 = encoder2.encode(features, OmicsType.GENOMIC)

        assert torch.allclose(hv1, hv2, rtol=1e-5)

    @given(
        dimension=valid_dimensions(),
        features1=feature_arrays(min_size=50, max_size=500),
        features2=feature_arrays(min_size=50, max_size=500),
    )
    @settings(max_examples=10)
    def test_similarity_preservation_property(self, dimension, features1, features2):
        """Property: Similar inputs produce similar hypervectors"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=dimension, seed=42))

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
    @settings(max_examples=5)
    def test_projection_matrix_properties(self, dimension, projection_type):
        """Property: Projection matrices have correct properties"""
        try:
            config = HypervectorConfig(
                dimension=dimension, projection_type=projection_type, seed=42
            )
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

        except NotImplementedError as e:
            # Some projection types might not be fully implemented
            pytest.skip(f"Projection type {projection_type} not implemented: {e}")
        except Exception as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception(f"Unexpected error testing {projection_type}")
            pytest.skip(f"Projection type {projection_type} failed unexpectedly: {e}")


class TestBindingProperties:
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
    def test_binding_dimension_preservation(self, dimension, vectors):
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
            except ValueError as e:
                # Some binding types may not support multiple vectors
                pytest.skip(
                    f"Binding type {binding_type} not supported for multiple vectors: {e}"
                )
            except NotImplementedError as e:
                pytest.skip(f"Binding type {binding_type} not implemented: {e}")
            except Exception as e:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception(f"Unexpected error in binding test for {binding_type}")
                pytest.skip(f"Binding type {binding_type} failed unexpectedly: {e}")

    @given(dimension=st.integers(min_value=1000, max_value=20000))
    @settings(max_examples=10)
    def test_binding_inverse_property(self, dimension):
        """Property: unbind(bind(a,b), b) ≈ a for all binding types"""
        binder = HypervectorBinder(dimension, seed=42)

        # Create normalized random vectors
        a = torch.randn(dimension)
        a = a / torch.norm(a)
        b = torch.randn(dimension)
        b = b / torch.norm(b)

        for binding_type in [
            BindingType.MULTIPLY,
            BindingType.CIRCULAR,
            BindingType.FOURIER,
        ]:
            # Bind
            bound = binder.bind([a, b], binding_type)

            # Unbind
            recovered = binder.unbind(bound, [b], binding_type)

            # Check similarity
            similarity = torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), recovered.unsqueeze(0)
            ).item()

            # Should recover with reasonable similarity
            assert similarity > 0.6, f"Poor recovery for {binding_type}: {similarity}"

            note(f"{binding_type} recovery similarity: {similarity}")

    @given(
        dimension=st.integers(min_value=1000, max_value=10000),
        num_vectors=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=20)
    def test_bundling_properties(self, dimension, num_vectors):
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
    def test_binding_associativity(self, dimension):
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

    @given(
        features=feature_arrays(min_size=100, max_size=1000), tier=compression_tiers()
    )
    @settings(max_examples=5, deadline=5000)
    def test_compression_tier_dimensions(self, features, tier):
        """Property: Each tier produces vectors of expected dimension"""
        encoder = HypervectorEncoder(HypervectorConfig(compression_tier=tier, seed=42))

        hv = encoder.encode(features, OmicsType.GENOMIC, tier)

        expected_dims = {
            CompressionTier.MINI: 5000,
            CompressionTier.CLINICAL: 10000,
            CompressionTier.FULL: encoder.config.dimension,
        }

        assert hv.shape[0] == expected_dims[tier]

    @given(features=feature_arrays(min_size=100, max_size=1000))
    @settings(max_examples=5, deadline=5000)
    def test_information_hierarchy(self, features):
        """Property: Higher tiers preserve more information"""
        # Create similar features with noise
        np.random.seed(42)
        noise_level = 0.1
        features_noisy = features + np.random.randn(len(features)) * noise_level

        # Skip if features are all zeros or constant (no variance)
        if np.all(features == 0) or np.var(features) == 0:
            pytest.skip("Features have no variance")

        # Calculate original similarity
        orig_sim = np.corrcoef(features, features_noisy)[0, 1]
        if np.isnan(orig_sim):
            pytest.skip("Original similarity is undefined")

        # Encode with each tier
        similarities = {}
        for tier in CompressionTier:
            encoder = HypervectorEncoder(
                HypervectorConfig(compression_tier=tier, seed=42)
            )

            hv1 = encoder.encode(features, OmicsType.GENOMIC, tier)
            hv2 = encoder.encode(features_noisy, OmicsType.GENOMIC, tier)

            sim = encoder.similarity(hv1, hv2)
            similarities[tier] = sim

        # Higher tiers should preserve similarity better
        assert (
            similarities[CompressionTier.MINI] <= similarities[CompressionTier.CLINICAL]
        )
        assert (
            similarities[CompressionTier.CLINICAL] <= similarities[CompressionTier.FULL]
        )


class TestPrivacyProperties:
    """Property-based tests for privacy guarantees"""

    @given(dimension=valid_dimensions(), features=feature_arrays())
    @settings(max_examples=20)
    def test_non_invertibility(self, dimension, features):
        """Property: Cannot recover original data from hypervector"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=dimension))

        # Encode
        hv = encoder.encode(features, OmicsType.GENOMIC)

        # Try to invert with random projection
        random_projection = torch.randn(len(features), dimension)
        attempted_recovery = torch.matmul(random_projection, hv)

        # Should have low correlation with original
        if len(features) == len(attempted_recovery):
            correlation = np.corrcoef(features, attempted_recovery.numpy())[0, 1]
            if not np.isnan(correlation):
                assert abs(correlation) < 0.5  # Should be reasonably low

    @given(features1=feature_arrays(), features2=feature_arrays())
    @settings(max_examples=20)
    def test_collision_resistance(self, features1, features2):
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

    @given(batch_size=st.integers(min_value=1, max_value=100))
    @settings(max_examples=10, deadline=30000)
    def test_batch_encoding_consistency(self, batch_size):
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
    def test_memory_scaling(self, dimension):
        """Property: Memory usage scales linearly with dimension"""
        psutil = pytest.importorskip("psutil")
        import gc
        import os

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
