"""
Adversarial tests for HDC implementation

Tests the robustness of HDC encoding against malicious or edge-case inputs
designed to break the system or reveal sensitive information.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest
import torch

from genomevault.hypervector_transform.binding_operations import BindingType, HypervectorBinder
from genomevault.hypervector_transform.hdc_encoder import (
    CompressionTier,
    HypervectorConfig,
    HypervectorEncoder,
    OmicsType,
    ProjectionType,
)
from genomevault.hypervector_transform.registry import HypervectorRegistry


class TestAdversarialInputs:
    """Test HDC robustness against adversarial inputs"""


    def test_extreme_values(self) -> None:
    """Test encoding with extreme input values"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000))

        # Test cases with extreme values
        test_cases = [
            np.array([1e10] * 100),  # Very large values
            np.array([1e-10] * 100),  # Very small values
            np.array([1e10, 1e-10] * 50),  # Mixed extreme values
            np.full(1000, np.finfo(np.float32).max),  # Max float32
            np.full(1000, np.finfo(np.float32).min),  # Min float32
        ]

        for i, features in enumerate(test_cases):
            # Should not crash or produce invalid output
            hv = encoder.encode(features, OmicsType.GENOMIC)

            # Verify output is valid
            assert hv.shape[0] == 10000
            assert torch.isfinite(hv).all(), f"Non-finite values in test case {i}"
            assert not torch.isnan(hv).any(), f"NaN values in test case {i}"


    def test_zero_and_constant_inputs(self) -> None:
    """Test encoding with degenerate inputs"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000))

        test_cases = [
            np.zeros(100),  # All zeros
            np.ones(100),  # All ones
            np.full(100, 42),  # Constant value
            np.array([0]),  # Single zero
            np.array([1]),  # Single value
        ]

        encoded_vectors = []
        for features in test_cases:
            hv = encoder.encode(features, OmicsType.GENOMIC)
            encoded_vectors.append(hv)

            # Should produce valid output
            assert torch.isfinite(hv).all()

        # Different constant inputs should produce different outputs
        for i in range(len(encoded_vectors)):
            for j in range(i + 1, len(encoded_vectors)):
                similarity = encoder.similarity(encoded_vectors[i], encoded_vectors[j])
                assert similarity < 0.99, "Different inputs produced nearly identical outputs"


    def test_adversarial_patterns(self) -> None:
    """Test with patterns designed to exploit weaknesses"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000))

        # Adversarial patterns
        test_cases = [
            # Alternating pattern that might cause issues with convolution
            np.array([1, -1] * 500),
            # Gradient pattern
            np.linspace(-1000, 1000, 1000),
            # Spike pattern
            np.concatenate([np.zeros(499), [1000], np.zeros(500)]),
            # Random permutation of same values
            np.random.permutation(np.arange(1000)),
        ]

        for features in test_cases:
            hv = encoder.encode(features, OmicsType.GENOMIC)
            assert torch.isfinite(hv).all()

            # Check that sparsity is reasonable
            if encoder.config.projection_type == ProjectionType.SPARSE_RANDOM:
                sparsity = (hv == 0).float().mean()
                assert 0 < sparsity < 1  # Not all zeros or all non-zeros


    def test_malicious_dict_inputs(self) -> None:
    """Test with maliciously crafted dictionary inputs"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000))

        # Malicious dictionary inputs
        test_cases = [
            # Empty dict
            {},
            # Nested structures
            {"a": {"b": {"c": [1, 2, 3]}}},
            # Mixed types
            {"numbers": [1, 2, 3], "string": "malicious", "none": None},
            # Very long keys
            {"a" * 1000: [1, 2, 3]},
            # Special characters in keys
            {"__proto__": [1, 2, 3], "constructor": [4, 5, 6]},
            # Circular reference attempt (though Python handles this)
            {"variants": {"nested": {"deep": np.array([1, 2, 3])}}},
        ]

        for data in test_cases:
            # Should handle gracefully without crashing
            try:
                hv = encoder.encode(data, OmicsType.GENOMIC)
                assert torch.isfinite(hv).all()
            except (ValueError, TypeError):
                # It's OK to reject malformed input
                pass


class TestPrivacyAttacks:
    """Test resistance to privacy attacks"""


    def test_membership_inference_attack(self) -> None:
    """Test resistance to membership inference"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000, seed=42))

        # Create training set
        train_size = 100
        train_data = [np.random.randn(1000) for _ in range(train_size)]
        train_hvs = [encoder.encode(d, OmicsType.GENOMIC) for d in train_data]

        # Create test set (not seen during "training")
        test_size = 100
        test_data = [np.random.randn(1000) for _ in range(test_size)]
        test_hvs = [encoder.encode(d, OmicsType.GENOMIC) for d in test_data]

        # Compute similarities to all training vectors
        train_max_sims = []
        for train_hv in train_hvs:
            sims = [
                encoder.similarity(train_hv, other) for other in train_hvs if other is not train_hv
            ]
            train_max_sims.append(max(sims))

        test_max_sims = []
        for test_hv in test_hvs:
            sims = [encoder.similarity(test_hv, train_hv) for train_hv in train_hvs]
            test_max_sims.append(max(sims))

        # The distributions should be similar (no easy membership inference)
        train_mean = np.mean(train_max_sims)
        test_mean = np.mean(test_max_sims)

        # Means should be close (within 10%)
        assert (
            abs(train_mean - test_mean) / train_mean < 0.1
        ), f"Membership inference possible: train={train_mean:.3f}, test={test_mean:.3f}"


    def test_model_inversion_attack(self) -> None:
    """Test resistance to model inversion attacks"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000))

        # Original sensitive data
        sensitive_features = np.array([1.5, -2.3, 0.7, 3.1, -0.5] * 200)  # 1000 features
        hv = encoder.encode(sensitive_features, OmicsType.GENOMIC)

        # Attacker tries to invert using optimization
        # Start with random guess
        guess = np.random.randn(1000)
        learning_rate = 0.01

        for _ in range(100):  # 100 optimization steps
            # Encode guess
            guess_hv = encoder.encode(guess, OmicsType.GENOMIC)

            # Compute gradient estimate (finite differences)
            eps = 1e-4
            gradients = np.zeros_like(guess)

            for i in range(min(10, len(guess))):  # Only optimize first 10 dims for speed
                guess_plus = guess.copy()
                guess_plus[i] += eps
                hv_plus = encoder.encode(guess_plus, OmicsType.GENOMIC)

                # Similarity-based loss
                loss_plus = encoder.similarity(hv_plus, hv)
                loss_base = encoder.similarity(guess_hv, hv)

                gradients[i] = (loss_plus - loss_base) / eps

            # Update guess
            guess += learning_rate * gradients[:10]

        # Check if attack recovered original data
        recovery_correlation = np.corrcoef(sensitive_features[:10], guess[:10])[0, 1]

        # Should not have recovered the data
        assert (
            abs(recovery_correlation) < 0.3
        ), f"Model inversion attack successful: correlation={recovery_correlation:.3f}"


    def test_dictionary_attack(self) -> None:
    """Test resistance to dictionary attacks on encoded vectors"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000, seed=42))

        # Create dictionary of known genomic patterns
        dictionary_size = 1000
        dictionary = []
        dictionary_hvs = []

        for i in range(dictionary_size):
            # Simulate known genomic patterns
            pattern = np.zeros(1000)
            pattern[i % 1000] = 1  # Simple one-hot patterns
            dictionary.append(pattern)
            dictionary_hvs.append(encoder.encode(pattern, OmicsType.GENOMIC))

        # Target pattern (not in dictionary)
        target = np.random.randn(1000)
        target_hv = encoder.encode(target, OmicsType.GENOMIC)

        # Find closest match in dictionary
        similarities = [encoder.similarity(target_hv, dhv) for dhv in dictionary_hvs]
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]

        # Even the best match should not be too similar
        assert (
            best_similarity < 0.7
        ), f"Dictionary attack found high similarity: {best_similarity:.3f}"


class TestBindingAttacks:
    """Test binding operation security"""


    def test_binding_extraction_attack(self) -> None:
    """Test resistance to extracting components from bound vectors"""
        binder = HypervectorBinder(10000)

        # Create secret vectors
        secret1 = torch.randn(10000)
        secret2 = torch.randn(10000)

        # Bind them
        bound = binder.bind([secret1, secret2], BindingType.CIRCULAR)

        # Attacker tries random keys to extract secrets
        num_attempts = 100
        max_similarity = 0

        for _ in range(num_attempts):
            random_key = torch.randn(10000)
            extracted = binder.unbind(bound, [random_key], BindingType.CIRCULAR)

            # Check similarity to actual secrets
            sim1 = torch.nn.functional.cosine_similarity(
                extracted.unsqueeze(0), secret1.unsqueeze(0)
            ).item()
            sim2 = torch.nn.functional.cosine_similarity(
                extracted.unsqueeze(0), secret2.unsqueeze(0)
            ).item()

            max_similarity = max(max_similarity, sim1, sim2)

        # Random unbinding should not recover secrets
        assert (
            max_similarity < 0.3
        ), f"Binding extraction attack successful: max_similarity={max_similarity:.3f}"


    def test_composite_binding_attack(self) -> None:
    """Test security of composite role-filler bindings"""
        binder = HypervectorBinder(10000)

        # Create sensitive role-filler pairs
        roles = [torch.randn(10000) for _ in range(5)]
        fillers = [torch.randn(10000) for _ in range(5)]  # Sensitive data

        # Create composite
        pairs = list(zip(roles, fillers))
        composite = binder.create_composite_binding(pairs, BindingType.FOURIER)

        # Attacker knows some roles but not all
        known_roles = roles[:2]  # Knows first 2 roles

        # Try to extract unknown fillers
        for i, role in enumerate(roles[2:], 2):
            # Use random probe instead of actual role
            probe = torch.randn(10000)
            extracted = binder.query_composite(composite, probe, BindingType.FOURIER)

            # Should not extract actual filler
            actual_filler = fillers[i]
            similarity = torch.nn.functional.cosine_similarity(
                extracted.unsqueeze(0), actual_filler.unsqueeze(0)
            ).item()

            assert (
                similarity < 0.3
            ), f"Extracted unknown filler with random probe: similarity={similarity:.3f}"


class TestSystematicVulnerabilities:
    """Test for systematic vulnerabilities in the implementation"""


    def test_timing_side_channel(self) -> None:
    """Test for timing side channels in encoding"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000))

        import time

        # Measure timing for different input patterns
        patterns = [
            np.zeros(1000),  # All zeros
            np.ones(1000),  # All ones
            np.random.randn(1000),  # Random
            np.array([1e10] * 1000),  # Large values
        ]

        timings = []
        num_trials = 100

        for pattern in patterns:
            trial_times = []
            for _ in range(num_trials):
                start = time.perf_counter()
                _ = encoder.encode(pattern, OmicsType.GENOMIC)
                end = time.perf_counter()
                trial_times.append(end - start)

            timings.append(np.mean(trial_times))

        # Timing should not vary significantly with input pattern
        timing_variance = (
            np.var(timings) / np.mean(timings) ** 2
        )  # Coefficient of variation squared

        assert (
            timing_variance < 0.01
        ), f"Timing varies with input pattern: CV²={timing_variance:.4f}"


    def test_error_message_leakage(self) -> None:
    """Test that error messages don't leak sensitive information"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000))

        # Test various invalid inputs
        invalid_inputs = [
            (None, "None input"),
            ("not_an_array", "String input"),
            ([[[1, 2, 3]]], "Deeply nested list"),
            ({"__proto__": "polluted"}, "Prototype pollution attempt"),
        ]

        error_messages = []

        for invalid_input, description in invalid_inputs:
            try:
                _ = encoder.encode(invalid_input, OmicsType.GENOMIC)
            except Exception as e:
                error_messages.append(str(e))

        # Error messages should not contain sensitive information
        for msg in error_messages:
            # Should not contain file paths
            assert (
                "/" not in msg and "\\" not in msg
            ), f"Error message contains path information: {msg}"
            # Should not contain internal state
            assert "0x" not in msg.lower(), f"Error message contains memory addresses: {msg}"


    def test_resource_exhaustion(self) -> None:
    """Test resistance to resource exhaustion attacks"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000))

        # Try to exhaust memory with large inputs
        try:
            # Attempt to create very large feature array
            large_features = np.random.randn(10_000_000)  # 10M features

            # Should handle gracefully
            start_time = time.time()
            _ = encoder.encode(large_features, OmicsType.GENOMIC)
            elapsed = time.time() - start_time

            # Should complete in reasonable time
            assert elapsed < 10, f"Encoding took too long: {elapsed:.2f}s"

        except MemoryError:
            # It's OK to fail with MemoryError
            pass
        except Exception as e:
            # Should not crash with other exceptions
            pytest.fail(f"Unexpected exception: {type(e).__name__}: {e}")


class TestCryptographicProperties:
    """Test cryptographic-like properties of HDC encoding"""


    def test_avalanche_effect(self) -> None:
    """Test that small input changes cause large output changes"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000, seed=42))

        # Original input
        original = np.random.randn(1000)
        original_hv = encoder.encode(original, OmicsType.GENOMIC)

        # Test small perturbations
        epsilons = [1e-10, 1e-5, 1e-3, 1e-1]

        for eps in epsilons:
            perturbed = original.copy()
            perturbed[0] += eps  # Change just one element slightly

            perturbed_hv = encoder.encode(perturbed, OmicsType.GENOMIC)

            # Count how many dimensions changed
            changes = (original_hv != perturbed_hv).float().mean()

            # Even tiny changes should affect many output dimensions
            assert (
                changes > 0.3
            ), f"Insufficient avalanche effect for eps={eps}: only {changes:.2%} dims changed"


    def test_collision_resistance_adversarial(self) -> None:
    """Test collision resistance against adversarial inputs"""
        encoder = HypervectorEncoder(HypervectorConfig(dimension=10000, seed=42))

        # Try to find collisions
        num_attempts = 1000
        encoded_set = set()

        for i in range(num_attempts):
            # Create similar but different inputs
            if i % 2 == 0:
                features = np.full(1000, i / num_attempts)
            else:
                features = np.full(1000, (i - 1) / num_attempts)
                features[0] += 1e-10  # Tiny difference

            hv = encoder.encode(features, OmicsType.GENOMIC)

            # Convert to hashable representation
            hv_bytes = hv.numpy().tobytes()

            # Check for collision
            assert hv_bytes not in encoded_set, f"Collision found at attempt {i}"

            encoded_set.add(hv_bytes)


def run_adversarial_tests() -> None:
    """Run all adversarial tests"""
    test_classes = [
        TestAdversarialInputs,
        TestPrivacyAttacks,
        TestBindingAttacks,
        TestSystematicVulnerabilities,
        TestCryptographicProperties,
    ]

    print("Running HDC Adversarial Tests")
    print("=" * 50)

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        test_instance = test_class()

        # Get all test methods
        test_methods = [method for method in dir(test_instance) if method.startswith("test_")]

        for method_name in test_methods:
            print(f"  - {method_name}...", end=" ")
            try:
                method = getattr(test_instance, method_name)
                method()
                print("PASSED")
            except Exception as e:
                print(f"FAILED: {e}")

    print("\nAdversarial testing complete!")

if __name__ == "__main__":
    run_adversarial_tests()
