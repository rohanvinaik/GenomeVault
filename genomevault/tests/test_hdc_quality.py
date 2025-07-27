"""
Test HDC compression quality and similarity preservation.

Quantifies the quality loss from hyperdimensional computing compression
to ensure clinical validity.
"""
import itertools
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from genomevault.hypervector_transform.encoding import HypervectorEncoder
from genomevault.hypervector_transform.registry import HypervectorRegistry
from genomevault.utils.metrics import get_metrics

metrics = get_metrics()


class TestHDCQuality:
    """Quantify HDC compression quality loss."""
    """Quantify HDC compression quality loss."""
    """Quantify HDC compression quality loss."""

    @pytest.fixture
    def encoder(self) -> None:
        """TODO: Add docstring for encoder"""
            """TODO: Add docstring for encoder"""
                """TODO: Add docstring for encoder"""
    """Get encoder

    TODO: This is a duplicate getter function that needs proper implementation.
    Consider refactoring to use a common registry or factory pattern.
    """
        """Create test encoder."""
        return HypervectorEncoder(dimension=10000, seed=42)

    @pytest.fixture
        def registry(self) -> None:
            """TODO: Add docstring for registry"""
                """TODO: Add docstring for registry"""
                    """TODO: Add docstring for registry"""
    """Get registry

    TODO: This is a duplicate getter function that needs proper implementation.
    Consider refactoring to use a common registry or factory pattern.
    """
        """Create test registry."""
        return HypervectorRegistry(registry_path="./test_registry.json")

            def test_variant_similarity_preservation(self, encoder) -> None:
                """TODO: Add docstring for test_variant_similarity_preservation"""
                    """TODO: Add docstring for test_variant_similarity_preservation"""
                        """TODO: Add docstring for test_variant_similarity_preservation"""
    """Test that similar variants remain similar after HDC encoding."""
        # Create pairs of variants with known edit distances
        variant_pairs = [
            # (variant1, variant2, edit_distance)
            (
                {"chr": "1", "pos": 12345, "ref": "A", "alt": "G"},
                {"chr": "1", "pos": 12345, "ref": "A", "alt": "T"},
                1,
            ),  # Same position, different alt
            (
                {"chr": "1", "pos": 12345, "ref": "A", "alt": "G"},
                {"chr": "1", "pos": 12346, "ref": "A", "alt": "G"},
                1,
            ),  # Adjacent position
            (
                {"chr": "1", "pos": 12345, "ref": "A", "alt": "G"},
                {"chr": "1", "pos": 12445, "ref": "A", "alt": "G"},
                100,
            ),  # 100bp apart
            (
                {"chr": "1", "pos": 12345, "ref": "A", "alt": "G"},
                {"chr": "2", "pos": 12345, "ref": "A", "alt": "G"},
                1000,
            ),  # Different chromosome
        ]

        results = []

        for var1, var2, expected_distance in variant_pairs:
            # Encode variants
            hv1 = encoder.encode_variant(var1)
            hv2 = encoder.encode_variant(var2)

            # Compute similarity (cosine)
            similarity = encoder.compute_similarity(hv1, hv2)

            # Record result
            result = {
                "variant_pair": (var1, var2),
                "edit_distance": expected_distance,
                "hv_similarity": similarity,
                "preserved": similarity > 0.5 if expected_distance < 10 else similarity < 0.5,
            }
            results.append(result)

            # Log metrics
            metrics.record(
                "hdc_similarity_test", similarity, metadata={"edit_distance": expected_distance}
            )

        # Assertions
        preservation_rate = sum(r["preserved"] for r in results) / len(results)
        assert (
            preservation_rate >= 0.9
        ), f"Similarity preservation rate too low: {preservation_rate}"

        # Check monotonicity: larger edit distances should have lower similarity
        sorted_results = sorted(results, key=lambda x: x["edit_distance"])
        similarities = [r["hv_similarity"] for r in sorted_results]

        # Check general trend (allow some variation)
        decreasing_count = sum(
            similarities[i] >= similarities[i + 1] for i in range(len(similarities) - 1)
        )
        monotonicity = decreasing_count / (len(similarities) - 1)

        assert monotonicity >= 0.7, f"Similarity not monotonic with distance: {monotonicity}"

        return results

            def test_compression_ratio(self, encoder) -> None:
                """TODO: Add docstring for test_compression_ratio"""
                    """TODO: Add docstring for test_compression_ratio"""
                        """TODO: Add docstring for test_compression_ratio"""
    """Test actual compression ratio achieved."""
        # Create realistic variant set
        num_variants = 1000
        variants = []

        # Generate variants with realistic distribution
        for i in range(num_variants):
            chr_num = np.random.choice(range(1, 23))  # Chromosomes 1-22
            position = np.random.randint(1, 250_000_000)  # Typical chromosome length
            ref = np.random.choice(["A", "C", "G", "T"])
            alt = np.random.choice([n for n in ["A", "C", "G", "T"] if n != ref])

            variant = {
                "chr": str(chr_num),
                "pos": position,
                "ref": ref,
                "alt": alt,
                "quality": np.random.rand() * 100,
            }
            variants.append(variant)

        # Calculate original size (VCF-like representation)
        original_size = 0
        for var in variants:
            # Typical VCF line: CHROM POS ID REF ALT QUAL FILTER INFO
            vcf_line = f"{var['chr']}\t{var['pos']}\t.\t{var['ref']}\t{var['alt']}\t{var['quality']:.1f}\tPASS\t.\n"
            original_size += len(vcf_line.encode())

        # Encode all variants
        hypervectors = []
        with metrics.time_operation("hdc_batch_encoding"):
            for var in variants:
                hv = encoder.encode_variant(var)
                hypervectors.append(hv)

        # Calculate compressed size
        hv_array = np.array(hypervectors)

        # Different storage strategies
        storage_strategies = {
            "float32": hv_array.astype(np.float32).nbytes,
            "float16": hv_array.astype(np.float16).nbytes,
            "int8_quantized": self._quantize_to_int8(hv_array).nbytes,
            "sparse": self._calculate_sparse_size(hv_array),
            "binary": np.packbits((hv_array > 0).astype(np.uint8)).nbytes,
        }

        # Calculate compression ratios
        compression_ratios = {}
        for strategy, compressed_size in storage_strategies.items():
            ratio = original_size / compressed_size
            compression_ratios[strategy] = ratio

            metrics.record("hdc_compression_ratio", ratio, metadata={"strategy": strategy})

        # Test assertions
        best_ratio = max(compression_ratios.values())
        assert best_ratio >= 100, f"Best compression ratio too low: {best_ratio:.1f}:1"

        # Binary should achieve close to 1000:1
        binary_ratio = compression_ratios["binary"]
        assert binary_ratio >= 500, f"Binary compression too low: {binary_ratio:.1f}:1"

        return {
            "original_size_bytes": original_size,
            "num_variants": num_variants,
            "compression_ratios": compression_ratios,
            "best_strategy": max(compression_ratios, key=compression_ratios.get),
        }

            def test_discrimination_ability(self, encoder) -> None:
                """TODO: Add docstring for test_discrimination_ability"""
                    """TODO: Add docstring for test_discrimination_ability"""
                        """TODO: Add docstring for test_discrimination_ability"""
    """Test ability to discriminate between different variants."""
        # Create sets of similar and dissimilar variants
        base_variant = {"chr": "1", "pos": 100000, "ref": "A", "alt": "G"}

        # Similar variants (should have high similarity)
        similar_variants = [
            {"chr": "1", "pos": 100001, "ref": "A", "alt": "G"},  # Adjacent
            {"chr": "1", "pos": 100010, "ref": "A", "alt": "G"},  # Nearby
            {"chr": "1", "pos": 100000, "ref": "A", "alt": "T"},  # Same pos, different alt
        ]

        # Dissimilar variants (should have low similarity)
        dissimilar_variants = [
            {"chr": "10", "pos": 100000, "ref": "A", "alt": "G"},  # Different chr
            {"chr": "1", "pos": 200000, "ref": "C", "alt": "T"},  # Far away
            {"chr": "X", "pos": 50000, "ref": "G", "alt": "A"},  # Sex chromosome
        ]

        # Encode all
        base_hv = encoder.encode_variant(base_variant)

        similar_similarities = []
        for var in similar_variants:
            hv = encoder.encode_variant(var)
            sim = encoder.compute_similarity(base_hv, hv)
            similar_similarities.append(sim)

        dissimilar_similarities = []
        for var in dissimilar_variants:
            hv = encoder.encode_variant(var)
            sim = encoder.compute_similarity(base_hv, hv)
            dissimilar_similarities.append(sim)

        # Statistical tests
        avg_similar = np.mean(similar_similarities)
        avg_dissimilar = np.mean(dissimilar_similarities)

        # Should have clear separation
        assert (
            avg_similar > avg_dissimilar + 0.2
        ), f"Insufficient discrimination: similar={avg_similar:.3f}, dissimilar={avg_dissimilar:.3f}"

        # All similar should be > 0.5
        assert all(
            s > 0.5 for s in similar_similarities
        ), f"Some similar variants have low similarity: {similar_similarities}"

        # Most dissimilar should be < 0.3
        low_dissimilar = sum(s < 0.3 for s in dissimilar_similarities)
        assert (
            low_dissimilar >= len(dissimilar_similarities) * 0.8
        ), f"Dissimilar variants not sufficiently separated"

        return {
            "avg_similar_similarity": avg_similar,
            "avg_dissimilar_similarity": avg_dissimilar,
            "discrimination_gap": avg_similar - avg_dissimilar,
            "similar_similarities": similar_similarities,
            "dissimilar_similarities": dissimilar_similarities,
        }

            def test_seed_reproducibility(self, registry) -> None:
                """TODO: Add docstring for test_seed_reproducibility"""
                    """TODO: Add docstring for test_seed_reproducibility"""
                        """TODO: Add docstring for test_seed_reproducibility"""
    """Test that same seed produces identical encodings."""
        # Register version with specific seed
        test_seed = 12345
        registry.register_version(
            version="test_v1",
            params={"dimension": 5000, "projection_type": "gaussian", "seed": test_seed},
            description="Test version for reproducibility",
        )

        # Create two encoders with same version
        encoder1 = registry.get_encoder("test_v1")
        encoder2 = registry.get_encoder("test_v1")

        # Encode same variant multiple times
        variant = {"chr": "1", "pos": 12345, "ref": "A", "alt": "G"}

        hv1_1 = encoder1.encode_variant(variant)
        hv1_2 = encoder1.encode_variant(variant)
        hv2_1 = encoder2.encode_variant(variant)
        hv2_2 = encoder2.encode_variant(variant)

        # All should be identical
        assert np.allclose(hv1_1, hv1_2), "Same encoder produces different results"
        assert np.allclose(hv1_1, hv2_1), "Different encoder instances produce different results"
        assert np.allclose(hv1_1, hv2_2), "Encodings not reproducible"

        # Test with different seed
        registry.register_version(
            version="test_v2",
            params={"dimension": 5000, "projection_type": "gaussian", "seed": test_seed + 1},
        )

        encoder3 = registry.get_encoder("test_v2")
        hv3 = encoder3.encode_variant(variant)

        # Should be different with different seed
        assert not np.allclose(hv1_1, hv3), "Different seeds produce same encoding"

        return True

                def test_clinical_variant_preservation(self, encoder) -> None:
                    """TODO: Add docstring for test_clinical_variant_preservation"""
                        """TODO: Add docstring for test_clinical_variant_preservation"""
                            """TODO: Add docstring for test_clinical_variant_preservation"""
    """Test preservation of clinically relevant variant properties."""
        # Pathogenic vs benign variants
        pathogenic_variants = [
            {"chr": "17", "pos": 41276045, "ref": "T", "alt": "C", "gene": "BRCA1"},  # BRCA1
            {"chr": "13", "pos": 32953886, "ref": "G", "alt": "A", "gene": "BRCA2"},  # BRCA2
            {
                "chr": "7",
                "pos": 117559590,
                "ref": "G",
                "alt": "A",
                "gene": "CFTR",
            },  # Cystic fibrosis
        ]

        benign_variants = [
            {"chr": "1", "pos": 1000000, "ref": "A", "alt": "G", "gene": "intergenic"},
            {"chr": "2", "pos": 2000000, "ref": "C", "alt": "T", "gene": "intergenic"},
            {"chr": "3", "pos": 3000000, "ref": "G", "alt": "T", "gene": "intergenic"},
        ]

        # Encode all variants
        pathogenic_hvs = [encoder.encode_variant(v) for v in pathogenic_variants]
        benign_hvs = [encoder.encode_variant(v) for v in benign_variants]

        # Test within-class similarity
        pathogenic_similarities = []
        for i, j in itertools.combinations(range(len(pathogenic_hvs)), 2):
            sim = encoder.compute_similarity(pathogenic_hvs[i], pathogenic_hvs[j])
            pathogenic_similarities.append(sim)

        benign_similarities = []
        for i, j in itertools.combinations(range(len(benign_hvs)), 2):
            sim = encoder.compute_similarity(benign_hvs[i], benign_hvs[j])
            benign_similarities.append(sim)

        # Test between-class similarity
        cross_similarities = []
        for p_hv in pathogenic_hvs:
            for b_hv in benign_hvs:
                sim = encoder.compute_similarity(p_hv, b_hv)
                cross_similarities.append(sim)

        # Analyze results
        results = {
            "avg_pathogenic_similarity": (
                np.mean(pathogenic_similarities) if pathogenic_similarities else 0
            ),
            "avg_benign_similarity": np.mean(benign_similarities) if benign_similarities else 0,
            "avg_cross_similarity": np.mean(cross_similarities),
            "separation_score": (np.mean(pathogenic_similarities) + np.mean(benign_similarities))
            / 2
            - np.mean(cross_similarities),
        }

        # Record metrics
        metrics.record("clinical_variant_separation", results["separation_score"])

        return results

                def _quantize_to_int8(self, hv_array: np.ndarray) -> np.ndarray:
                    """TODO: Add docstring for _quantize_to_int8"""
                        """TODO: Add docstring for _quantize_to_int8"""
                            """TODO: Add docstring for _quantize_to_int8"""
    """Quantize hypervectors to int8 for compression."""
        # Scale to [-127, 127] range
        scaled = hv_array * 127 / np.max(np.abs(hv_array))
        return scaled.astype(np.int8)

                    def _calculate_sparse_size(self, hv_array: np.ndarray, threshold: float = 0.01) -> int:
                        """TODO: Add docstring for _calculate_sparse_size"""
                            """TODO: Add docstring for _calculate_sparse_size"""
                                """TODO: Add docstring for _calculate_sparse_size"""
    """Calculate size when stored as sparse matrix."""
        # Count non-zero elements
        mask = np.abs(hv_array) > threshold
        nnz = np.sum(mask)

        # Sparse format: (row_idx, col_idx, value)
        # 4 bytes each for indices, 4 bytes for float value
        sparse_size = nnz * (4 + 4 + 4)

        return sparse_size

                        def test_batch_processing_efficiency(self, encoder) -> None:
                            """TODO: Add docstring for test_batch_processing_efficiency"""
                                """TODO: Add docstring for test_batch_processing_efficiency"""
                                    """TODO: Add docstring for test_batch_processing_efficiency"""
    """Test efficiency of batch variant encoding."""
        # Create variant batches of different sizes
        batch_sizes = [10, 100, 1000]
        results = {}

        for batch_size in batch_sizes:
            # Generate variants
            variants = []
            for i in range(batch_size):
                variant = {
                    "chr": str(np.random.randint(1, 23)),
                    "pos": np.random.randint(1, 250_000_000),
                    "ref": np.random.choice(["A", "C", "G", "T"]),
                    "alt": np.random.choice(["A", "C", "G", "T"]),
                }
                variants.append(variant)

            # Time individual encoding
            with metrics.time_operation(f"individual_encoding_{batch_size}"):
                individual_hvs = []
                for var in variants:
                    hv = encoder.encode_variant(var)
                    individual_hvs.append(hv)

            # Time batch encoding (if available)
            with metrics.time_operation(f"batch_encoding_{batch_size}"):
                if hasattr(encoder, "encode_variants_batch"):
                    batch_hvs = encoder.encode_variants_batch(variants)
                else:
                    # Simulate batch processing
                    batch_hvs = np.array([encoder.encode_variant(v) for v in variants])

            # Verify results are identical
            if hasattr(encoder, "encode_variants_batch"):
                assert np.allclose(
                    individual_hvs, batch_hvs
                ), "Batch encoding differs from individual"

            results[batch_size] = {
                "variants_encoded": batch_size,
                "individual_time": metrics.get_summary()
                .get(f"individual_encoding_{batch_size}_time", {})
                .get("mean", 0),
                "batch_time": metrics.get_summary()
                .get(f"batch_encoding_{batch_size}_time", {})
                .get("mean", 0),
            }

        return results


                def run_hdc_quality_assessment() -> None:
                    """TODO: Add docstring for run_hdc_quality_assessment"""
                        """TODO: Add docstring for run_hdc_quality_assessment"""
                            """TODO: Add docstring for run_hdc_quality_assessment"""
    """Run comprehensive HDC quality assessment."""
    print("Running HDC Quality Assessment")
    print("=" * 50)

    # Initialize test class
    test = TestHDCQuality()
    encoder = HypervectorEncoder(dimension=10000, seed=42)
    registry = HypervectorRegistry()

    # 1. Similarity preservation
    print("\n1. Testing similarity preservation...")
    similarity_results = test.test_variant_similarity_preservation(encoder)
    print(
        f"   Preservation rate: {sum(r['preserved'] for r in similarity_results) / len(similarity_results):.2%}"
    )

    # 2. Compression ratio
    print("\n2. Testing compression ratio...")
    compression_results = test.test_compression_ratio(encoder)
    print(f"   Best compression: {max(compression_results['compression_ratios'].values()):.1f}:1")
    print(f"   Best strategy: {compression_results['best_strategy']}")

    # 3. Discrimination ability
    print("\n3. Testing discrimination ability...")
    discrimination_results = test.test_discrimination_ability(encoder)
    print(f"   Discrimination gap: {discrimination_results['discrimination_gap']:.3f}")

    # 4. Reproducibility
    print("\n4. Testing reproducibility...")
    reproducible = test.test_seed_reproducibility(registry)
    print(f"   Reproducible: {reproducible}")

    # 5. Clinical variant preservation
    print("\n5. Testing clinical variant preservation...")
    clinical_results = test.test_clinical_variant_preservation(encoder)
    print(f"   Separation score: {clinical_results['separation_score']:.3f}")

    # 6. Batch processing
    print("\n6. Testing batch processing efficiency...")
    batch_results = test.test_batch_processing_efficiency(encoder)
    for size, results in batch_results.items():
        if results["batch_time"] > 0:
            speedup = results["individual_time"] / results["batch_time"]
            print(f"   Batch size {size}: {speedup:.2f}x speedup")

    # Export metrics
    print("\n" + "=" * 50)
    metrics_file = metrics.export_json("hdc_quality_metrics.json")
    print(f"Metrics saved to: {metrics_file}")

    return True


if __name__ == "__main__":
    run_hdc_quality_assessment()
