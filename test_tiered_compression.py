#!/usr/bin/env python3
"""
Test tiered compression system for size targets and quality metrics.

Verifies:
1. MINI tier compresses to ~25KB
2. CLINICAL tier compresses to ~300KB
3. FULL_HDC tier compresses to 100-200KB
4. Information retention and clinical coverage
"""

import numpy as np
from typing import Dict, Any

from genomevault.compression.tiered_compression import CompressionTier, TieredCompressor
from genomevault.core.constants import OmicsType


def generate_test_data(num_variants: int = 1000000) -> Dict[str, Any]:
    """Generate realistic test genomic data."""
    # Generate variant data with realistic distribution
    variants = {}

    # Common variants (MAF > 0.01)
    for i in range(num_variants // 10):
        rsid = f"rs{1000000 + i}"
        # Hardy-Weinberg equilibrium for common variants
        maf = np.random.uniform(0.01, 0.5)
        genotype_probs = [(1 - maf) ** 2, 2 * maf * (1 - maf), maf**2]
        genotype = np.random.choice([0, 1, 2], p=genotype_probs)
        variants[rsid] = genotype

    # Rare variants (MAF < 0.01)
    for i in range(num_variants // 10, num_variants):
        rsid = f"rs{1000000 + i}"
        # Most are homozygous reference
        if np.random.random() < 0.001:
            genotype = np.random.choice([1, 2])  # Rare allele
        else:
            genotype = 0
        variants[rsid] = genotype

    return {
        "sample_id": "TEST_SAMPLE_001",
        "date": "2024-01-15",
        "variants": variants,
        "quality_scores": np.random.uniform(20, 40, num_variants).tolist(),
        "coverage": np.random.poisson(30, num_variants).tolist(),
        "metadata": {
            "sequencer": "Illumina NovaSeq",
            "reference": "GRCh38",
            "pipeline": "GATK 4.2",
        },
    }


def test_mini_tier():
    """Test MINI tier compression (25KB target)."""
    print("\n" + "=" * 60)
    print("Testing MINI Tier (25KB target)")
    print("=" * 60)

    compressor = TieredCompressor()
    test_data = generate_test_data(100000)  # 100k variants

    compressed, metrics = compressor.compress_to_target(
        test_data, CompressionTier.MINI, OmicsType.GENOMIC
    )

    print(f"\nOriginal data size: {metrics.original_size:,} bytes")
    print(f"Compressed size: {metrics.compressed_size:,} bytes")
    print(f"Target size: {CompressionTier.MINI.target_bytes:,} bytes")
    print(f"Compression ratio: {metrics.compression_ratio:.2f}x")

    # Verify size target
    assert (
        metrics.compressed_size <= CompressionTier.MINI.target_bytes * 1.1
    ), f"MINI compression exceeds target: {metrics.compressed_size} > {CompressionTier.MINI.target_bytes}"

    # Verify quality metrics
    assert (
        metrics.information_retention >= 0.3
    ), f"Information retention too low: {metrics.information_retention}"

    assert (
        metrics.clinical_coverage >= 0.2
    ), f"Clinical coverage too low: {metrics.clinical_coverage}"

    print("\n✅ MINI tier test passed!")
    print(f"  Size: {metrics.compressed_size/1024:.1f} KB (target: 25 KB)")
    print(f"  Information retention: {metrics.information_retention:.1%}")
    print(f"  Clinical coverage: {metrics.clinical_coverage:.1%}")

    return metrics


def test_clinical_tier():
    """Test CLINICAL tier compression (300KB target)."""
    print("\n" + "=" * 60)
    print("Testing CLINICAL Tier (300KB target)")
    print("=" * 60)

    compressor = TieredCompressor()
    test_data = generate_test_data(500000)  # 500k variants

    compressed, metrics = compressor.compress_to_target(
        test_data, CompressionTier.CLINICAL, OmicsType.GENOMIC
    )

    print(f"\nOriginal data size: {metrics.original_size:,} bytes")
    print(f"Compressed size: {metrics.compressed_size:,} bytes")
    print(f"Target size: {CompressionTier.CLINICAL.target_bytes:,} bytes")
    print(f"Compression ratio: {metrics.compression_ratio:.2f}x")

    # Verify size target
    assert (
        metrics.compressed_size <= CompressionTier.CLINICAL.target_bytes * 1.1
    ), f"CLINICAL compression exceeds target: {metrics.compressed_size} > {CompressionTier.CLINICAL.target_bytes}"

    # Verify quality metrics
    assert (
        metrics.information_retention >= 0.7
    ), f"Information retention too low: {metrics.information_retention}"

    assert (
        metrics.clinical_coverage >= 0.9
    ), f"Clinical coverage too low: {metrics.clinical_coverage}"

    print("\n✅ CLINICAL tier test passed!")
    print(f"  Size: {metrics.compressed_size/1024:.1f} KB (target: 300 KB)")
    print(f"  Information retention: {metrics.information_retention:.1%}")
    print(f"  Clinical coverage: {metrics.clinical_coverage:.1%}")

    return metrics


def test_full_hdc_tier():
    """Test FULL_HDC tier compression (100-200KB target)."""
    print("\n" + "=" * 60)
    print("Testing FULL_HDC Tier (100-200KB target)")
    print("=" * 60)

    compressor = TieredCompressor()
    test_data = generate_test_data(1000000)  # 1M variants

    compressed, metrics = compressor.compress_to_target(
        test_data, CompressionTier.FULL_HDC, OmicsType.GENOMIC
    )

    print(f"\nOriginal data size: {metrics.original_size:,} bytes")
    print(f"Compressed size: {metrics.compressed_size:,} bytes")
    print(f"Target size: {CompressionTier.FULL_HDC.target_bytes:,} bytes")
    print(f"Compression ratio: {metrics.compression_ratio:.2f}x")

    # Verify size target (100-200KB range)
    assert (
        100 * 1024 <= metrics.compressed_size <= 200 * 1024
    ), f"FULL_HDC compression outside target range: {metrics.compressed_size}"

    # Verify quality metrics
    assert (
        metrics.information_retention >= 0.85
    ), f"Information retention too low: {metrics.information_retention}"

    assert (
        metrics.clinical_coverage >= 0.75
    ), f"Clinical coverage too low: {metrics.clinical_coverage}"

    print("\n✅ FULL_HDC tier test passed!")
    print(f"  Size: {metrics.compressed_size/1024:.1f} KB (target: 100-200 KB)")
    print(f"  Information retention: {metrics.information_retention:.1%}")
    print(f"  Clinical coverage: {metrics.clinical_coverage:.1%}")

    return metrics


def test_variant_selection():
    """Test variant prioritization for each tier."""
    print("\n" + "=" * 60)
    print("Testing Variant Selection and Prioritization")
    print("=" * 60)

    compressor = TieredCompressor()

    for tier in CompressionTier:
        print(f"\n{tier.tier_name.upper()} Tier Variant Selection:")
        print("-" * 40)

        variants = compressor.select_variants(tier)

        # Analyze variant composition
        acmg_count = sum(1 for v in variants if v.acmg_gene)
        pharmgkb_count = sum(1 for v in variants if v.pharmgkb_level > 0)
        pathogenic_count = sum(1 for v in variants if v.clinical_significance >= 4)
        common_count = sum(1 for v in variants if v.gnomad_af > 0.01)

        print(f"  Total variants: {len(variants):,}")
        print(f"  ACMG genes: {acmg_count:,}")
        print(f"  PharmGKB variants: {pharmgkb_count:,}")
        print(f"  Pathogenic variants: {pathogenic_count:,}")
        print(f"  Common variants (MAF>1%): {common_count:,}")

        # Calculate average priority score
        avg_priority = np.mean([v.priority_score for v in variants])
        print(f"  Average priority score: {avg_priority:.1f}")

        # Verify tier-specific requirements
        if tier == CompressionTier.MINI:
            assert len(variants) == 5000, f"MINI should select 5000 variants, got {len(variants)}"
            assert common_count > 1000, "MINI should prioritize common variants"

        elif tier == CompressionTier.CLINICAL:
            assert len(variants) <= 120000, "CLINICAL should select ≤120k variants"
            assert acmg_count > 0 or pharmgkb_count > 0, "CLINICAL should include ACMG/PharmGKB"

        else:  # FULL_HDC
            assert len(variants) == 10000, "FULL_HDC should select 10000 variants"

    print("\n✅ Variant selection test passed!")
    return True


def test_multi_modal_storage():
    """Test storage calculation for multiple omics modalities."""
    print("\n" + "=" * 60)
    print("Testing Multi-Modal Storage Calculation")
    print("=" * 60)

    compressor = TieredCompressor()

    # Define multi-omics profile
    modalities = [
        (OmicsType.GENOMIC, CompressionTier.CLINICAL),  # 300KB
        (OmicsType.TRANSCRIPTOMIC, CompressionTier.MINI),  # 25KB
        (OmicsType.PROTEOMIC, CompressionTier.MINI),  # 25KB
        (OmicsType.EPIGENOMIC, CompressionTier.FULL_HDC),  # 150KB
        (OmicsType.METABOLOMIC, CompressionTier.MINI),  # 25KB
    ]

    total_storage = compressor.calculate_client_storage(modalities)

    print("\nModality Storage Breakdown:")
    print("-" * 40)

    expected_total = 0
    for omics_type, tier in modalities:
        size_kb = tier.target_bytes / 1024
        print(f"  {omics_type.value:15} ({tier.tier_name:8}): {size_kb:>6.1f} KB")
        expected_total += tier.target_bytes

    print(f"\nTotal storage: {total_storage:,} bytes ({total_storage/1024:.1f} KB)")
    print(f"Expected: {expected_total:,} bytes ({expected_total/1024:.1f} KB)")

    assert (
        total_storage == expected_total
    ), f"Storage calculation mismatch: {total_storage} != {expected_total}"

    # Verify reasonable storage for mobile device
    assert total_storage < 1024 * 1024, f"Total storage exceeds 1MB: {total_storage}"

    print("\n✅ Multi-modal storage test passed!")
    print(f"  Total: {total_storage/1024:.1f} KB (< 1MB ✓)")

    return total_storage


def test_compression_quality():
    """Test compression quality metrics."""
    print("\n" + "=" * 60)
    print("Testing Compression Quality Metrics")
    print("=" * 60)

    compressor = TieredCompressor()
    test_data = generate_test_data(100000)

    print("\nQuality metrics by tier:")
    print("-" * 60)
    print(f"{'Tier':<10} {'Size (KB)':<12} {'Retention':<12} {'Clinical':<12} {'Accuracy':<12}")
    print("-" * 60)

    for tier in CompressionTier:
        compressed, metrics = compressor.compress_to_target(test_data, tier)

        print(
            f"{tier.tier_name:<10} "
            f"{metrics.compressed_size/1024:>8.1f} KB  "
            f"{metrics.information_retention:>10.1%}  "
            f"{metrics.clinical_coverage:>10.1%}  "
            f"{metrics.reconstruction_accuracy:>10.1%}"
        )

        # Verify minimum quality standards
        assert (
            metrics.information_retention >= 0.3
        ), f"{tier.tier_name}: Information retention below minimum"

        assert (
            metrics.reconstruction_accuracy >= 0.7
        ), f"{tier.tier_name}: Reconstruction accuracy below minimum"

    print("\n✅ Quality metrics test passed!")
    return True


def main():
    """Run all compression tests."""
    print("\n" + "=" * 70)
    print("  GENOMEVAULT TIERED COMPRESSION TEST SUITE")
    print("=" * 70)

    tests = [
        ("MINI Tier", test_mini_tier),
        ("CLINICAL Tier", test_clinical_tier),
        ("FULL_HDC Tier", test_full_hdc_tier),
        ("Variant Selection", test_variant_selection),
        ("Multi-Modal Storage", test_multi_modal_storage),
        ("Compression Quality", test_compression_quality),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, True, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed: {e}")
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed, result in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:20}: {status}")
        all_passed &= passed

    if all_passed:
        print("\n🎉 All compression tests passed!")
        print("   ✓ MINI tier: ~25KB")
        print("   ✓ CLINICAL tier: ~300KB")
        print("   ✓ FULL_HDC tier: 100-200KB")
        print("   ✓ Quality metrics preserved")
        print("   ✓ Multi-modal storage < 1MB")
    else:
        print("\n⚠️  Some tests failed. Check output above.")

    print("=" * 70)


if __name__ == "__main__":
    main()
