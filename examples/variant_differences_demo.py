#!/usr/bin/env python3
"""
Variant Difference Computation Demo

Demonstrates the variant difference computation functionality from Section 5.1
of the differential encoding specification. Shows how to:
1. Create experimental and reference genome sections
2. Compute variant differences
3. Analyze the three types of differences (new, missing, genotype)
4. Verify performance with large variant sets
"""

import time
from typing import List

from genomevault.differential_encoding import (
    Variant,
    GenomeSection,
    DifferenceType,
    FunctionalImpact,
    VariantDifference,
    compute_variant_differences,
    get_functional_impact,
)


def create_sample_variants() -> tuple[List[Variant], List[Variant]]:
    """
    Create sample experimental and reference variants for demonstration.

    Returns:
        Tuple of (experimental_variants, reference_variants)
    """
    # Experimental variants (patient genome)
    experimental = [
        # New mutation - only in experimental
        Variant(
            chromosome="chr1",
            position=100000,
            ref="A",
            alt="G",
            genotype="0/1",
            quality=99.0,
            info={"Consequence": "missense_variant", "Gene": "BRCA1"}
        ),

        # Shared variant with same genotype
        Variant(
            chromosome="chr1",
            position=150000,
            ref="C",
            alt="T",
            genotype="0/1",
            quality=99.0,
            info={"Consequence": "synonymous_variant"}
        ),

        # Shared variant with different genotype
        Variant(
            chromosome="chr1",
            position=200000,
            ref="G",
            alt="A",
            genotype="1/1",  # Homozygous in experimental
            quality=99.0,
            info={"Consequence": "stop_gained", "IMPACT": "HIGH"}
        ),

        # Another new mutation
        Variant(
            chromosome="chr1",
            position=250000,
            ref="T",
            alt="C",
            genotype="0/1",
            quality=95.0,
            info={"Consequence": "intergenic_variant"}
        ),
    ]

    # Reference variants (population reference)
    reference = [
        # Shared variant with same genotype
        Variant(
            chromosome="chr1",
            position=150000,
            ref="C",
            alt="T",
            genotype="0/1",
            quality=99.0,
            info={"Consequence": "synonymous_variant"}
        ),

        # Shared variant with different genotype
        Variant(
            chromosome="chr1",
            position=200000,
            ref="G",
            alt="A",
            genotype="0/1",  # Heterozygous in reference
            quality=99.0,
            info={"Consequence": "stop_gained", "IMPACT": "HIGH"}
        ),

        # Missing from experimental
        Variant(
            chromosome="chr1",
            position=300000,
            ref="A",
            alt="T",
            genotype="0/1",
            quality=98.0,
            info={"Consequence": "frameshift_variant", "IMPACT": "HIGH"}
        ),

        # Another missing variant
        Variant(
            chromosome="chr1",
            position=350000,
            ref="C",
            alt="G",
            genotype="1/1",
            quality=99.0,
            info={"Consequence": "intronic"}
        ),
    ]

    return experimental, reference


def analyze_differences(differences: List[VariantDifference]) -> None:
    """
    Analyze and print details about variant differences.

    Args:
        differences: List of VariantDifference objects
    """
    # Count by type
    new_count = sum(1 for d in differences if d.is_new_mutation)
    missing_count = sum(1 for d in differences if d.is_missing)
    genotype_count = sum(1 for d in differences if d.is_genotype_diff)

    print(f"\n📊 Difference Summary:")
    print(f"  Total differences: {len(differences)}")
    print(f"  New mutations: {new_count}")
    print(f"  Missing variants: {missing_count}")
    print(f"  Genotype differences: {genotype_count}")

    # Count by functional impact
    impact_counts = {}
    for d in differences:
        impact = d.functional_impact.value
        impact_counts[impact] = impact_counts.get(impact, 0) + 1

    print(f"\n🎯 Functional Impact Distribution:")
    for impact, count in sorted(impact_counts.items()):
        print(f"  {impact}: {count}")

    # Show detailed information for each difference
    print(f"\n📝 Detailed Differences:")
    for i, diff in enumerate(differences, 1):
        print(f"\n  {i}. {diff.difference_type.value.upper()}")
        print(f"     Position: {diff.chromosome}:{diff.position}")

        if diff.is_new_mutation:
            print(f"     New mutation: {diff.exp_ref}>{diff.exp_alt}")
            print(f"     Genotype: {diff.exp_genotype}")
            print(f"     Impact: {diff.functional_impact.value}")
            if "Gene" in diff.metadata:
                print(f"     Gene: {diff.metadata['Gene']}")

        elif diff.is_missing:
            print(f"     Missing variant: {diff.ref_ref}>{diff.ref_alt}")
            print(f"     Reference genotype: {diff.ref_genotype}")
            print(f"     Impact: {diff.functional_impact.value}")

        elif diff.is_genotype_diff:
            print(f"     Variant: {diff.exp_ref}>{diff.exp_alt}")
            print(f"     Experimental genotype: {diff.exp_genotype}")
            print(f"     Reference genotype: {diff.ref_genotype}")
            print(f"     Impact: {diff.functional_impact.value}")


def performance_test(num_variants: int = 10000) -> None:
    """
    Test performance with large variant sets.

    Args:
        num_variants: Number of variants to generate for testing
    """
    print(f"\n⚡ Performance Test ({num_variants:,} variants)")

    # Generate large variant sets
    exp_variants = [
        Variant(
            chromosome="chr1",
            position=1000 + i * 100,
            ref="A",
            alt="G",
            genotype="0/1",
            quality=99.0
        )
        for i in range(num_variants)
    ]

    # Create reference with 50% overlap
    ref_variants = [
        Variant(
            chromosome="chr1",
            position=1000 + i * 100,
            ref="A",
            alt="G",
            genotype="0/1" if i % 2 == 0 else "1/1",  # Some genotype diffs
            quality=99.0
        )
        for i in range(num_variants // 2)
    ]

    # Add some unique reference variants
    ref_variants.extend([
        Variant(
            chromosome="chr1",
            position=2000000 + i * 100,
            ref="C",
            alt="T",
            genotype="0/1",
            quality=99.0
        )
        for i in range(num_variants // 4)
    ])

    # Create genome sections
    exp_section = GenomeSection(
        chromosome="chr1",
        start_position=1000,
        end_position=3000000,
        variants=exp_variants
    )

    ref_section = GenomeSection(
        chromosome="chr1",
        start_position=1000,
        end_position=3000000,
        variants=ref_variants
    )

    # Measure computation time
    start_time = time.time()
    differences = compute_variant_differences(exp_section, ref_section)
    elapsed_time = time.time() - start_time

    print(f"  Experimental variants: {len(exp_variants):,}")
    print(f"  Reference variants: {len(ref_variants):,}")
    print(f"  Computation time: {elapsed_time:.3f} seconds")
    print(f"  Differences found: {len(differences):,}")
    print(f"  Throughput: {(len(exp_variants) + len(ref_variants)) / elapsed_time:,.0f} variants/sec")

    # Verify correctness
    new_count = sum(1 for d in differences if d.is_new_mutation)
    missing_count = sum(1 for d in differences if d.is_missing)
    genotype_count = sum(1 for d in differences if d.is_genotype_diff)

    print(f"\n  Results breakdown:")
    print(f"    New mutations: {new_count:,}")
    print(f"    Missing variants: {missing_count:,}")
    print(f"    Genotype differences: {genotype_count:,}")


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("Variant Difference Computation Demo")
    print("Section 5.1: Differential Encoding")
    print("=" * 70)

    # Create sample data
    print("\n🧬 Creating sample genome sections...")
    exp_variants, ref_variants = create_sample_variants()

    exp_section = GenomeSection(
        chromosome="chr1",
        start_position=100000,
        end_position=400000,
        variants=exp_variants
    )

    ref_section = GenomeSection(
        chromosome="chr1",
        start_position=100000,
        end_position=400000,
        variants=ref_variants
    )

    print(f"  Experimental variants: {len(exp_variants)}")
    print(f"  Reference variants: {len(ref_variants)}")

    # Compute differences
    print("\n🔍 Computing variant differences...")
    differences = compute_variant_differences(exp_section, ref_section)

    # Analyze results
    analyze_differences(differences)

    # Demonstrate functional impact prediction
    print("\n" + "=" * 70)
    print("Functional Impact Prediction")
    print("=" * 70)

    test_variants = [
        ("Stop gain", Variant("chr1", 1000, "C", "T", info={"Consequence": "stop_gained"})),
        ("Frameshift", Variant("chr1", 2000, "ATG", "A")),  # -2 bp, not divisible by 3
        ("Missense", Variant("chr1", 3000, "G", "A", info={"Consequence": "missense_variant"})),
        ("Synonymous", Variant("chr1", 4000, "T", "C", info={"Consequence": "synonymous_variant"})),
        ("Intergenic", Variant("chr1", 5000, "A", "G", info={"Consequence": "intergenic_variant"})),
    ]

    for name, variant in test_variants:
        impact = get_functional_impact(variant)
        print(f"  {name}: {impact.value}")

    # Performance test
    print("\n" + "=" * 70)
    performance_test(10000)

    print("\n" + "=" * 70)
    print("✅ Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
