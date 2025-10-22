#!/usr/bin/env python3
"""
Feature Vector Construction Demo

Demonstrates the feature vector construction functionality from Section 6.1
of the specification. Shows how to:
1. Convert variant differences into 95-dimensional feature vectors
2. Use sinusoidal position encoding
3. Compute functional impact vectors
4. Analyze feature vector components
5. Integrate with the complete differential encoding pipeline
"""

import numpy as np
from genomevault.differential_encoding import (
    Variant,
    GenomeSection,
    compute_variant_differences,
    differences_to_feature_vector,
    sinusoidal_position_encoding,
    compute_functional_impact_vector,
    compute_allele_composition,
    compute_genotype_distribution,
    compute_quality_metrics,
    get_feature_names,
    describe_feature_vector,
    TOTAL_FEATURE_DIM,
    DifferenceType,
    FunctionalImpact,
)


def demo_position_encoding():
    """Demonstrate sinusoidal position encoding."""
    print("=" * 70)
    print("1. Sinusoidal Position Encoding")
    print("=" * 70)

    # Encode a genomic position
    position = 117_150_000  # Within CFTR gene
    encoding = sinusoidal_position_encoding(position, dimension=64)

    print(f"\n📍 Position: {position:,} bp")
    print(f"Encoding dimension: {encoding.shape[0]}")
    print(f"Value range: [{encoding.min():.3f}, {encoding.max():.3f}]")
    print(f"First 10 values: {encoding[:10]}")

    # Compare nearby and distant positions
    enc1 = sinusoidal_position_encoding(100_000, dimension=64)
    enc2 = sinusoidal_position_encoding(100_100, dimension=64)
    enc3 = sinusoidal_position_encoding(1_000_000, dimension=64)

    # Cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_nearby = cosine_similarity(enc1, enc2)
    sim_distant = cosine_similarity(enc1, enc3)

    print(f"\n🔍 Position Similarity:")
    print(f"  100,000 vs 100,100 (100bp apart): {sim_nearby:.4f}")
    print(f"  100,000 vs 1,000,000 (900kb apart): {sim_distant:.4f}")
    print(f"  Nearby positions are more similar!")


def demo_component_features():
    """Demonstrate individual feature components."""
    print("\n" + "=" * 70)
    print("2. Feature Vector Components")
    print("=" * 70)

    # Create sample variant differences
    from genomevault.differential_encoding import VariantDifference

    differences = [
        VariantDifference(
            difference_type=DifferenceType.NEW_MUTATION,
            chromosome="chr17",
            position=43_050_000,
            exp_ref="A",
            exp_alt="G",
            exp_genotype="0/1",
            exp_quality=99.0,
            functional_impact=FunctionalImpact.HIGH,
        ),
        VariantDifference(
            difference_type=DifferenceType.NEW_MUTATION,
            chromosome="chr17",
            position=43_055_000,
            exp_ref="C",
            exp_alt="T",
            exp_genotype="1/1",
            exp_quality=98.0,
            functional_impact=FunctionalImpact.MODERATE,
        ),
        VariantDifference(
            difference_type=DifferenceType.MISSING,
            chromosome="chr17",
            position=43_060_000,
            ref_ref="G",
            ref_alt="A",
            ref_genotype="0/1",
            ref_quality=95.0,
            functional_impact=FunctionalImpact.LOW,
        ),
        VariantDifference(
            difference_type=DifferenceType.GENOTYPE_DIFF,
            chromosome="chr17",
            position=43_065_000,
            exp_ref="T",
            exp_alt="C",
            exp_genotype="1/1",
            exp_quality=97.0,
            ref_ref="T",
            ref_alt="C",
            ref_genotype="0/1",
            ref_quality=96.0,
            functional_impact=FunctionalImpact.MODERATE,
        ),
    ]

    # Compute each component
    print(f"\n📊 Components for {len(differences)} variants:")

    # Allele composition
    allele_comp = compute_allele_composition(differences)
    print(f"\n  Allele Composition (8D):")
    print(f"    Ref: A={allele_comp[0]:.2f}, C={allele_comp[1]:.2f}, "
          f"G={allele_comp[2]:.2f}, T={allele_comp[3]:.2f}")
    print(f"    Alt: A={allele_comp[4]:.2f}, C={allele_comp[5]:.2f}, "
          f"G={allele_comp[6]:.2f}, T={allele_comp[7]:.2f}")

    # Genotype distribution
    geno_dist = compute_genotype_distribution(differences)
    print(f"\n  Genotype Distribution (5D):")
    print(f"    0/0: {geno_dist[0]:.2f}, 0/1: {geno_dist[1]:.2f}, "
          f"1/1: {geno_dist[2]:.2f}")
    print(f"    1/2: {geno_dist[3]:.2f}, other: {geno_dist[4]:.2f}")

    # Functional impact
    impact_vec = compute_functional_impact_vector(differences)
    print(f"\n  Functional Impact (10D):")
    print(f"    HIGH: {impact_vec[0]:.2f}, MODERATE: {impact_vec[1]:.2f}, "
          f"LOW: {impact_vec[2]:.2f}")
    print(f"    Average impact score: {impact_vec[5]:.2f}")
    print(f"    High/Moderate fraction: {impact_vec[7]:.2f}")
    print(f"    Indel fraction: {impact_vec[9]:.2f}")

    # Quality metrics
    quality = compute_quality_metrics(differences)
    print(f"\n  Quality Metrics (5D):")
    print(f"    Mean: {quality[0]:.2f}, Std: {quality[1]:.2f}")
    print(f"    Min: {quality[2]:.2f}, Max: {quality[3]:.2f}, "
          f"Median: {quality[4]:.2f}")


def demo_full_feature_vector():
    """Demonstrate complete feature vector generation."""
    print("\n" + "=" * 70)
    print("3. Complete Feature Vector Generation")
    print("=" * 70)

    # Create experimental and reference sections
    exp_section = GenomeSection(
        chromosome="chr17",
        start_position=43_044_295,  # BRCA1 start
        end_position=43_125_483,    # BRCA1 end
        variants=[
            Variant(chromosome="chr17", position=43_050_000, ref="A", alt="G",
                   genotype="0/1", quality=99.0,
                   info={"Consequence": "missense_variant", "Gene": "BRCA1"}),
            Variant(chromosome="chr17", position=43_055_000, ref="C", alt="T",
                   genotype="1/1", quality=98.0,
                   info={"IMPACT": "HIGH", "Consequence": "stop_gained"}),
            Variant(chromosome="chr17", position=43_060_000, ref="G", alt="A",
                   genotype="0/1", quality=97.0,
                   info={"Consequence": "synonymous_variant"}),
            Variant(chromosome="chr17", position=43_065_000, ref="T", alt="C",
                   genotype="1/1", quality=96.0,
                   info={"IMPACT": "MODERATE"}),
        ],
    )

    ref_section = GenomeSection(
        chromosome="chr17",
        start_position=43_044_295,
        end_position=43_125_483,
        variants=[
            Variant(chromosome="chr17", position=43_052_000, ref="C", alt="G",
                   genotype="0/1", quality=95.0),
            Variant(chromosome="chr17", position=43_060_000, ref="G", alt="A",
                   genotype="0/1", quality=97.0,
                   info={"Consequence": "synonymous_variant"}),
            Variant(chromosome="chr17", position=43_065_000, ref="T", alt="C",
                   genotype="0/1", quality=96.0,
                   info={"IMPACT": "MODERATE"}),
        ],
    )

    # Compute differences
    differences = compute_variant_differences(exp_section, ref_section)

    print(f"\n📈 Computed {len(differences)} differences:")
    for diff in differences:
        print(f"  - {diff.difference_type.value}: {diff.chromosome}:{diff.position}")

    # Generate feature vector
    feature_vector = differences_to_feature_vector(differences)

    print(f"\n🎯 Feature Vector:")
    print(f"  Dimension: {feature_vector.shape[0]}D (expected {TOTAL_FEATURE_DIM}D)")
    print(f"  Value range: [{feature_vector.min():.3f}, {feature_vector.max():.3f}]")
    print(f"  Mean: {feature_vector.mean():.3f}")
    print(f"  Std: {feature_vector.std():.3f}")

    # Show feature breakdown
    print(f"\n📋 Feature Breakdown:")
    offset = 0
    print(f"  [0:3]   Difference types: {feature_vector[offset:offset+3]}")
    offset += 3
    print(f"  [3:67]  Position encoding (showing first 5): {feature_vector[offset:offset+5]}...")
    offset += 64
    print(f"  [67:75] Allele composition: {feature_vector[offset:offset+8]}")
    offset += 8
    print(f"  [75:80] Genotype distribution: {feature_vector[offset:offset+5]}")
    offset += 5
    print(f"  [80:90] Functional impact: {feature_vector[offset:offset+10]}")
    offset += 10
    print(f"  [90:95] Quality metrics: {feature_vector[offset:offset+5]}")


def demo_feature_description():
    """Demonstrate feature vector description."""
    print("\n" + "=" * 70)
    print("4. Feature Vector Description and Interpretation")
    print("=" * 70)

    # Create simple differences for easy interpretation
    from genomevault.differential_encoding import VariantDifference

    differences = [
        VariantDifference(
            difference_type=DifferenceType.NEW_MUTATION,
            chromosome="chr7",
            position=117_199_563,  # CFTR common mutation
            exp_ref="CTT",
            exp_alt="C",  # Deletion (deltaF508)
            exp_genotype="0/1",
            exp_quality=99.0,
            functional_impact=FunctionalImpact.HIGH,
        ),
    ]

    feature_vector = differences_to_feature_vector(differences)
    description = describe_feature_vector(feature_vector)

    print(f"\n📖 Human-Readable Description:")
    print(f"\n  Difference Types:")
    for dtype, freq in description['difference_types'].items():
        print(f"    {dtype}: {freq:.2%}")

    print(f"\n  Position Encoding:")
    print(f"    Dimension: {description['position_encoding']['dimension']}")
    print(f"    Mean: {description['position_encoding']['mean']:.3f}")
    print(f"    Std: {description['position_encoding']['std']:.3f}")

    print(f"\n  Allele Composition:")
    print(f"    Ref: {description['allele_composition']['ref']}")
    print(f"    Alt: {description['allele_composition']['alt']}")

    print(f"\n  Genotype Distribution:")
    for geno, freq in description['genotype_distribution'].items():
        if freq > 0:
            print(f"    {geno}: {freq:.2%}")

    print(f"\n  Functional Impact:")
    print(f"    High impact: {description['functional_impact']['high_freq']:.2%}")
    print(f"    Moderate impact: {description['functional_impact']['moderate_freq']:.2%}")
    print(f"    Average score: {description['functional_impact']['avg_score']:.2f}")
    print(f"    Indel fraction: {description['functional_impact']['indel_frac']:.2%}")

    print(f"\n  Quality Metrics:")
    print(f"    Mean: {description['quality_metrics']['mean']:.2f}")
    print(f"    Range: [{description['quality_metrics']['min']:.2f}, "
          f"{description['quality_metrics']['max']:.2f}]")


def demo_feature_names():
    """Demonstrate feature naming."""
    print("\n" + "=" * 70)
    print("5. Feature Names and Indexing")
    print("=" * 70)

    names = get_feature_names()

    print(f"\n📝 Total features: {len(names)}")
    print(f"\n  First 15 features:")
    for i in range(15):
        print(f"    [{i:2d}] {names[i]}")

    print(f"\n  Last 10 features:")
    for i in range(len(names) - 10, len(names)):
        print(f"    [{i:2d}] {names[i]}")

    # Find specific features
    print(f"\n  🔍 Finding specific features:")
    for name in names:
        if 'impact_high' in name:
            idx = names.index(name)
            print(f"    High impact frequency: index {idx} ({name})")
        if 'qual_mean' in name:
            idx = names.index(name)
            print(f"    Mean quality: index {idx} ({name})")


def demo_multiple_regions():
    """Demonstrate feature vectors for multiple genomic regions."""
    print("\n" + "=" * 70)
    print("6. Batch Feature Vector Generation")
    print("=" * 70)

    regions = [
        {
            "name": "BRCA1",
            "chr": "chr17",
            "start": 43_044_295,
            "end": 43_125_483,
            "variants": 4,
        },
        {
            "name": "BRCA2",
            "chr": "chr13",
            "start": 32_315_086,
            "end": 32_400_268,
            "variants": 3,
        },
        {
            "name": "CFTR",
            "chr": "chr7",
            "start": 117_120_000,
            "end": 117_308_000,
            "variants": 5,
        },
    ]

    feature_vectors = []

    print(f"\n🧬 Generating feature vectors for {len(regions)} regions:\n")

    for region in regions:
        # Create simple mock differences
        from genomevault.differential_encoding import VariantDifference

        differences = [
            VariantDifference(
                difference_type=DifferenceType.NEW_MUTATION,
                chromosome=region["chr"],
                position=region["start"] + i * 10000,
                exp_ref="A",
                exp_alt="G",
                exp_genotype="0/1",
                exp_quality=95.0 + i,
                functional_impact=FunctionalImpact.MODERATE,
            )
            for i in range(region["variants"])
        ]

        vector = differences_to_feature_vector(differences)
        feature_vectors.append(vector)

        print(f"  {region['name']:8s} ({region['chr']:5s}:{region['start']:,}-{region['end']:,})")
        print(f"    Variants: {len(differences)}")
        print(f"    Vector shape: {vector.shape}")
        print(f"    Value range: [{vector.min():.3f}, {vector.max():.3f}]")
        print()

    # Stack into matrix
    feature_matrix = np.stack(feature_vectors)

    print(f"📊 Feature Matrix:")
    print(f"  Shape: {feature_matrix.shape} (regions × features)")
    print(f"  Memory: {feature_matrix.nbytes / 1024:.2f} KB")

    # Compute pairwise similarities
    print(f"\n🔗 Pairwise Cosine Similarities:")
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            sim = np.dot(feature_vectors[i], feature_vectors[j]) / \
                  (np.linalg.norm(feature_vectors[i]) * np.linalg.norm(feature_vectors[j]))
            print(f"  {regions[i]['name']:8s} vs {regions[j]['name']:8s}: {sim:.4f}")


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("Feature Vector Construction Demo")
    print("Section 6.1: Differential Encoding to Fixed-Dimensional Features")
    print("=" * 70)

    # Run all demos
    demo_position_encoding()
    demo_component_features()
    demo_full_feature_vector()
    demo_feature_description()
    demo_feature_names()
    demo_multiple_regions()

    print("\n" + "=" * 70)
    print("✅ All demos completed successfully!")
    print("=" * 70)
    print(f"\nKey Takeaways:")
    print(f"  • Feature vectors are {TOTAL_FEATURE_DIM}-dimensional")
    print(f"  • Position encoding captures genomic location (64D)")
    print(f"  • Multiple feature types capture variant characteristics")
    print(f"  • Vectors are suitable for ML/HD encoding")
    print(f"  • Efficient batch processing for multiple regions")
    print("=" * 70)


if __name__ == "__main__":
    main()
