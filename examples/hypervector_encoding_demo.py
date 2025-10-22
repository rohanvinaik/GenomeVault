#!/usr/bin/env python3
"""
Hypervector Encoding Demo

Demonstrates the hyperdimensional encoding functionality from Section 6.2
of the specification. Shows how to:
1. Initialize the DifferentialHypervectorEncoder
2. Encode variant differences into high-dimensional hypervectors
3. Use binding and bundling operations
4. Compute similarity between encoded chunks
5. Batch encoding for multiple genomic regions
6. Integrate with the complete differential encoding pipeline
"""

import numpy as np
from genomevault.differential_encoding import (
    Variant,
    GenomeSection,
    Genome,
    compute_variant_differences,
    DifferentialHypervectorEncoder,
    create_metadata_from_chunk,
    CryptoRNG,
    AnalysisType,
    DifferenceType,
    FunctionalImpact,
)


def demo_initialization():
    """Demonstrate encoder initialization."""
    print("=" * 70)
    print("1. Encoder Initialization")
    print("=" * 70)

    # Default initialization (10,000D)
    encoder = DifferentialHypervectorEncoder()
    print(f"\n📐 Default Encoder:")
    print(f"  Dimension: {encoder.dimension}")
    print(f"  Feature dimension: {encoder.feature_dim}")
    print(f"  Projection matrix shape: {encoder.projection_matrix.shape}")

    # Custom dimension
    encoder_5k = DifferentialHypervectorEncoder(dimension=5000, seed=42)
    print(f"\n📐 Custom Encoder (5K):")
    print(f"  Dimension: {encoder_5k.dimension}")
    print(f"  Projection matrix shape: {encoder_5k.projection_matrix.shape}")

    # Reproducibility with seed
    enc1 = DifferentialHypervectorEncoder(seed=42)
    enc2 = DifferentialHypervectorEncoder(seed=42)
    print(f"\n🔒 Seed Reproducibility:")
    print(f"  Same seed produces identical encoders: {np.allclose(enc1.projection_matrix, enc2.projection_matrix)}")

    # Base vectors
    print(f"\n🎯 Base Vectors:")
    for name, vec in encoder.base_vectors.items():
        print(f"  {name}: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")


def demo_basic_encoding():
    """Demonstrate basic hypervector encoding."""
    print("\n" + "=" * 70)
    print("2. Basic Hypervector Encoding")
    print("=" * 70)

    # Create simple genome sections
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
        ],
    )

    ref_section = GenomeSection(
        chromosome="chr17",
        start_position=43_044_295,
        end_position=43_125_483,
        variants=[
            Variant(chromosome="chr17", position=43_052_000, ref="C", alt="G",
                   genotype="0/1", quality=95.0),
        ],
    )

    # Compute differences
    differences = compute_variant_differences(exp_section, ref_section)

    print(f"\n🧬 Computed {len(differences)} differences:")
    for diff in differences:
        print(f"  - {diff.difference_type.value}: {diff.chromosome}:{diff.position}")

    # Encode to hypervector
    encoder = DifferentialHypervectorEncoder(dimension=10000, seed=42)
    hypervector = encoder.encode_difference_vector(differences)

    print(f"\n🎯 Encoded Hypervector:")
    print(f"  Shape: {hypervector.shape}")
    print(f"  Norm: {np.linalg.norm(hypervector):.4f} (should be 1.0)")
    print(f"  Value range: [{hypervector.min():.4f}, {hypervector.max():.4f}]")
    print(f"  Mean: {hypervector.mean():.4f}")
    print(f"  Std: {hypervector.std():.4f}")
    print(f"  Non-zero elements: {np.count_nonzero(hypervector)}/{len(hypervector)}")


def demo_with_metadata():
    """Demonstrate encoding with metadata."""
    print("\n" + "=" * 70)
    print("3. Encoding with Metadata")
    print("=" * 70)

    # Create genome sections
    exp_section = GenomeSection(
        chromosome="chr7",
        start_position=117_120_000,  # CFTR region
        end_position=117_308_000,
        variants=[
            Variant(chromosome="chr7", position=117_199_563, ref="CTT", alt="C",
                   genotype="0/1", quality=99.0,
                   info={"Gene": "CFTR", "Consequence": "frameshift_variant"}),
        ],
    )

    ref_section = GenomeSection(
        chromosome="chr7",
        start_position=117_120_000,
        end_position=117_308_000,
        variants=[],
    )

    # Compute differences
    differences = compute_variant_differences(exp_section, ref_section)

    # Create metadata
    rng = CryptoRNG()
    metadata = create_metadata_from_chunk(
        chunk_id=rng.derive_seed(b'cftr_chunk_001'),
        chromosome='chr7',
        start_position=117_120_000,
        end_position=117_308_000,
        reference_genome_id='GRCh38_001',
        reference_seed=rng.derive_seed(b'ref_seed'),
        reference_hash=rng.derive_seed(b'ref_hash'),
        chunking_strategy='sliding_window',
        chunking_seed=rng.derive_seed(b'chunk_seed'),
        analysis_type=AnalysisType.SLIDING_WINDOW,
        new_mutations=sum(1 for d in differences if d.is_new_mutation),
        missing_variants=sum(1 for d in differences if d.is_missing),
        genotype_differences=sum(1 for d in differences if d.is_genotype_diff),
        chunk_data=b'test_chunk_data',
        reference_data=b'test_reference_data',
    )

    print(f"\n📋 Metadata:")
    print(f"  Chunk ID: {metadata.chunk_id.hex()[:16]}...")
    print(f"  Reference: {metadata.reference_genome_id}")
    print(f"  Region: {metadata.get_region_string()}")
    print(f"  Differences: {metadata.difference_counts['total']}")

    # Encode with metadata
    encoder = DifferentialHypervectorEncoder(dimension=10000, seed=42)
    hypervector_with_metadata = encoder.encode_difference_vector(differences, metadata)
    hypervector_without_metadata = encoder.encode_difference_vector(differences)

    print(f"\n🎯 Encoded Hypervectors:")
    print(f"  With metadata: shape={hypervector_with_metadata.shape}, norm={np.linalg.norm(hypervector_with_metadata):.4f}")
    print(f"  Without metadata: shape={hypervector_without_metadata.shape}, norm={np.linalg.norm(hypervector_without_metadata):.4f}")

    # Compare similarity
    similarity = encoder.similarity(hypervector_with_metadata, hypervector_without_metadata)
    print(f"\n🔍 Similarity (with vs without metadata): {similarity:.4f}")
    print(f"  Metadata binding changes the representation!")


def demo_similarity_computation():
    """Demonstrate similarity computation between hypervectors."""
    print("\n" + "=" * 70)
    print("4. Similarity Computation")
    print("=" * 70)

    encoder = DifferentialHypervectorEncoder(dimension=10000, seed=42)

    # Create similar variants (same chromosome, nearby positions)
    from genomevault.differential_encoding import VariantDifference

    similar_diffs_1 = [
        VariantDifference(
            difference_type=DifferenceType.NEW_MUTATION,
            chromosome="chr1",
            position=100000,
            exp_ref="A",
            exp_alt="G",
            exp_genotype="0/1",
            functional_impact=FunctionalImpact.HIGH,
        ),
        VariantDifference(
            difference_type=DifferenceType.NEW_MUTATION,
            chromosome="chr1",
            position=100100,
            exp_ref="C",
            exp_alt="T",
            exp_genotype="0/1",
            functional_impact=FunctionalImpact.HIGH,
        ),
    ]

    similar_diffs_2 = [
        VariantDifference(
            difference_type=DifferenceType.NEW_MUTATION,
            chromosome="chr1",
            position=100050,
            exp_ref="A",
            exp_alt="G",
            exp_genotype="0/1",
            functional_impact=FunctionalImpact.MODERATE,
        ),
        VariantDifference(
            difference_type=DifferenceType.NEW_MUTATION,
            chromosome="chr1",
            position=100150,
            exp_ref="C",
            exp_alt="T",
            exp_genotype="0/1",
            functional_impact=FunctionalImpact.MODERATE,
        ),
    ]

    # Different variants (different chromosome)
    different_diffs = [
        VariantDifference(
            difference_type=DifferenceType.NEW_MUTATION,
            chromosome="chr2",
            position=100000,
            exp_ref="A",
            exp_alt="G",
            exp_genotype="0/1",
            functional_impact=FunctionalImpact.HIGH,
        ),
    ]

    # Encode all
    hv1 = encoder.encode_difference_vector(similar_diffs_1)
    hv2 = encoder.encode_difference_vector(similar_diffs_2)
    hv3 = encoder.encode_difference_vector(different_diffs)

    print(f"\n🔍 Similarity Comparisons:")
    print(f"  Self-similarity: {encoder.similarity(hv1, hv1):.4f}")
    print(f"  Similar variants (chr1, nearby): {encoder.similarity(hv1, hv2):.4f}")
    print(f"  Different chromosome: {encoder.similarity(hv1, hv3):.4f}")

    print(f"\n💡 Insight:")
    print(f"  Similar genomic differences produce similar hypervectors!")
    print(f"  Different chromosomes produce more dissimilar hypervectors.")


def demo_batch_encoding():
    """Demonstrate batch encoding for multiple regions."""
    print("\n" + "=" * 70)
    print("5. Batch Encoding")
    print("=" * 70)

    encoder = DifferentialHypervectorEncoder(dimension=10000, seed=42)

    # Create multiple genomic regions
    regions = [
        {"name": "BRCA1", "chr": "chr17", "start": 43_044_295, "end": 43_125_483, "variants": 5},
        {"name": "BRCA2", "chr": "chr13", "start": 32_315_086, "end": 32_400_268, "variants": 3},
        {"name": "CFTR", "chr": "chr7", "start": 117_120_000, "end": 117_308_000, "variants": 4},
        {"name": "TP53", "chr": "chr17", "start": 7_571_720, "end": 7_590_868, "variants": 6},
    ]

    # Create mock differences for each region
    from genomevault.differential_encoding import VariantDifference

    differences_batch = []
    for region in regions:
        diffs = [
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
        differences_batch.append(diffs)

    print(f"\n🧬 Encoding {len(regions)} genomic regions:")
    for i, region in enumerate(regions):
        print(f"  {i+1}. {region['name']:8s} ({region['chr']:5s}:{region['start']:,}-{region['end']:,}) - {region['variants']} variants")

    # Batch encode
    hypervector_matrix = encoder.encode_batch(differences_batch)

    print(f"\n📊 Batch Encoding Results:")
    print(f"  Hypervector matrix shape: {hypervector_matrix.shape} (regions × dimension)")
    print(f"  Memory usage: {hypervector_matrix.nbytes / 1024 / 1024:.2f} MB")

    # Check normalization
    norms = np.linalg.norm(hypervector_matrix, axis=1)
    print(f"  All vectors normalized: {np.allclose(norms, 1.0)}")
    print(f"  Norm range: [{norms.min():.4f}, {norms.max():.4f}]")

    # Compute pairwise similarities
    print(f"\n🔗 Pairwise Similarities:")
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            sim = encoder.similarity(hypervector_matrix[i], hypervector_matrix[j])
            print(f"  {regions[i]['name']:8s} vs {regions[j]['name']:8s}: {sim:.4f}")

    # BRCA1 and TP53 are on the same chromosome (chr17)
    brca1_idx = 0
    tp53_idx = 3
    same_chr_sim = encoder.similarity(hypervector_matrix[brca1_idx], hypervector_matrix[tp53_idx])
    print(f"\n💡 Same chromosome (chr17): BRCA1 vs TP53 = {same_chr_sim:.4f}")


def demo_binding_and_bundling():
    """Demonstrate binding and bundling operations."""
    print("\n" + "=" * 70)
    print("6. Binding and Bundling Operations")
    print("=" * 70)

    encoder = DifferentialHypervectorEncoder(dimension=1000, seed=42)

    # Create random hypervectors
    hv_a = np.random.randn(1000).astype(np.float32)
    hv_b = np.random.randn(1000).astype(np.float32)
    hv_c = np.random.randn(1000).astype(np.float32)

    # Normalize
    hv_a /= np.linalg.norm(hv_a)
    hv_b /= np.linalg.norm(hv_b)
    hv_c /= np.linalg.norm(hv_c)

    print(f"\n⚡ Binding Operation (circular convolution):")
    bound_ab = encoder._bind(hv_a, hv_b)
    print(f"  bind(A, B) shape: {bound_ab.shape}")
    print(f"  Similarity to A: {encoder.similarity(bound_ab, hv_a):.4f} (should be low)")
    print(f"  Similarity to B: {encoder.similarity(bound_ab, hv_b):.4f} (should be low)")
    print(f"  Binding creates a new, dissimilar vector!")

    print(f"\n➕ Bundling Operation (superposition):")
    bundled = encoder._bundle([hv_a, hv_b, hv_c])
    print(f"  bundle(A, B, C) shape: {bundled.shape}")
    print(f"  Norm: {np.linalg.norm(bundled):.4f} (normalized to 1.0)")
    print(f"  Similarity to A: {encoder.similarity(bundled, hv_a):.4f}")
    print(f"  Similarity to B: {encoder.similarity(bundled, hv_b):.4f}")
    print(f"  Similarity to C: {encoder.similarity(bundled, hv_c):.4f}")
    print(f"  Bundling preserves similarity to components!")

    # Weighted bundling
    weighted_bundle = encoder._bundle([hv_a, hv_b, hv_c], weights=[0.5, 0.3, 0.2])
    print(f"\n⚖️  Weighted Bundling:")
    print(f"  Weights: [0.5, 0.3, 0.2]")
    print(f"  Similarity to A (weight 0.5): {encoder.similarity(weighted_bundle, hv_a):.4f}")
    print(f"  Similarity to B (weight 0.3): {encoder.similarity(weighted_bundle, hv_b):.4f}")
    print(f"  Similarity to C (weight 0.2): {encoder.similarity(weighted_bundle, hv_c):.4f}")
    print(f"  Higher weight = higher similarity!")


def demo_end_to_end():
    """Demonstrate complete end-to-end workflow."""
    print("\n" + "=" * 70)
    print("7. Complete End-to-End Workflow")
    print("=" * 70)

    # Create genome
    genome = Genome(
        genome_id="patient_001",
        assembly="GRCh38",
        chromosomes={
            "chr17": [
                Variant(chromosome="chr17", position=43_050_000, ref="A", alt="G",
                       genotype="0/1", quality=99.0,
                       info={"Gene": "BRCA1", "Consequence": "missense_variant"}),
                Variant(chromosome="chr17", position=43_055_000, ref="C", alt="T",
                       genotype="1/1", quality=98.0,
                       info={"Gene": "BRCA1", "IMPACT": "HIGH"}),
                Variant(chromosome="chr17", position=7_577_548, ref="G", alt="A",
                       genotype="0/1", quality=97.0,
                       info={"Gene": "TP53", "Consequence": "missense_variant"}),
            ]
        }
    )

    print(f"\n🧬 Genome: {genome.genome_id}")
    print(f"  Assembly: {genome.assembly}")
    print(f"  Chromosomes: {list(genome.chromosomes.keys())}")
    print(f"  Total variants: {sum(len(v) for v in genome.chromosomes.values())}")

    # Create reference section
    ref_section = GenomeSection(
        chromosome="chr17",
        start_position=43_044_295,
        end_position=43_125_483,
        variants=[
            Variant(chromosome="chr17", position=43_052_000, ref="C", alt="G",
                   genotype="0/1", quality=95.0),
        ],
    )

    # Get experimental section
    exp_section = genome.get_chromosome_section("chr17", 43_044_295, 43_125_483)

    # Compute differences
    differences = compute_variant_differences(exp_section, ref_section)

    print(f"\n🔬 Variant Differences:")
    print(f"  New mutations: {sum(1 for d in differences if d.is_new_mutation)}")
    print(f"  Missing variants: {sum(1 for d in differences if d.is_missing)}")
    print(f"  Genotype differences: {sum(1 for d in differences if d.is_genotype_diff)}")

    # Create metadata
    rng = CryptoRNG()
    metadata = create_metadata_from_chunk(
        chunk_id=rng.derive_seed(b'brca1_chunk'),
        chromosome='chr17',
        start_position=43_044_295,
        end_position=43_125_483,
        reference_genome_id='GRCh38_brca1_ref',
        reference_seed=rng.derive_seed(b'ref_seed'),
        reference_hash=rng.derive_seed(b'ref_hash'),
        chunking_strategy='gene_region',
        chunking_seed=rng.derive_seed(b'chunk_seed'),
        analysis_type=AnalysisType.GENE_REGION,
        new_mutations=sum(1 for d in differences if d.is_new_mutation),
        missing_variants=sum(1 for d in differences if d.is_missing),
        genotype_differences=sum(1 for d in differences if d.is_genotype_diff),
        chunk_data=b'brca1_chunk_data',
        reference_data=b'brca1_reference_data',
    )

    # Encode to hypervector
    encoder = DifferentialHypervectorEncoder(dimension=10000, seed=42)
    hypervector = encoder.encode_difference_vector(differences, metadata)

    print(f"\n🎯 Final Hypervector:")
    print(f"  Shape: {hypervector.shape}")
    print(f"  Norm: {np.linalg.norm(hypervector):.4f}")
    print(f"  Value range: [{hypervector.min():.4f}, {hypervector.max():.4f}]")

    # Verify metadata binding
    print(f"\n🔐 Cryptographic Binding:")
    binding_valid = metadata.verify_binding(b'brca1_chunk_data', b'brca1_reference_data')
    print(f"  Metadata binding verified: {binding_valid}")

    print(f"\n✅ Complete pipeline:")
    print(f"  1. Genome variants → Differences")
    print(f"  2. Differences → 95D feature vector")
    print(f"  3. Feature vector → 10,000D hypervector")
    print(f"  4. Hypervector + Metadata → Final encoding")
    print(f"  5. Cryptographic binding verification")


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("Hypervector Encoding Demo")
    print("Section 6.2: Differential Encoding to Hyperdimensional Space")
    print("=" * 70)

    # Run all demos
    demo_initialization()
    demo_basic_encoding()
    demo_with_metadata()
    demo_similarity_computation()
    demo_batch_encoding()
    demo_binding_and_bundling()
    demo_end_to_end()

    print("\n" + "=" * 70)
    print("✅ All demos completed successfully!")
    print("=" * 70)
    print(f"\nKey Takeaways:")
    print(f"  • Hypervectors are high-dimensional (default 10,000D)")
    print(f"  • Random projection: 95D feature → 10,000D hypervector")
    print(f"  • Binding creates new representations (circular convolution)")
    print(f"  • Bundling combines information (superposition)")
    print(f"  • Similarity preserved for similar genomic differences")
    print(f"  • Efficient batch processing for multiple regions")
    print(f"  • Full integration with differential encoding pipeline")
    print("=" * 70)


if __name__ == "__main__":
    main()
