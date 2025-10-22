"""
Demonstration of Cryptographic Chunking Strategies.

This script demonstrates the analysis-type-specific genomic chunking with:
1. Multiple analysis types (single SNP, gene region, sliding window, etc.)
2. Configurable chunking strategies
3. Cryptographically secure random boundary generation
4. Feature-aware chunking
5. Deterministic chunking for reproducibility

Run this script to see all chunking features in action.
"""

from genomevault.differential_encoding import (
    # Core components
    CryptoRNG,
    Variant,
    GenomeSection,
    # Chunking
    AnalysisType,
    ChunkingStrategy,
    STRATEGY_CONFIGS,
    GenomicFeature,
    GenomeChunk,
    CryptographicChunker,
    get_strategy_for_analysis,
)


def demo_analysis_types():
    """Demonstrate all analysis types."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 1: Analysis Types")
    print("=" * 70)

    print("\nAvailable analysis types:")
    for i, analysis_type in enumerate(AnalysisType, 1):
        print(f"  {i}. {analysis_type.value}")

    print("\n✅ All 7 analysis types defined")


def demo_strategy_configs():
    """Demonstrate pre-configured strategies."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 2: Pre-configured Chunking Strategies")
    print("=" * 70)

    for analysis_type in AnalysisType:
        strategy = STRATEGY_CONFIGS[analysis_type]
        print(f"\n{analysis_type.value}:")
        print(f"  Chunk size: {strategy.chunk_size:,} bp" if strategy.chunk_size else "  Chunk size: Dynamic")
        print(f"  Overlap: {strategy.overlap:,} bp")
        print(f"  Min variants: {strategy.min_variants}")
        print(f"  Max variants: {strategy.max_variants:,}")
        print(f"  Randomize boundaries: {strategy.randomize_boundaries}")
        print(f"  Respect features: {strategy.respect_features}")

    print("\n✅ All strategies configured")


def demo_basic_chunking():
    """Demonstrate basic sliding window chunking."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 3: Basic Sliding Window Chunking")
    print("=" * 70)

    # Create RNG
    rng = CryptoRNG(master_seed=b"\x00" * 32)

    # Create test variants
    variants = []
    for i in range(100):
        variants.append(
            Variant(
                chromosome="chr1",
                position=100000 + (i * 5000),  # Every 5kb
                ref="A",
                alt="G"
            )
        )

    # Create genome section
    section = GenomeSection(
        chromosome="chr1",
        start_position=100000,
        end_position=600000,
        variants=variants
    )

    print(f"\nCreated genome section:")
    print(f"  Location: {section}")
    print(f"  Variants: {section.variant_count:,}")

    # Get strategy
    strategy = get_strategy_for_analysis(AnalysisType.SLIDING_WINDOW)
    print(f"\nUsing SLIDING_WINDOW strategy:")
    print(f"  Chunk size: {strategy.chunk_size:,} bp")
    print(f"  Overlap: {strategy.overlap:,} bp")

    # Create chunker
    chunker = CryptographicChunker(strategy, rng)

    # Chunk
    master_seed = rng.derive_seed(b"experiment_1")
    chunks = chunker.chunk_genome_section(section, master_seed)

    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks[:5], 1):  # Show first 5
        print(f"  Chunk {i}:")
        print(f"    Range: {chunk.chromosome}:{chunk.start_position:,}-{chunk.end_position:,}")
        print(f"    Length: {chunk.length:,} bp")
        print(f"    Variants: {chunk.variant_count}")
        print(f"    Chunk ID: {chunk.chunk_id.hex()[:16]}...")

    if len(chunks) > 5:
        print(f"  ... and {len(chunks) - 5} more chunks")

    print("\n✅ Basic chunking complete")


def demo_feature_based_chunking():
    """Demonstrate feature-based chunking."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 4: Feature-Based Chunking")
    print("=" * 70)

    # Create RNG
    rng = CryptoRNG()

    # Create test variants
    variants = []
    for i in range(200):
        variants.append(
            Variant(
                chromosome="chr13",
                position=32000000 + (i * 10000),  # Every 10kb
                ref="A",
                alt="G"
            )
        )

    # Create genome section
    section = GenomeSection(
        chromosome="chr13",
        start_position=32000000,
        end_position=34000000,
        variants=variants
    )

    # Create genomic features
    features = [
        GenomicFeature(
            feature_id="ENSG00000139618",
            feature_type="gene",
            chromosome="chr13",
            start=32889617,
            end=32973809,
            name="BRCA2",
            strand="+"
        ),
        GenomicFeature(
            feature_id="ENSG00000185811",
            feature_type="gene",
            chromosome="chr13",
            start=33200000,
            end=33250000,
            name="EXAMPLE_GENE",
            strand="-"
        ),
    ]

    print("\nGenomic features:")
    for feature in features:
        print(f"  {feature.name} ({feature.feature_id})")
        print(f"    Location: {feature.chromosome}:{feature.start:,}-{feature.end:,}")
        print(f"    Length: {feature.length:,} bp")

    # Get GENE_REGION strategy
    strategy = get_strategy_for_analysis(AnalysisType.GENE_REGION)
    print(f"\nUsing GENE_REGION strategy (feature-aware)")

    # Create chunker
    chunker = CryptographicChunker(strategy, rng)

    # Chunk with features
    master_seed = rng.derive_seed(b"gene_analysis")
    chunks = chunker.chunk_genome_section(section, master_seed, features=features)

    print(f"\nCreated {len(chunks)} feature-based chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i} ({chunk.feature_name}):")
        print(f"    Range: {chunk.chromosome}:{chunk.start_position:,}-{chunk.end_position:,}")
        print(f"    Length: {chunk.length:,} bp")
        print(f"    Variants: {chunk.variant_count}")
        print(f"    Feature ID: {chunk.feature_id}")

    print("\n✅ Feature-based chunking complete")


def demo_determinism():
    """Demonstrate deterministic chunking."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 5: Deterministic Chunking")
    print("=" * 70)

    print("\nDemonstrating that same seed produces identical chunks...")

    # Create two separate RNG instances with same seed
    rng1 = CryptoRNG(master_seed=b"\x00" * 32)
    rng2 = CryptoRNG(master_seed=b"\x00" * 32)

    # Create identical sections
    variants = [
        Variant(chromosome="chr7", position=100000 + (i * 1000), ref="A", alt="G")
        for i in range(50)
    ]

    section = GenomeSection(
        chromosome="chr7",
        start_position=100000,
        end_position=150000,
        variants=variants
    )

    # Get strategy
    strategy = get_strategy_for_analysis(AnalysisType.SINGLE_SNP_QUERY)

    # Create two chunkers
    chunker1 = CryptographicChunker(strategy, rng1)
    chunker2 = CryptographicChunker(strategy, rng2)

    # Derive same master seed from both RNGs
    master_seed1 = rng1.derive_seed(b"test")
    master_seed2 = rng2.derive_seed(b"test")

    # Chunk with both
    chunks1 = chunker1.chunk_genome_section(section, master_seed1)
    chunks2 = chunker2.chunk_genome_section(section, master_seed2)

    print(f"\nRun 1: Created {len(chunks1)} chunks")
    print(f"Run 2: Created {len(chunks2)} chunks")

    # Verify identical
    assert len(chunks1) == len(chunks2), "Different number of chunks!"

    all_match = True
    for i, (c1, c2) in enumerate(zip(chunks1, chunks2)):
        if c1.chunk_id != c2.chunk_id:
            all_match = False
            print(f"  Chunk {i}: MISMATCH")
        else:
            print(f"  Chunk {i}: ✓ Identical (ID: {c1.chunk_id.hex()[:16]}...)")

    if all_match:
        print("\n✅ Perfect determinism - all chunks identical!")
    else:
        print("\n❌ Determinism failed")


def demo_analysis_comparison():
    """Compare different analysis types on same data."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 6: Comparison of Analysis Types")
    print("=" * 70)

    # Create test section
    variants = []
    for i in range(500):
        variants.append(
            Variant(
                chromosome="chr1",
                position=1000000 + (i * 2000),  # Every 2kb
                ref="A",
                alt="G"
            )
        )

    section = GenomeSection(
        chromosome="chr1",
        start_position=1000000,
        end_position=2000000,
        variants=variants
    )

    print(f"\nTest section: {section}")
    print(f"Total variants: {section.variant_count:,}")

    rng = CryptoRNG()
    master_seed = rng.derive_seed(b"comparison")

    analysis_types = [
        AnalysisType.SINGLE_SNP_QUERY,
        AnalysisType.SLIDING_WINDOW,
        AnalysisType.STRUCTURAL_VARIANT,
        AnalysisType.WHOLE_CHROMOSOME,
    ]

    print("\nChunking with different strategies:")
    print(f"{'Strategy':<25} {'Chunks':<10} {'Avg Length':<15} {'Avg Variants':<15}")
    print("-" * 70)

    for analysis_type in analysis_types:
        strategy = get_strategy_for_analysis(analysis_type)
        chunker = CryptographicChunker(strategy, rng)
        chunks = chunker.chunk_genome_section(section, master_seed)

        if chunks:
            avg_length = sum(c.length for c in chunks) / len(chunks)
            avg_variants = sum(c.variant_count for c in chunks) / len(chunks)

            print(f"{analysis_type.value:<25} {len(chunks):<10} {avg_length:<15,.0f} {avg_variants:<15,.1f}")
        else:
            print(f"{analysis_type.value:<25} {'0':<10} {'N/A':<15} {'N/A':<15}")

    print("\n✅ Comparison complete")


def demo_variant_constraints():
    """Demonstrate variant count constraints."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 7: Variant Count Constraints")
    print("=" * 70)

    # Create section with variable variant density
    variants = []
    # Dense region
    for i in range(100):
        variants.append(
            Variant(
                chromosome="chr1",
                position=100000 + (i * 100),  # Every 100bp (dense)
                ref="A",
                alt="G"
            )
        )
    # Sparse region
    for i in range(50):
        variants.append(
            Variant(
                chromosome="chr1",
                position=120000 + (i * 5000),  # Every 5kb (sparse)
                ref="A",
                alt="G"
            )
        )

    section = GenomeSection(
        chromosome="chr1",
        start_position=100000,
        end_position=400000,
        variants=sorted(variants, key=lambda v: v.position)
    )

    print(f"\nTest section with variable density:")
    print(f"  Total variants: {section.variant_count}")

    # Use sliding window with constraints
    strategy = get_strategy_for_analysis(AnalysisType.SLIDING_WINDOW)
    print(f"\nStrategy constraints:")
    print(f"  Min variants per chunk: {strategy.min_variants}")
    print(f"  Max variants per chunk: {strategy.max_variants}")

    rng = CryptoRNG()
    chunker = CryptographicChunker(strategy, rng)
    master_seed = rng.derive_seed(b"constraints")
    chunks = chunker.chunk_genome_section(section, master_seed)

    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks[:10], 1):
        status = "✓" if strategy.min_variants <= chunk.variant_count <= strategy.max_variants else "⚠"
        print(f"  {status} Chunk {i}: {chunk.variant_count:,} variants")

    if len(chunks) > 10:
        print(f"  ... and {len(chunks) - 10} more")

    # Verify all chunks meet constraints (except possibly last chunk)
    violations = sum(
        1 for i, c in enumerate(chunks)
        if c.variant_count > strategy.max_variants
    )

    if violations == 0:
        print("\n✅ All chunks respect max_variants constraint")
    else:
        print(f"\n⚠ {violations} chunks exceed max_variants")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("CRYPTOGRAPHIC CHUNKING DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows all features of the chunking system:")
    print("1. Analysis types")
    print("2. Pre-configured strategies")
    print("3. Basic sliding window chunking")
    print("4. Feature-based chunking")
    print("5. Deterministic chunking")
    print("6. Analysis type comparison")
    print("7. Variant count constraints")

    try:
        demo_analysis_types()
        demo_strategy_configs()
        demo_basic_chunking()
        demo_feature_based_chunking()
        demo_determinism()
        demo_analysis_comparison()
        demo_variant_constraints()

        print("\n" + "=" * 70)
        print("✅ All demonstrations completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
