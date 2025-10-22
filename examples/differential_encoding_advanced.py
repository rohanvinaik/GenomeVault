"""
Differential Encoding - Advanced Example

This example demonstrates advanced features of differential encoding including:
- Multiple analysis types
- Custom chunking strategies
- Batch processing
- Performance optimization
- Genome similarity analysis
- Query optimization

For a simpler introduction, see differential_encoding_basic.py
"""

import tempfile
import time
from pathlib import Path
from typing import List, Dict

import numpy as np

# Import differential encoding components
from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode
from genomevault.differential_encoding import (
    AnalysisType,
    Genome,
    Variant,
    EncodedGenome,
    DifferentialGenomeQuery,
    setup_default_references,
    ChunkingStrategy,
    GenomicFeature,
    DifferentialHypervectorEncoder,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def create_test_genomes() -> List[Genome]:
    """Create multiple test genomes for batch processing."""

    genomes = []

    # Genome 1: Patient with variants on chr1
    genomes.append(Genome(
        genome_id="patient_001",
        assembly="GRCh38",
        chromosomes={
            "chr1": [
                Variant(chromosome="chr1", position=100000 + i * 10000, ref="A", alt="G", genotype="0/1", quality=95.0 + i)
                for i in range(10)
            ],
        }
    ))

    # Genome 2: Patient with variants on chr2
    genomes.append(Genome(
        genome_id="patient_002",
        assembly="GRCh38",
        chromosomes={
            "chr2": [
                Variant(chromosome="chr2", position=200000 + i * 10000, ref="C", alt="T", genotype="1/1", quality=90.0 + i)
                for i in range(10)
            ],
        }
    ))

    # Genome 3: Patient with variants on both chromosomes
    genomes.append(Genome(
        genome_id="patient_003",
        assembly="GRCh38",
        chromosomes={
            "chr1": [
                Variant(chromosome="chr1", position=150000 + i * 5000, ref="G", alt="A", genotype="0/1", quality=92.0 + i)
                for i in range(5)
            ],
            "chr2": [
                Variant(chromosome="chr2", position=250000 + i * 5000, ref="T", alt="C", genotype="0/1", quality=93.0 + i)
                for i in range(5)
            ],
        }
    ))

    return genomes


def demo_multiple_analysis_types():
    """Demonstrate encoding with different analysis types."""

    print_section("DEMO 1: Multiple Analysis Types")

    # Setup
    temp_dir = Path(tempfile.mkdtemp())
    reference_dir = temp_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    manager = setup_default_references(reference_dir=reference_dir, use_case="development")
    encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL, reference_dir=reference_dir, dimension=1000, seed=42)

    # Create test genome
    genome = create_test_genomes()[0]

    print(f"Encoding {genome.genome_id} with multiple analysis types...")
    print()

    # Try different analysis types
    analysis_types = [
        AnalysisType.SLIDING_WINDOW,
        AnalysisType.GENE_REGION,
        AnalysisType.VARIANT_DENSITY,
        AnalysisType.CHROMOSOMAL,
    ]

    results = {}

    for analysis_type in analysis_types:
        print(f"Analysis type: {analysis_type.value}")

        start_time = time.time()
        encoded = encoder.encode_genome(genome=genome, analysis_type=analysis_type, bundle_chunks=True)
        encoding_time = time.time() - start_time

        results[analysis_type.value] = {
            "chunks": len(encoded.chunk_hypervectors),
            "storage_kb": encoded.storage_size_kb(),
            "encoding_time_ms": encoding_time * 1000,
        }

        print(f"  Chunks: {results[analysis_type.value]['chunks']}")
        print(f"  Storage: {results[analysis_type.value]['storage_kb']:.2f} KB")
        print(f"  Time: {results[analysis_type.value]['encoding_time_ms']:.2f} ms")
        print()

    # Comparison table
    print("Comparison Summary:")
    print(f"{'Analysis Type':<25} {'Chunks':<10} {'Storage (KB)':<15} {'Time (ms)':<12}")
    print("-" * 65)
    for analysis_type, result in results.items():
        print(f"{analysis_type:<25} {result['chunks']:<10} {result['storage_kb']:<15.2f} {result['encoding_time_ms']:<12.2f}")
    print()

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def demo_custom_chunking_strategy():
    """Demonstrate custom chunking strategies."""

    print_section("DEMO 2: Custom Chunking Strategy")

    # Setup
    temp_dir = Path(tempfile.mkdtemp())
    reference_dir = temp_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    manager = setup_default_references(reference_dir=reference_dir, use_case="development")
    encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL, reference_dir=reference_dir, dimension=1000, seed=42)

    genome = create_test_genomes()[0]

    # Define custom chunking strategy
    custom_strategy = ChunkingStrategy(
        name="custom_targeted_panel",
        window_size=50000,  # 50 kb windows
        overlap=5000,       # 5 kb overlap
        features=[
            GenomicFeature.CODING,      # Coding regions
            GenomicFeature.SPLICE_SITE, # Splice sites
            GenomicFeature.UTR_5,       # 5' UTR
            GenomicFeature.UTR_3,       # 3' UTR
        ],
        min_variants=3,
        max_chunk_size=100000,
    )

    print("Custom strategy configuration:")
    print(f"  Name: {custom_strategy.name}")
    print(f"  Window size: {custom_strategy.window_size:,} bp")
    print(f"  Overlap: {custom_strategy.overlap:,} bp")
    print(f"  Features: {[f.value for f in custom_strategy.features]}")
    print(f"  Min variants: {custom_strategy.min_variants}")
    print(f"  Max chunk size: {custom_strategy.max_chunk_size:,} bp")
    print()

    # Note: Custom strategy would be used with CUSTOM_INTERVALS analysis type
    # For this demo, we'll use GENE_REGION as a proxy
    print("Encoding with custom-like strategy (using GENE_REGION)...")
    encoded = encoder.encode_genome(genome=genome, analysis_type=AnalysisType.GENE_REGION, bundle_chunks=True)

    print(f"✅ Encoding complete!")
    print(f"   Chunks: {len(encoded.chunk_hypervectors)}")
    print(f"   Storage: {encoded.storage_size_kb():.2f} KB")
    print()

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def demo_batch_processing():
    """Demonstrate batch processing of multiple genomes."""

    print_section("DEMO 3: Batch Processing")

    # Setup
    temp_dir = Path(tempfile.mkdtemp())
    reference_dir = temp_dir / "references"
    encoded_dir = temp_dir / "encoded"
    reference_dir.mkdir(parents=True, exist_ok=True)
    encoded_dir.mkdir(parents=True, exist_ok=True)

    manager = setup_default_references(reference_dir=reference_dir, use_case="development")
    encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL, reference_dir=reference_dir, dimension=1000, seed=42)

    # Create multiple genomes
    genomes = create_test_genomes()

    print(f"Processing {len(genomes)} genomes...")
    print()

    # Batch encode
    total_start_time = time.time()
    encoded_genomes = []

    for i, genome in enumerate(genomes, 1):
        print(f"[{i}/{len(genomes)}] Encoding {genome.genome_id}...")

        start_time = time.time()
        encoded = encoder.encode_genome(genome=genome, analysis_type=AnalysisType.GENE_REGION, bundle_chunks=True)
        encoding_time = time.time() - start_time

        # Save
        save_path = encoded_dir / f"{genome.genome_id}.enc.gz"
        compressed_bytes = encoded.save(save_path, compress=True)

        encoded_genomes.append(encoded)

        print(f"  ✅ Encoded in {encoding_time * 1000:.2f} ms")
        print(f"  Chunks: {len(encoded.chunk_hypervectors)}")
        print(f"  Storage: {encoded.storage_size_kb():.2f} KB (compressed: {compressed_bytes / 1024:.2f} KB)")
        print()

    total_time = time.time() - total_start_time

    print(f"✅ Batch processing complete!")
    print(f"   Total genomes: {len(genomes)}")
    print(f"   Total time: {total_time:.2f} s")
    print(f"   Average time per genome: {total_time / len(genomes):.2f} s")
    print()

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def demo_genome_similarity():
    """Demonstrate genome similarity analysis using hypervectors."""

    print_section("DEMO 4: Genome Similarity Analysis")

    # Setup
    temp_dir = Path(tempfile.mkdtemp())
    reference_dir = temp_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    manager = setup_default_references(reference_dir=reference_dir, use_case="development")
    encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL, reference_dir=reference_dir, dimension=10000, seed=42)

    # Create test genomes
    genomes = create_test_genomes()

    print(f"Analyzing similarity between {len(genomes)} genomes...")
    print()

    # Encode all genomes
    encoded_genomes = {}
    for genome in genomes:
        encoded = encoder.encode_genome(genome=genome, analysis_type=AnalysisType.GENE_REGION, bundle_chunks=True)
        encoded_genomes[genome.genome_id] = encoded

    # Create hypervector encoder for similarity computation
    hv_encoder = encoder.differential_encoder.hypervector_encoder

    # Compute pairwise similarities
    print("Pairwise Similarity Matrix:")
    print()

    genome_ids = list(encoded_genomes.keys())
    print(f"{'':>15}", end="")
    for genome_id in genome_ids:
        print(f"{genome_id:>15}", end="")
    print()
    print("-" * (15 + 15 * len(genome_ids)))

    for i, genome_id_1 in enumerate(genome_ids):
        print(f"{genome_id_1:>15}", end="")

        hv1 = encoded_genomes[genome_id_1].bundled_hypervector

        for j, genome_id_2 in enumerate(genome_ids):
            hv2 = encoded_genomes[genome_id_2].bundled_hypervector

            # Compute cosine similarity
            similarity = hv_encoder.similarity(hv1, hv2)

            print(f"{similarity:>15.4f}", end="")

        print()
    print()

    # Find most similar pair
    max_similarity = 0.0
    most_similar_pair = None

    for i, genome_id_1 in enumerate(genome_ids):
        for j, genome_id_2 in enumerate(genome_ids):
            if i < j:  # Only upper triangle
                hv1 = encoded_genomes[genome_id_1].bundled_hypervector
                hv2 = encoded_genomes[genome_id_2].bundled_hypervector
                similarity = hv_encoder.similarity(hv1, hv2)

                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = (genome_id_1, genome_id_2)

    if most_similar_pair:
        print(f"Most similar genomes: {most_similar_pair[0]} ↔ {most_similar_pair[1]}")
        print(f"Similarity score: {max_similarity:.4f}")
    print()

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def demo_performance_optimization():
    """Demonstrate performance optimization techniques."""

    print_section("DEMO 5: Performance Optimization")

    # Setup
    temp_dir = Path(tempfile.mkdtemp())
    reference_dir = temp_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    manager = setup_default_references(reference_dir=reference_dir, use_case="development")

    genome = create_test_genomes()[0]

    print("Testing different hypervector dimensions...")
    print()

    # Test different dimensions
    dimensions = [1000, 5000, 10000, 50000]

    results = {}

    for dimension in dimensions:
        print(f"Dimension: {dimension}")

        encoder = UnifiedGenomicEncoder(
            mode=EncodingMode.DIFFERENTIAL,
            reference_dir=reference_dir,
            dimension=dimension,
            seed=42,
        )

        start_time = time.time()
        encoded = encoder.encode_genome(genome=genome, analysis_type=AnalysisType.GENE_REGION, bundle_chunks=True)
        encoding_time = time.time() - start_time

        # Test compression
        save_path = temp_dir / f"test_{dimension}.enc.gz"
        compressed_bytes = encoded.save(save_path, compress=True)

        results[dimension] = {
            "encoding_time_ms": encoding_time * 1000,
            "storage_kb": encoded.storage_size_kb(),
            "compressed_kb": compressed_bytes / 1024,
            "compression_ratio": encoded.storage_size_kb() / (compressed_bytes / 1024),
        }

        print(f"  Encoding time: {results[dimension]['encoding_time_ms']:.2f} ms")
        print(f"  Storage: {results[dimension]['storage_kb']:.2f} KB")
        print(f"  Compressed: {results[dimension]['compressed_kb']:.2f} KB")
        print(f"  Compression ratio: {results[dimension]['compression_ratio']:.1f}x")
        print()

    # Summary table
    print("Performance Summary:")
    print(f"{'Dimension':<12} {'Encoding (ms)':<15} {'Storage (KB)':<15} {'Compressed (KB)':<18} {'Ratio':<8}")
    print("-" * 75)
    for dimension, result in results.items():
        print(
            f"{dimension:<12} "
            f"{result['encoding_time_ms']:<15.2f} "
            f"{result['storage_kb']:<15.2f} "
            f"{result['compressed_kb']:<18.2f} "
            f"{result['compression_ratio']:<8.1f}x"
        )
    print()

    print("Recommendations:")
    print("  • For speed: Use dimension=1000 (fastest encoding)")
    print("  • For balance: Use dimension=10000 (production default)")
    print("  • For accuracy: Use dimension=50000 (best similarity)")
    print()

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def demo_advanced_querying():
    """Demonstrate advanced querying techniques."""

    print_section("DEMO 6: Advanced Querying")

    # Setup
    temp_dir = Path(tempfile.mkdtemp())
    reference_dir = temp_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    manager = setup_default_references(reference_dir=reference_dir, use_case="development")
    encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL, reference_dir=reference_dir, dimension=1000, seed=42)

    # Create and encode genome
    genome = create_test_genomes()[2]  # Genome with variants on both chromosomes
    encoded = encoder.encode_genome(genome=genome, analysis_type=AnalysisType.GENE_REGION, bundle_chunks=True)

    # Create query interface
    query_interface = DifferentialGenomeQuery(
        reference_manager=encoder.reference_manager,
        hv_encoder=encoder.differential_encoder.hypervector_encoder,
    )

    print(f"Querying encoded genome: {encoded.genome_id}")
    print()

    # Query 1: Specific region on chr1
    print("Query 1: Region chr1:100000-200000")
    result1 = query_interface.query_region(encoded, "chr1", 100000, 200000)
    print(f"  Variants found: {result1.variant_count}")
    print(f"  Chunks used: {result1.chunks_used}")
    print(f"  Query time: {result1.query_time_ms:.2f} ms")
    print()

    # Query 2: Entire chr2
    print("Query 2: Entire chr2")
    result2 = query_interface.query_region(encoded, "chr2", 0, 300000000)
    print(f"  Variants found: {result2.variant_count}")
    print(f"  Chunks used: {result2.chunks_used}")
    print(f"  Query time: {result2.query_time_ms:.2f} ms")
    print()

    # Query 3: Multiple regions
    print("Query 3: Multiple regions (chr1 and chr2)")
    regions = [
        ("chr1", 140000, 160000),
        ("chr2", 240000, 260000),
    ]

    total_variants = 0
    for chrom, start, end in regions:
        result = query_interface.query_region(encoded, chrom, start, end)
        total_variants += result.variant_count
        print(f"  Region {chrom}:{start}-{end}: {result.variant_count} variants")

    print(f"  Total variants: {total_variants}")
    print()

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def demo_determinism():
    """Demonstrate deterministic encoding with seeds."""

    print_section("DEMO 7: Deterministic Encoding")

    # Setup
    temp_dir = Path(tempfile.mkdtemp())
    reference_dir = temp_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    manager = setup_default_references(reference_dir=reference_dir, use_case="development")

    genome = create_test_genomes()[0]

    print("Testing determinism: encoding same genome twice with same seed...")
    print()

    # Encode with seed 42 (first time)
    encoder1 = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL, reference_dir=reference_dir, dimension=1000, seed=42)
    encoded1 = encoder1.encode_genome(genome=genome, analysis_type=AnalysisType.GENE_REGION, bundle_chunks=True)

    # Encode with seed 42 (second time)
    encoder2 = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL, reference_dir=reference_dir, dimension=1000, seed=42)
    encoded2 = encoder2.encode_genome(genome=genome, analysis_type=AnalysisType.GENE_REGION, bundle_chunks=True)

    # Encode with different seed
    encoder3 = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL, reference_dir=reference_dir, dimension=1000, seed=123)
    encoded3 = encoder3.encode_genome(genome=genome, analysis_type=AnalysisType.GENE_REGION, bundle_chunks=True)

    # Compare hypervectors
    hv1 = encoded1.bundled_hypervector
    hv2 = encoded2.bundled_hypervector
    hv3 = encoded3.bundled_hypervector

    # Check if identical (within floating point precision)
    same_seed_identical = np.allclose(hv1, hv2, rtol=1e-9)
    diff_seed_identical = np.allclose(hv1, hv3, rtol=1e-9)

    print(f"Same seed (42) produces identical encoding: {same_seed_identical}")
    print(f"Different seed (123) produces identical encoding: {diff_seed_identical}")
    print()

    if same_seed_identical:
        print("✅ DETERMINISM VERIFIED")
        print("   Same seed → same encoding (reproducible)")
    else:
        print("⚠️  Encoding not identical (possible floating point differences)")

    print()

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def main():
    """Run all advanced demos."""

    print_section("DIFFERENTIAL ENCODING - ADVANCED EXAMPLES")

    print("This demo showcases advanced features of differential encoding:")
    print("  1. Multiple analysis types")
    print("  2. Custom chunking strategies")
    print("  3. Batch processing")
    print("  4. Genome similarity analysis")
    print("  5. Performance optimization")
    print("  6. Advanced querying")
    print("  7. Deterministic encoding")
    print()

    input("Press Enter to continue...")

    # Run all demos
    demo_multiple_analysis_types()
    demo_custom_chunking_strategy()
    demo_batch_processing()
    demo_genome_similarity()
    demo_performance_optimization()
    demo_advanced_querying()
    demo_determinism()

    # Summary
    print_section("Summary")

    print("✅ All advanced demos completed successfully!")
    print()
    print("Key takeaways:")
    print("  • Different analysis types optimize for different use cases")
    print("  • Custom strategies allow fine-grained control")
    print("  • Batch processing scales efficiently")
    print("  • Hypervectors enable fast similarity queries")
    print("  • Dimension affects speed vs. accuracy trade-off")
    print("  • Advanced queries support complex genomic regions")
    print("  • Deterministic encoding ensures reproducibility")
    print()
    print("For more information:")
    print("  • docs/differential_encoding_guide.md - Complete user guide")
    print("  • docs/api_reference_differential.md - API documentation")
    print("  • examples/differential_encoding_basic.py - Simple introduction")
    print()


if __name__ == "__main__":
    main()
