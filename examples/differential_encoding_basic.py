"""
Differential Encoding - Basic Example

This example demonstrates the basic workflow for encoding genomic data using
differential encoding in GenomeVault.

Workflow:
1. Setup reference genomes
2. Create a simple genome with variants
3. Encode the genome using differential encoding
4. Save the encoded genome
5. Load and verify the encoded genome
6. Query specific regions

For production use with VCF files, see differential_encoding_advanced.py
"""

import tempfile
from pathlib import Path

# Import differential encoding components
from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode
from genomevault.differential_encoding import (
    AnalysisType,
    Genome,
    Variant,
    EncodedGenome,
    DifferentialGenomeQuery,
    setup_default_references,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def main():
    """Run the basic differential encoding example."""

    print_section("DIFFERENTIAL ENCODING - BASIC EXAMPLE")

    # =========================================================================
    # STEP 1: Setup Reference Genomes
    # =========================================================================
    print_section("Step 1: Setup Reference Genomes")

    # Create a temporary directory for this example
    # In production, use a permanent directory like ~/.genomevault/references
    temp_dir = Path(tempfile.mkdtemp())
    reference_dir = temp_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reference directory: {reference_dir}")
    print()

    # Setup development references (synthetic test data, very fast)
    print("Setting up reference genomes...")
    manager = setup_default_references(
        reference_dir=reference_dir,
        use_case="development",  # Uses synthetic test data (~0.1 MB)
    )

    print(f"✅ Loaded {manager.reference_count} reference genome(s)")
    print()

    # =========================================================================
    # STEP 2: Create Encoder
    # =========================================================================
    print_section("Step 2: Create Differential Encoder")

    # Initialize the unified encoder with differential mode
    encoder = UnifiedGenomicEncoder(
        mode=EncodingMode.DIFFERENTIAL,  # Use differential encoding
        reference_dir=reference_dir,      # Point to reference directory
        dimension=1000,                    # Hypervector dimension (1K for speed)
        seed=42,                          # Seed for reproducibility
    )

    print(f"✅ Encoder initialized")
    print(f"   Mode: {encoder.mode.value}")
    print(f"   Dimension: {encoder.dimension}")
    print(f"   References loaded: {encoder.reference_manager.reference_count}")
    print()

    # =========================================================================
    # STEP 3: Create Test Genome
    # =========================================================================
    print_section("Step 3: Create Test Genome")

    # Create a simple genome with variants on chr1
    # In production, you would load this from a VCF file
    genome = Genome(
        genome_id="patient_001",
        assembly="GRCh38",
        chromosomes={
            "chr1": [
                # Each variant specifies: chromosome, position, ref, alt, genotype, quality
                Variant(
                    chromosome="chr1",
                    position=100000,
                    ref="A",
                    alt="G",
                    genotype="0/1",  # Heterozygous
                    quality=99.0,
                ),
                Variant(
                    chromosome="chr1",
                    position=200000,
                    ref="C",
                    alt="T",
                    genotype="1/1",  # Homozygous alternate
                    quality=98.0,
                ),
                Variant(
                    chromosome="chr1",
                    position=300000,
                    ref="G",
                    alt="A",
                    genotype="0/1",
                    quality=97.0,
                ),
            ],
            "chr2": [
                Variant(
                    chromosome="chr2",
                    position=150000,
                    ref="T",
                    alt="C",
                    genotype="0/1",
                    quality=96.0,
                ),
                Variant(
                    chromosome="chr2",
                    position=250000,
                    ref="A",
                    alt="G",
                    genotype="1/1",
                    quality=95.0,
                ),
            ],
        }
    )

    total_variants = sum(len(variants) for variants in genome.chromosomes.values())
    print(f"✅ Created test genome: {genome.genome_id}")
    print(f"   Assembly: {genome.assembly}")
    print(f"   Chromosomes: {list(genome.chromosomes.keys())}")
    print(f"   Total variants: {total_variants}")
    print()

    # =========================================================================
    # STEP 4: Encode the Genome
    # =========================================================================
    print_section("Step 4: Encode Genome with Differential Encoding")

    print("Encoding genome...")
    print(f"  Analysis type: GENE_REGION")
    print(f"  Bundle chunks: Yes (creates genome-level hypervector)")
    print()

    # Encode the genome
    encoded = encoder.encode_genome(
        genome=genome,
        analysis_type=AnalysisType.GENE_REGION,  # Use gene-region chunking
        bundle_chunks=True,  # Create bundled hypervector for genome-level queries
    )

    print(f"✅ Encoding complete!")
    print(f"   Genome ID: {encoded.genome_id}")
    print(f"   Assembly: {encoded.assembly}")
    print(f"   Total chunks: {len(encoded.chunk_hypervectors)}")
    print(f"   Hypervector dimension: {len(encoded.bundled_hypervector)}")
    print(f"   Storage size: {encoded.storage_size_kb():.2f} KB")
    print()

    # Show chunk details
    print("Chunk details:")
    for i, metadata in enumerate(encoded.metadata, 1):
        print(f"  Chunk {i}:")
        print(f"    Region: {metadata.get_region_string()}")
        print(f"    Differences: {metadata.difference_counts['total']}")
        print(f"      New mutations: {metadata.difference_counts['new_mutations']}")
        print(f"      Missing variants: {metadata.difference_counts['missing_variants']}")
        print(f"      Genotype differences: {metadata.difference_counts['genotype_differences']}")
    print()

    # =========================================================================
    # STEP 5: Verify Cryptographic Integrity
    # =========================================================================
    print_section("Step 5: Verify Cryptographic Integrity")

    # Verify the encoded genome's cryptographic binding
    is_valid = encoded.verify()

    if is_valid:
        print("✅ VERIFICATION PASSED")
        print("   All chunks have valid cryptographic bindings")
        print("   Data integrity confirmed")
    else:
        print("❌ VERIFICATION FAILED")
        print("   Data may have been tampered with")
    print()

    # =========================================================================
    # STEP 6: Save Encoded Genome
    # =========================================================================
    print_section("Step 6: Save Encoded Genome")

    # Save the encoded genome to disk with compression
    save_path = temp_dir / "patient_001.enc.gz"

    print(f"Saving to: {save_path}")
    print()

    compressed_bytes = encoded.save(save_path, compress=True)

    # Calculate compression statistics
    uncompressed_kb = encoded.storage_size_kb()
    compressed_kb = compressed_bytes / 1024
    compression_ratio = uncompressed_kb / compressed_kb

    print(f"✅ Saved successfully!")
    print(f"   Uncompressed size: {uncompressed_kb:.2f} KB")
    print(f"   Compressed size: {compressed_kb:.2f} KB")
    print(f"   Compression ratio: {compression_ratio:.1f}x")
    print()

    # =========================================================================
    # STEP 7: Load Encoded Genome
    # =========================================================================
    print_section("Step 7: Load Encoded Genome")

    print(f"Loading from: {save_path}")
    print()

    # Load the encoded genome from disk
    loaded = EncodedGenome.load(save_path)

    print(f"✅ Loaded successfully!")
    print(f"   Genome ID: {loaded.genome_id}")
    print(f"   Chunks: {len(loaded.chunk_hypervectors)}")
    print(f"   Verified: {loaded.verify()}")
    print()

    # =========================================================================
    # STEP 8: Query Specific Regions
    # =========================================================================
    print_section("Step 8: Query Specific Regions")

    # Create query interface
    query_interface = DifferentialGenomeQuery(
        reference_manager=encoder.reference_manager,
        hv_encoder=encoder.differential_encoder.hypervector_encoder,
    )

    print("Querying chr1:50000-250000...")
    print()

    # Query a specific genomic region
    result = query_interface.query_region(
        encoded_genome=loaded,
        chromosome="chr1",
        start=50000,
        end=250000,
    )

    print(f"✅ Query complete!")
    print(f"   Variants found: {result.variant_count}")
    print(f"   Chunks used: {result.chunks_used}")
    print()

    if result.variants:
        print("Variants in region:")
        for variant in result.variants[:5]:  # Show first 5
            print(f"  {variant.chromosome}:{variant.position} {variant.ref}→{variant.alt}")
        if len(result.variants) > 5:
            print(f"  ... and {len(result.variants) - 5} more")
    print()

    # =========================================================================
    # STEP 9: Summary and Next Steps
    # =========================================================================
    print_section("Summary")

    print("✅ Basic differential encoding workflow complete!")
    print()
    print("What we did:")
    print("  1. ✅ Setup reference genomes (synthetic test data)")
    print("  2. ✅ Created differential encoder (1000D hypervectors)")
    print("  3. ✅ Created test genome (5 variants across 2 chromosomes)")
    print("  4. ✅ Encoded genome using GENE_REGION analysis")
    print(f"  5. ✅ Verified cryptographic integrity ({len(encoded.chunk_hypervectors)} chunks)")
    print(f"  6. ✅ Saved with compression ({compression_ratio:.1f}x ratio)")
    print("  7. ✅ Loaded and verified from disk")
    print("  8. ✅ Queried specific genomic region")
    print()

    print("Next steps:")
    print("  • Try differential_encoding_advanced.py for:")
    print("    - Loading VCF files")
    print("    - Multiple analysis types")
    print("    - Batch processing")
    print("    - Performance optimization")
    print()
    print("  • Read the documentation:")
    print("    - docs/differential_encoding_guide.md")
    print("    - docs/api_reference_differential.md")
    print()
    print("  • Setup production references:")
    print("    - python scripts/genomevault_setup_references.py --use-case production")
    print()

    # Cleanup temporary directory
    import shutil
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
