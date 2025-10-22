"""
Reference Genome Setup Demo

Demonstrates how to download, validate, and manage reference genomes
for differential encoding in GenomeVault.

This script shows:
1. Downloading reference genomes
2. Validating reference pools
3. Setting up default references for different use cases
4. Getting reference information
5. Using references with differential encoding
"""

import tempfile
from pathlib import Path

from genomevault.differential_encoding import (
    download_reference_genomes,
    validate_reference_pool,
    setup_default_references,
    get_reference_info,
    SecureReferenceGenomeManager,
    STANDARD_REFERENCES,
    RECOMMENDED_POOLS,
    Genome,
    Variant,
    AnalysisType,
)
from genomevault.hypervector_transform import UnifiedGenomicEncoder, EncodingMode


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print()


def progress_callback(name: str, current: int, total: int):
    """Display progress for downloads."""
    if total > 0:
        percent = (current / total) * 100
        print(f"  {name}: {percent:.0f}% ({current}/{total})")


def demo_download_references():
    """Demo: Downloading reference genomes."""
    print_section("1. DOWNLOADING REFERENCE GENOMES")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_dir = Path(tmpdir)

        print("Downloading synthetic test reference...")
        print()

        # Download references
        references = download_reference_genomes(
            sources=["synthetic_test"],
            output_dir=ref_dir,
            progress_callback=progress_callback,
        )

        print()
        print(f"✅ Downloaded {len(references)} reference(s)")
        print()

        for ref_id, ref_genome in references.items():
            variant_count = sum(len(v) for v in ref_genome.variants.values())
            print(f"  📚 {ref_id}:")
            print(f"     Assembly: {ref_genome.assembly}")
            print(f"     Variants: {variant_count:,}")
            print(f"     Chromosomes: {list(ref_genome.variants.keys())}")
            print(f"     Hash: {ref_genome.cryptographic_hash[:16]}...")
            print()


def demo_validate_references():
    """Demo: Validating reference pools."""
    print_section("2. VALIDATING REFERENCE POOLS")

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_dir = Path(tmpdir)

        # Setup references
        manager = setup_default_references(
            reference_dir=ref_dir,
            use_case="development",
        )

        print(f"Validating {manager.reference_count} reference(s)...")
        print()

        # Validate
        result = validate_reference_pool(manager)

        # Print results
        status_emoji = "✅" if result.is_valid else "❌"
        print(f"{status_emoji} Validation Status: {'VALID' if result.is_valid else 'INVALID'}")
        print(f"   References checked: {result.reference_count}")
        print(f"   Errors: {len(result.errors)}")
        print(f"   Warnings: {len(result.warnings)}")
        print()

        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  ❌ {error}")
            print()

        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
            print()

        # Per-reference details
        print("Reference Details:")
        for ref_id, status in result.reference_status.items():
            hash_emoji = "✅" if status.get('hash_valid', False) else "❌"
            print(f"  {hash_emoji} {ref_id}")
            print(f"     Assembly: {status.get('assembly', 'Unknown')}")
            print(f"     Variants: {status.get('variant_count', 0):,}")
            print(f"     Chromosomes: {status.get('chromosome_count', 0)}")
            print()


def demo_setup_default_references():
    """Demo: Setting up default references for different use cases."""
    print_section("3. SETTING UP DEFAULT REFERENCES")

    print("Available use cases:")
    for use_case, sources in RECOMMENDED_POOLS.items():
        print(f"  • {use_case}: {', '.join(sources)}")
    print()

    # Demo development setup
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_dir = Path(tmpdir)

        print("Setting up 'development' references...")
        print()

        manager = setup_default_references(
            reference_dir=ref_dir,
            use_case="development",
            progress_callback=progress_callback,
        )

        print()
        print(f"✅ Setup complete!")
        print(f"   References loaded: {manager.reference_count}")
        print()


def demo_get_reference_info():
    """Demo: Getting reference information."""
    print_section("4. GETTING REFERENCE INFORMATION")

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_dir = Path(tmpdir)

        # Setup references
        manager = setup_default_references(
            reference_dir=ref_dir,
            use_case="development",
        )

        print("Retrieving reference information...")
        print()

        # Get info
        info = get_reference_info(ref_dir)

        print(f"Total references: {info['reference_count']}")
        print()

        for ref_id, ref_info in info["references"].items():
            print(f"📚 {ref_id}")
            print(f"   Genome ID: {ref_info['genome_id']}")
            print(f"   Assembly: {ref_info['assembly']}")
            print(f"   Variants: {ref_info['variant_count']:,}")
            print(f"   Chromosomes: {', '.join(ref_info['chromosomes'])}")
            print(f"   Hash: {ref_info['hash']}")
            print()


def demo_use_with_differential_encoding():
    """Demo: Using references with differential encoding."""
    print_section("5. USING REFERENCES WITH DIFFERENTIAL ENCODING")

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_dir = Path(tmpdir)

        # Setup references
        print("Setting up references...")
        manager = setup_default_references(
            reference_dir=ref_dir,
            use_case="development",
        )
        print(f"✅ Loaded {manager.reference_count} reference(s)")
        print()

        # Create encoder with references
        print("Creating differential encoder...")
        encoder = UnifiedGenomicEncoder(
            mode=EncodingMode.DIFFERENTIAL,
            reference_dir=ref_dir,
            dimension=1000,
            seed=42,
        )
        print(f"✅ Encoder initialized with {encoder.reference_manager.reference_count} reference(s)")
        print()

        # Create a test genome
        print("Creating test genome...")
        genome = Genome(
            genome_id="test_patient_001",
            assembly="GRCh38",
            chromosomes={
                "chr1": [
                    Variant(chromosome="chr1", position=100000, ref="A", alt="G", genotype="0/1", quality=99.0),
                    Variant(chromosome="chr1", position=200000, ref="C", alt="T", genotype="1/1", quality=98.0),
                ]
            }
        )
        print(f"✅ Created genome: {genome.genome_id}")
        print()

        # Encode genome
        print("Encoding genome with differential encoding...")
        encoded = encoder.encode_genome(
            genome=genome,
            analysis_type=AnalysisType.GENE_REGION,
            bundle_chunks=True,
        )
        print(f"✅ Encoding complete!")
        print(f"   Genome ID: {encoded.genome_id}")
        print(f"   Chunks: {len(encoded.chunk_hypervectors)}")
        print(f"   Dimension: {len(encoded.bundled_hypervector)}")
        print(f"   Storage size: {encoded.storage_size_kb():.2f} KB")
        print(f"   Verified: {encoded.verify()}")
        print()

        # Save
        save_path = ref_dir / "encoded_genome.enc.gz"
        compressed_bytes = encoded.save(save_path, compress=True)
        print(f"✅ Saved to: {save_path.name}")
        print(f"   Compressed size: {compressed_bytes / 1024:.2f} KB")
        print()


def demo_advanced_usage():
    """Demo: Advanced usage patterns."""
    print_section("6. ADVANCED USAGE")

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_dir = Path(tmpdir)

        # Multiple use cases
        print("Scenario A: Switching between use cases")
        print("-" * 80)
        print()

        for use_case in ["development"]:  # Only development for demo
            print(f"Setting up {use_case} references...")
            manager = setup_default_references(
                reference_dir=ref_dir / use_case,
                use_case=use_case,
            )
            print(f"✅ {use_case}: {manager.reference_count} reference(s)")
            print()

        print()
        print("Scenario B: Custom reference selection")
        print("-" * 80)
        print()

        custom_refs = download_reference_genomes(
            sources=["synthetic_test"],
            output_dir=ref_dir / "custom",
        )
        print(f"✅ Downloaded {len(custom_refs)} custom reference(s)")
        print()

        print()
        print("Scenario C: Validation and monitoring")
        print("-" * 80)
        print()

        manager = SecureReferenceGenomeManager(reference_dir=ref_dir / "development")
        result = validate_reference_pool(manager)

        print(f"Validation: {'✅ PASS' if result.is_valid else '❌ FAIL'}")
        print(f"References: {result.reference_count}")
        print(f"Issues: {len(result.errors) + len(result.warnings)}")
        print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("REFERENCE GENOME SETUP DEMO")
    print("=" * 80)
    print()
    print("This demo shows how to download, validate, and use reference genomes")
    print("for differential encoding in GenomeVault.")
    print()

    # Run demos
    demo_download_references()
    demo_validate_references()
    demo_setup_default_references()
    demo_get_reference_info()
    demo_use_with_differential_encoding()
    demo_advanced_usage()

    # Summary
    print_section("DEMO COMPLETE")
    print("All demonstrations completed successfully!")
    print()
    print("Next steps:")
    print("  1. Run the interactive wizard: python scripts/genomevault_setup_references.py")
    print("  2. Read the documentation: docs/reference_genome_setup.md")
    print("  3. Explore the API: genomevault/differential_encoding/reference_setup.py")
    print()
    print("For production use:")
    print("  • Use production-grade references (gnomAD, 1000 Genomes)")
    print("  • Set up proper validation and monitoring")
    print("  • Configure backups and redundancy")
    print()


if __name__ == "__main__":
    main()
