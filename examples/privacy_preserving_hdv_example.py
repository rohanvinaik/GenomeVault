#!/usr/bin/env python3
"""
Example: Privacy-Preserving Genome HDV Encoding

Demonstrates the hybrid HDV architecture for nucleotide-resolution queries
with full privacy preservation.

This example shows:
1. Encoding a genome with different schemas (nucleotide-resolution, phenotype-risk, casual-health)
2. Querying nucleotides with privacy preservation
3. Multi-encoding voting for accuracy
4. Storage and performance characteristics
"""

from pathlib import Path
from genomevault.hypervector_transform import (
    PrivacyPreservingGenomeHDV,
    EncodingSchema,
    SchemaConfig,
)

def example_nucleotide_resolution():
    """Example 1: Nucleotide-resolution encoding (stress test)"""

    print("=" * 80)
    print("EXAMPLE 1: NUCLEOTIDE-RESOLUTION ENCODING")
    print("=" * 80)
    print("\nThis is the most stringent stress-test for HDC.")
    print("Nucleotide resolution is less aligned with HDC structural advantages,")
    print("so success here validates the entire approach.\n")

    # Paths
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    local_guide_dir = Path("/Volumes/1TBStorage/guide_strands")

    # Create encoder with nucleotide resolution
    encoder = PrivacyPreservingGenomeHDV(
        gdiff_path=gdiff_path,
        local_guide_dir=local_guide_dir,
        schema=EncodingSchema.NUCLEOTIDE_RESOLUTION,
        num_encodings=3,  # 3 independent encodings for voting
        use_gpu=True  # Enable Metal/CUDA acceleration
    )

    print("Encoding genome...")
    encoder.encode()

    print("\nSaving HDV database...")
    encoder.save(Path("genome_nucleotide_hdv.npz"))

    print("\nQuerying nucleotides with privacy...")

    # Query specific positions
    queries = [
        ("chr1", 12345),
        ("chr2", 100000),
        ("chr22", 50000),
    ]

    for chrom, pos in queries:
        result = encoder.query(chrom=chrom, pos=pos)
        print(f"  {chrom}:{pos} = {result.nucleotide} (confidence: {result.confidence:.1%}, votes: {result.votes})")

    encoder.close()

    print("\n✓ Nucleotide-resolution encoding complete")


def example_phenotype_risk():
    """Example 2: Phenotype risk encoding (hospitals/clinical)"""

    print("\n" + "=" * 80)
    print("EXAMPLE 2: PHENOTYPE RISK ENCODING (CLINICAL)")
    print("=" * 80)
    print("\nOptimized for disease phenotype and risk assessment.")
    print("Used by hospitals for clinical genomics.\n")

    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    local_guide_dir = Path("/Volumes/1TBStorage/guide_strands")

    encoder = PrivacyPreservingGenomeHDV(
        gdiff_path=gdiff_path,
        local_guide_dir=local_guide_dir,
        schema=EncodingSchema.PHENOTYPE_RISK,
        num_encodings=5,  # Higher accuracy for clinical use
        use_gpu=True
    )

    print("Encoding genome with phenotype risk schema...")
    encoder.encode()

    print("\nSaving HDV database...")
    encoder.save(Path("genome_phenotype_hdv.npz"))

    print("\nStorage requirements:")
    print(f"  Total: {encoder._estimate_storage_gb() * encoder.num_encodings:.2f} GB")
    print(f"  Regions: {len(encoder.region_index):,}")
    print(f"  Dimension: {encoder.config.dimension:,}D")

    encoder.close()

    print("\n✓ Phenotype risk encoding complete")


def example_casual_health():
    """Example 3: Casual health encoding (consumer genomics)"""

    print("\n" + "=" * 80)
    print("EXAMPLE 3: CASUAL HEALTH ENCODING (CONSUMER)")
    print("=" * 80)
    print("\nMinimal data for lifestyle/consumer genomics.")
    print("Key nucleotides only, optimized for speed and privacy.\n")

    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    local_guide_dir = Path("/Volumes/1TBStorage/guide_strands")

    encoder = PrivacyPreservingGenomeHDV(
        gdiff_path=gdiff_path,
        local_guide_dir=local_guide_dir,
        schema=EncodingSchema.CASUAL_HEALTH,
        num_encodings=3,
        use_gpu=True
    )

    print("Encoding genome with casual health schema...")
    encoder.encode()

    print("\nSaving HDV database...")
    encoder.save(Path("genome_casual_hdv.npz"))

    print("\nComparing schemas:")
    print(f"  Casual health: {encoder.config.dimension:,}D, {encoder.config.region_size:,} bp regions")
    print(f"  Storage: {encoder._estimate_storage_gb() * encoder.num_encodings:.2f} GB")
    print("\nVs Nucleotide resolution: 10,000D, 10,000 bp regions, ~36 GB")

    encoder.close()

    print("\n✓ Casual health encoding complete")


def example_custom_schema():
    """Example 4: Custom schema configuration"""

    print("\n" + "=" * 80)
    print("EXAMPLE 4: CUSTOM SCHEMA CONFIGURATION")
    print("=" * 80)
    print("\nCreate a custom schema for specific research needs.\n")

    # Custom configuration
    custom_config = SchemaConfig(
        schema=EncodingSchema.NUCLEOTIDE_RESOLUTION,
        dimension=15_000,  # Custom dimension
        region_size=20_000,  # 20 KB regions
        include_variants=True,
        include_reference=True,
        reference_sampling_rate=0.5,  # 50% reference sampling
        target_genes=["BRCA1", "BRCA2", "TP53"],  # Prioritize specific genes
        min_base_quality=30  # Higher quality threshold
    )

    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    local_guide_dir = Path("/Volumes/1TBStorage/guide_strands")

    encoder = PrivacyPreservingGenomeHDV(
        gdiff_path=gdiff_path,
        local_guide_dir=local_guide_dir,
        schema=EncodingSchema.NUCLEOTIDE_RESOLUTION,
        num_encodings=3,
        custom_config=custom_config,  # Use custom config
        use_gpu=True
    )

    print("Custom configuration:")
    print(f"  Dimension: {custom_config.dimension:,}D")
    print(f"  Region size: {custom_config.region_size:,} bp")
    print(f"  Reference sampling: {custom_config.reference_sampling_rate*100:.1f}%")
    print(f"  Target genes: {custom_config.target_genes}")

    print("\nEncoding genome...")
    encoder.encode()

    encoder.close()

    print("\n✓ Custom schema encoding complete")


def example_loading_and_querying():
    """Example 5: Loading pre-encoded HDV database"""

    print("\n" + "=" * 80)
    print("EXAMPLE 5: LOADING PRE-ENCODED HDV DATABASE")
    print("=" * 80)
    print("\nLoad a previously encoded HDV database and query it.\n")

    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    local_guide_dir = Path("/Volumes/1TBStorage/guide_strands")

    # Create encoder (no encoding yet)
    encoder = PrivacyPreservingGenomeHDV(
        gdiff_path=gdiff_path,
        local_guide_dir=local_guide_dir,
        schema=EncodingSchema.NUCLEOTIDE_RESOLUTION,
        num_encodings=3,
        use_gpu=True
    )

    # Load pre-encoded database
    print("Loading HDV database from disk...")
    encoder.load(Path("genome_nucleotide_hdv.npz"))

    print("✓ Database loaded")

    # Query
    print("\nQuerying nucleotides...")
    for i in range(10):
        # Random queries
        import random
        chrom = random.choice(["chr1", "chr2", "chr3", "chr22"])
        pos = random.randint(10000, 100000)

        try:
            result = encoder.query(chrom=chrom, pos=pos)
            print(f"  {chrom}:{pos} = {result.nucleotide} (conf: {result.confidence:.1%})")
        except ValueError:
            print(f"  {chrom}:{pos} = NOT IN ENCODED REGION")

    encoder.close()

    print("\n✓ Loading and querying complete")


if __name__ == "__main__":
    print("Privacy-Preserving Genome HDV - Usage Examples")
    print("=" * 80)
    print("\nThese examples demonstrate the hybrid HDV architecture for")
    print("nucleotide-resolution queries with full privacy preservation.")
    print("\nNOTE: Requires completed k=11 GDiff encoding at:")
    print("  data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    print("\n" + "=" * 80)

    # Choose which example to run
    print("\nAvailable examples:")
    print("  1. Nucleotide-resolution encoding (stress test)")
    print("  2. Phenotype risk encoding (clinical)")
    print("  3. Casual health encoding (consumer)")
    print("  4. Custom schema configuration")
    print("  5. Loading and querying pre-encoded database")
    print("\nRun with: python examples/privacy_preserving_hdv_example.py")

    # Uncomment to run specific examples:
    # example_nucleotide_resolution()
    # example_phenotype_risk()
    # example_casual_health()
    # example_custom_schema()
    # example_loading_and_querying()
