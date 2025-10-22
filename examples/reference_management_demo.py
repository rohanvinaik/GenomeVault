#!/usr/bin/env python3
"""
Demonstration of Reference Genome Management.

This script demonstrates the reference genome management system for differential
encoding, including:
- Creating and managing genomic variants
- Building reference genomes with position indexing
- VCF file parsing
- Cryptographic verification
- Random reference selection
- Genomic section extraction
"""

import gzip
import tempfile
from pathlib import Path

from genomevault.differential_encoding import (
    Variant,
    GenomeSection,
    ReferenceGenome,
    ReferencePool,
    SecureReferenceGenomeManager,
    CryptoRNG,
    compute_reference_hash,
)


def demo_variant_operations():
    """Demonstrate Variant creation and operations."""
    print("=" * 70)
    print("1. Genomic Variant Operations")
    print("=" * 70)

    # Create variants
    print("\n📍 Creating genomic variants...")
    variant1 = Variant(
        chromosome="chr7",
        position=117,
        ref="C",
        alt="T",
        genotype="0/1",
        quality=0.99,
        info={"GENE": "CFTR", "IMPACT": "HIGH"}
    )

    variant2 = Variant(
        chromosome="chr7",
        position=155230,
        ref="A",
        alt="G",
        genotype="1/1"
    )

    print(f"   Variant 1: {variant1}")
    print(f"   Variant 2: {variant2}")

    # Sorting
    print("\n📍 Sorting variants by position...")
    variants = [variant2, variant1]  # Wrong order
    variants.sort()
    print(f"   Sorted: {[v.position for v in variants]}")

    print("\n✅ Variant operations complete\n")


def demo_genome_section():
    """Demonstrate GenomeSection functionality."""
    print("=" * 70)
    print("2. Genomic Section Management")
    print("=" * 70)

    # Create section with variants
    print("\n📍 Creating genomic section...")
    variants = [
        Variant(chromosome="chr1", position=120000, ref="A", alt="G"),
        Variant(chromosome="chr1", position=150000, ref="C", alt="T"),
        Variant(chromosome="chr1", position=180000, ref="G", alt="A"),
    ]

    section = GenomeSection(
        chromosome="chr1",
        start_position=100000,
        end_position=200000,
        variants=variants
    )

    print(f"   Section: {section}")
    print(f"   Length: {section.length:,} bp")
    print(f"   Variants: {section.variant_count}")
    print(f"   First variant: {section.variants[0]}")

    print("\n✅ Genome section demo complete\n")


def demo_reference_genome():
    """Demonstrate ReferenceGenome with position indexing."""
    print("=" * 70)
    print("3. Reference Genome with Position Indexing")
    print("=" * 70)

    # Create reference genome
    print("\n📍 Creating reference genome...")
    variants = {
        "chr1": [
            Variant(chromosome="chr1", position=10000, ref="A", alt="G", genotype="0/1"),
            Variant(chromosome="chr1", position=20000, ref="C", alt="T", genotype="0/1"),
            Variant(chromosome="chr1", position=30000, ref="G", alt="A", genotype="1/1"),
        ],
        "chr2": [
            Variant(chromosome="chr2", position=50000, ref="T", alt="C", genotype="0/1"),
            Variant(chromosome="chr2", position=60000, ref="A", alt="G", genotype="1/1"),
        ]
    }

    ref = ReferenceGenome(
        genome_id="GRCh38",
        assembly="GRCh38.p13",
        variants=variants,
        cryptographic_hash="",  # Will be computed
        source="NCBI",
        population="reference"
    )

    # Compute hash
    ref.cryptographic_hash = compute_reference_hash(ref)

    print(f"   Genome ID: {ref.genome_id}")
    print(f"   Assembly: {ref.assembly}")
    print(f"   Chromosomes: {ref.chromosomes}")
    print(f"   Total variants: {ref.total_variants}")
    print(f"   Hash: {ref.cryptographic_hash[:32]}...")

    # Query section
    print("\n📍 Querying genomic section (chr1:5000-25000)...")
    section = ref.get_section("chr1", 5000, 25000)
    print(f"   Found {section.variant_count} variants")
    for var in section.variants:
        print(f"     - {var}")

    # Position index
    print("\n📍 Position index statistics...")
    for chrom, tree in ref.position_index.items():
        print(f"   {chrom}: {len(tree)} variants indexed")

    print("\n✅ Reference genome demo complete\n")


def demo_reference_pool():
    """Demonstrate ReferencePool management."""
    print("=" * 70)
    print("4. Reference Pool Management")
    print("=" * 70)

    # Create pool
    print("\n📍 Creating reference pool...")
    pool = ReferencePool()

    # Create and add references
    print("\n📍 Adding multiple reference genomes...")
    for genome_id in ["GRCh38", "GRCh37", "CHM13"]:
        variants = {
            "chr1": [
                Variant(chromosome="chr1", position=1000 * ord(genome_id[0]),
                       ref="A", alt="G", genotype="0/1")
            ]
        }

        ref = ReferenceGenome(
            genome_id=genome_id,
            assembly=genome_id,
            variants=variants,
            cryptographic_hash=""
        )
        ref.cryptographic_hash = compute_reference_hash(ref)

        pool.add_reference(ref, verify=True)
        print(f"   Added: {genome_id}")

    print(f"\n   Pool size: {pool.size}")
    print(f"   Genome IDs: {pool.genome_ids}")

    # Verify all
    print("\n📍 Verifying cryptographic hashes...")
    if pool.verify_all():
        print("   ✅ All references verified successfully")
        for genome_id, status in pool.verification_status.items():
            print(f"      {genome_id}: {'✅ PASS' if status else '❌ FAIL'}")

    print("\n✅ Reference pool demo complete\n")


def demo_vcf_parsing():
    """Demonstrate VCF file parsing."""
    print("=" * 70)
    print("5. VCF File Parsing")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test VCF file
        print("\n📍 Creating test VCF file...")
        vcf_content = """##fileformat=VCFv4.2
##reference=GRCh38
##source=TestData
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
chr1\t10177\trs367896724\tA\tAC\t100\tPASS\tDP=50;AF=0.425319\tGT:DP\t0/1:30
chr1\t10352\trs555500075\tT\tTA\t99\tPASS\tDP=45;AF=0.395207\tGT:DP\t0/1:28
chr7\t117199646\trs113993960\tC\tT\t100\tPASS\tGENE=CFTR;IMPACT=HIGH\tGT:DP\t0/1:50
"""

        vcf_path = tmpdir_path / "test.vcf.gz"
        with gzip.open(vcf_path, "wt") as f:
            f.write(vcf_content)

        print(f"   Created: {vcf_path.name}")

        # Parse VCF
        print("\n📍 Initializing reference manager...")
        manager = SecureReferenceGenomeManager(tmpdir_path)

        print(f"   Loaded {manager.reference_count} reference(s)")

        # Get the loaded reference
        ref = manager.pool.get_reference("test")

        print(f"\n   Reference: {ref.genome_id}")
        print(f"   Assembly: {ref.assembly}")
        print(f"   Source: {ref.source}")
        print(f"   Variants: {ref.total_variants}")

        # Show parsed variants
        print("\n   Parsed variants:")
        for chrom, vars in ref.variants.items():
            print(f"\n   {chrom}: {len(vars)} variants")
            for var in vars:
                gene_info = var.info.get("GENE", "-")
                impact = var.info.get("IMPACT", "-")
                print(f"      pos={var.position:,} {var.ref}>{var.alt} "
                      f"GT={var.genotype} gene={gene_info} impact={impact}")

    print("\n✅ VCF parsing demo complete\n")


def demo_secure_manager():
    """Demonstrate SecureReferenceGenomeManager."""
    print("=" * 70)
    print("6. Secure Reference Manager with Random Selection")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create multiple reference VCFs
        print("\n📍 Creating reference genome pool...")

        references = {
            "GRCh38": """##fileformat=VCFv4.2
##reference=GRCh38
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
chr1\t10000\t.\tA\tG\t100\tPASS\t.\tGT\t0/1
chr1\t20000\t.\tC\tT\t100\tPASS\t.\tGT\t0/1
""",
            "GRCh37": """##fileformat=VCFv4.2
##reference=GRCh37
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
chr1\t10500\t.\tG\tA\t100\tPASS\t.\tGT\t1/1
chr1\t21000\t.\tT\tC\t100\tPASS\t.\tGT\t1/1
""",
            "CHM13": """##fileformat=VCFv4.2
##reference=CHM13
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
chr1\t11000\t.\tA\tT\t100\tPASS\t.\tGT\t0/1
chr1\t22000\t.\tC\tG\t100\tPASS\t.\tGT\t0/1
"""
        }

        for genome_id, content in references.items():
            vcf_path = tmpdir_path / f"{genome_id}.vcf.gz"
            with gzip.open(vcf_path, "wt") as f:
                f.write(content)
            print(f"   Created: {genome_id}.vcf.gz")

        # Initialize manager with crypto RNG
        print("\n📍 Initializing secure reference manager...")
        rng = CryptoRNG(master_seed=b"\x42" * 32)  # Deterministic for demo
        manager = SecureReferenceGenomeManager(tmpdir_path, crypto_rng=rng)

        print(f"   Loaded references: {manager.genome_ids}")
        print(f"   Reference count: {manager.reference_count}")

        # Demonstrate random selection
        print("\n📍 Demonstrating cryptographic random selection...")
        for i in range(3):
            chunk_seed = rng.derive_seed(f"chunk_{i}".encode())
            selected = manager.get_random_reference(chunk_seed)
            print(f"   Chunk {i}: seed={chunk_seed[:8].hex()}... → {selected.genome_id}")

        # Show determinism
        print("\n📍 Verifying determinism (same seed → same reference)...")
        seed1 = rng.derive_seed(b"test_seed")
        ref1 = manager.get_random_reference(seed1)
        ref2 = manager.get_random_reference(seed1)
        print(f"   First call:  {ref1.genome_id}")
        print(f"   Second call: {ref2.genome_id}")
        print(f"   Match: {ref1.genome_id == ref2.genome_id} ✅")

        # Extract section
        print("\n📍 Extracting genomic section...")
        section = manager.get_reference_section(
            ref1.genome_id,
            "chr1",
            9000,
            23000
        )
        print(f"   Section: {section}")
        print(f"   Variants in section: {section.variant_count}")

    print("\n✅ Secure manager demo complete\n")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("Reference Genome Management Demonstration")
    print("=" * 70)
    print("\nDemonstrating reference genome management for differential encoding")
    print("with cryptographic verification and secure random selection.\n")

    demo_variant_operations()
    demo_genome_section()
    demo_reference_genome()
    demo_reference_pool()
    demo_vcf_parsing()
    demo_secure_manager()

    print("=" * 70)
    print("All demonstrations completed successfully! ✅")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✅ Variant creation and validation")
    print("  ✅ Genomic section management")
    print("  ✅ Position indexing with IntervalTree")
    print("  ✅ Reference genome with cryptographic hashing")
    print("  ✅ Reference pool with verification")
    print("  ✅ VCF file parsing")
    print("  ✅ Secure random reference selection")
    print("  ✅ Deterministic behavior")
    print("=" * 70)


if __name__ == "__main__":
    main()
