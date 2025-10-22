#!/usr/bin/env python3
"""
Demonstration of Cryptographic Primitives for Differential Encoding.

This script demonstrates the basic usage of the cryptographic primitives
implemented for differential genomic encoding.
"""

from dataclasses import dataclass
from typing import Dict, List

from genomevault.differential_encoding import (
    CryptoRNG,
    compute_chunk_id,
    compute_chunk_reference_binding,
    compute_reference_hash,
)


# Mock data structures for demonstration
@dataclass
class Variant:
    """Genomic variant."""

    position: int
    ref: str
    alt: str
    genotype: str = "0/1"


@dataclass
class GenomeChunk:
    """Genomic chunk with variants."""

    chromosome: str
    start_position: int
    end_position: int
    variants: List[Variant]


@dataclass
class ReferenceGenome:
    """Reference genome."""

    assembly: str
    variants: Dict[str, List[Variant]]


def demo_crypto_rng():
    """Demonstrate CryptoRNG usage."""
    print("=" * 70)
    print("1. CryptoRNG - Cryptographically Secure Random Number Generator")
    print("=" * 70)

    # Initialize with deterministic seed for reproducibility
    print("\n📍 Initializing CryptoRNG with deterministic seed...")
    master_seed = b"\x42" * 32  # Deterministic for demonstration
    rng = CryptoRNG(master_seed=master_seed)
    print(f"   Master seed: {master_seed[:8].hex()}... (32 bytes)")
    print(f"   Counter: {rng.get_counter()}")

    # Derive seeds
    print("\n📍 Deriving seeds for different contexts...")
    chunk_seed = rng.derive_seed(b"chunk_123")
    ref_seed = rng.derive_seed(b"reference_selection")
    print(f"   Chunk seed:     {chunk_seed[:8].hex()}...")
    print(f"   Reference seed: {ref_seed[:8].hex()}...")
    print(f"   Counter after derivation: {rng.get_counter()}")

    # Generate random integers
    print("\n📍 Generating deterministic random integers...")
    for i in range(3):
        value = rng.random_int(0, 1000, chunk_seed)
        print(f"   Trial {i+1}: random_int(0, 1000) = {value} (deterministic!)")

    # Select from list
    print("\n📍 Cryptographically selecting from reference genome pool...")
    references = ["GRCh38", "GRCh37", "CHM13", "HG002", "HG003"]
    selected = rng.random_choice(references, ref_seed)
    print(f"   Available: {references}")
    print(f"   Selected:  {selected} (deterministic for this seed)")

    print("\n✅ CryptoRNG demonstration complete\n")


def demo_chunk_id():
    """Demonstrate chunk ID generation."""
    print("=" * 70)
    print("2. Chunk ID Generation - Collision-Resistant Identifiers")
    print("=" * 70)

    # Create sample chunk
    print("\n📍 Creating genomic chunk...")
    chunk = GenomeChunk(
        chromosome="chr1",
        start_position=100000,
        end_position=200000,
        variants=[
            Variant(position=120000, ref="A", alt="G"),
            Variant(position=150000, ref="C", alt="T"),
            Variant(position=180000, ref="G", alt="A"),
        ],
    )
    print(f"   Chromosome: {chunk.chromosome}")
    print(f"   Region: {chunk.start_position:,}-{chunk.end_position:,}")
    print(f"   Variants: {len(chunk.variants)}")

    # Generate chunk ID
    print("\n📍 Computing chunk identifier...")
    master_seed = b"\x00" * 32
    chunk_id = compute_chunk_id(chunk, master_seed)
    print(f"   Chunk ID: {chunk_id.hex()}")
    print(f"   Length: {len(chunk_id)} bytes (SHA-256)")

    # Demonstrate determinism
    print("\n📍 Verifying determinism...")
    chunk_id2 = compute_chunk_id(chunk, master_seed)
    print(f"   Second computation: {chunk_id2.hex()}")
    print(f"   Match: {chunk_id == chunk_id2} ✅")

    # Demonstrate collision resistance
    print("\n📍 Testing collision resistance...")
    chunk_modified = GenomeChunk(
        chromosome="chr1",
        start_position=100000,
        end_position=200000,
        variants=[
            Variant(position=120000, ref="A", alt="T"),  # Different variant
            Variant(position=150000, ref="C", alt="T"),
            Variant(position=180000, ref="G", alt="A"),
        ],
    )
    chunk_id_modified = compute_chunk_id(chunk_modified, master_seed)
    print(f"   Modified chunk ID: {chunk_id_modified.hex()}")
    print(f"   Different: {chunk_id != chunk_id_modified} ✅")

    print("\n✅ Chunk ID demonstration complete\n")


def demo_reference_hash():
    """Demonstrate reference genome hashing."""
    print("=" * 70)
    print("3. Reference Hash - Integrity Verification")
    print("=" * 70)

    # Create reference genome
    print("\n📍 Creating reference genome...")
    reference = ReferenceGenome(
        assembly="GRCh38",
        variants={
            "chr1": [
                Variant(position=1000, ref="A", alt="G", genotype="0/1"),
                Variant(position=2000, ref="C", alt="T", genotype="0/1"),
            ],
            "chr2": [
                Variant(position=5000, ref="G", alt="A", genotype="1/1"),
            ],
        },
    )
    print(f"   Assembly: {reference.assembly}")
    print(f"   Chromosomes: {list(reference.variants.keys())}")
    total_variants = sum(len(v) for v in reference.variants.values())
    print(f"   Total variants: {total_variants}")

    # Compute hash
    print("\n📍 Computing reference hash...")
    ref_hash = compute_reference_hash(reference)
    print(f"   Hash: {ref_hash}")
    print(f"   Length: {len(ref_hash)} characters (SHA-256 hex)")

    # Verify integrity
    print("\n📍 Verifying integrity...")
    ref_hash2 = compute_reference_hash(reference)
    print(f"   Second computation: {ref_hash2}")
    print(f"   Integrity verified: {ref_hash == ref_hash2} ✅")

    # Demonstrate tampering detection
    print("\n📍 Testing tampering detection...")
    reference.variants["chr1"][0].alt = "T"  # Modify variant
    tampered_hash = compute_reference_hash(reference)
    print(f"   Tampered hash: {tampered_hash}")
    print(f"   Tampering detected: {ref_hash != tampered_hash} ✅")

    print("\n✅ Reference hash demonstration complete\n")


def demo_chunk_reference_binding():
    """Demonstrate chunk-reference binding."""
    print("=" * 70)
    print("4. Chunk-Reference Binding - Cryptographic Association")
    print("=" * 70)

    # Generate chunk ID
    print("\n📍 Generating chunk ID...")
    chunk = GenomeChunk(
        chromosome="chr1",
        start_position=100000,
        end_position=200000,
        variants=[Variant(position=150000, ref="A", alt="G")],
    )
    master_seed = b"\x00" * 32
    chunk_id = compute_chunk_id(chunk, master_seed)
    print(f"   Chunk ID: {chunk_id[:8].hex()}...")

    # Create binding
    print("\n📍 Creating cryptographic binding...")
    reference_id = "GRCh38"
    binding = compute_chunk_reference_binding(chunk_id, reference_id)
    print(f"   Reference: {reference_id}")
    print(f"   Binding: {binding.hex()}")
    print(f"   Length: {len(binding)} bytes (HMAC-SHA256)")

    # Verify binding
    print("\n📍 Verifying chunk-reference association...")
    claimed_reference = "GRCh38"
    computed_binding = compute_chunk_reference_binding(chunk_id, claimed_reference)
    print(f"   Claimed reference: {claimed_reference}")
    print(f"   Verification: {computed_binding == binding} ✅")

    # Detect reference substitution attack
    print("\n📍 Detecting reference substitution attack...")
    wrong_reference = "GRCh37"
    forged_binding = compute_chunk_reference_binding(chunk_id, wrong_reference)
    print(f"   Attacker claims: {wrong_reference}")
    print(f"   Forged binding: {forged_binding.hex()}")
    print(f"   Attack detected: {forged_binding != binding} ✅")

    print("\n✅ Chunk-reference binding demonstration complete\n")


def demo_complete_workflow():
    """Demonstrate complete workflow."""
    print("=" * 70)
    print("5. Complete Workflow - End-to-End Example")
    print("=" * 70)

    # Step 1: Initialize RNG
    print("\n📍 Step 1: Initialize cryptographic RNG...")
    rng = CryptoRNG()
    master_seed = rng.master_seed
    print(f"   Master seed generated: {master_seed[:8].hex()}...")

    # Step 2: Create experimental chunk
    print("\n📍 Step 2: Create experimental genome chunk...")
    chunk = GenomeChunk(
        chromosome="chr7",
        start_position=500000,
        end_position=600000,
        variants=[
            Variant(position=520000, ref="A", alt="G"),
            Variant(position=550000, ref="C", alt="T"),
        ],
    )
    print(f"   Location: {chunk.chromosome}:{chunk.start_position}-{chunk.end_position}")

    # Step 3: Generate chunk ID
    print("\n📍 Step 3: Generate cryptographic chunk identifier...")
    chunk_seed = rng.derive_seed(b"chunk_encoding")
    chunk_id = compute_chunk_id(chunk, chunk_seed)
    print(f"   Chunk ID: {chunk_id[:16].hex()}...")

    # Step 4: Select random reference
    print("\n📍 Step 4: Cryptographically select reference genome...")
    reference_pool = ["GRCh38", "GRCh37", "CHM13", "HG002"]
    ref_seed = rng.derive_seed(chunk_id)
    selected_reference = rng.random_choice(reference_pool, ref_seed)
    print(f"   Pool: {reference_pool}")
    print(f"   Selected: {selected_reference}")

    # Step 5: Create cryptographic binding
    print("\n📍 Step 5: Create chunk-reference binding...")
    binding = compute_chunk_reference_binding(chunk_id, selected_reference)
    print(f"   Binding: {binding[:16].hex()}...")

    # Step 6: Compute reference hash
    print("\n📍 Step 6: Verify reference genome integrity...")
    reference = ReferenceGenome(
        assembly=selected_reference,
        variants={
            "chr7": [Variant(position=510000, ref="G", alt="A", genotype="0/1")]
        },
    )
    ref_hash = compute_reference_hash(reference)
    print(f"   Reference hash: {ref_hash[:32]}...")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Encoded Chunk Metadata")
    print("=" * 70)
    print(f"  Chunk ID:        {chunk_id.hex()}")
    print(f"  Reference:       {selected_reference}")
    print(f"  Ref Hash:        {ref_hash}")
    print(f"  Binding:         {binding.hex()}")
    print(f"  Location:        {chunk.chromosome}:{chunk.start_position}-{chunk.end_position}")
    print(f"  Variants:        {len(chunk.variants)}")

    print("\n✅ Complete workflow demonstration finished\n")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("Differential Encoding - Cryptographic Primitives Demonstration")
    print("=" * 70)
    print("\nThis demo shows the cryptographic foundation for differential encoding")
    print("of genomic data with security guarantees.\n")

    demo_crypto_rng()
    demo_chunk_id()
    demo_reference_hash()
    demo_chunk_reference_binding()
    demo_complete_workflow()

    print("=" * 70)
    print("All demonstrations completed successfully! ✅")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Section 3: Reference Genome Management")
    print("  - Section 4: Cryptographic Chunking")
    print("  - Section 5: Differential Encoding")
    print("=" * 70)


if __name__ == "__main__":
    main()
