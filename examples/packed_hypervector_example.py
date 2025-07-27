from typing import Any, Dict

"""
Example usage of packed hypervector encoding for genomic data
"""

import numpy as np

from genomevault.hypervector.encoding import PackedGenomicEncoder


def main() -> None:
       """TODO: Add docstring for main"""
     """Demonstrate packed hypervector usage"""

    print("Packed Hypervector Genomic Encoding Example")
    print("=" * 50)

    # Initialize encoder with packed mode
    encoder = PackedGenomicEncoder(
        dimension=10000,
        packed=True,  # Enable bit-packing
        device="cpu",  # Use 'gpu' if CUDA available
    )

    print(f"\nEncoder initialized:")
    print(f"- Dimension: {encoder.dimension}")
    print(f"- Memory efficiency: {encoder.memory_efficiency}x")
    print(f"- Bits per hypervector: {encoder.dimension}")
    print(f"- Bytes per hypervector: {(encoder.dimension + 63) // 64 * 8}")

    # Example 1: Encode a single variant
    print("\n\nExample 1: Single Variant Encoding")
    print("-" * 30)

    variant = {
        "chromosome": "chr7",
        "position": 117559590,  # CFTR gene position
        "ref": "G",
        "alt": "A",
        "type": "SNP",
    }

    hv = encoder.encode_variant(**variant)
    print(f"Encoded variant: {variant}")
    print(f"Hypervector type: {type(hv).__name__}")
    print(f"Memory usage: {hv.memory_bytes} bytes")

    # Example 2: Encode multiple variants (a genome)
    print("\n\nExample 2: Genome Encoding")
    print("-" * 30)

    # Common disease-associated variants
    variants = [
        {
            "chromosome": "chr7",
            "position": 117559590,  # CFTR - Cystic Fibrosis
            "ref": "G",
            "alt": "A",
            "type": "SNP",
        },
        {
            "chromosome": "chr17",
            "position": 41245466,  # BRCA1 - Breast Cancer
            "ref": "G",
            "alt": "A",
            "type": "SNP",
        },
        {
            "chromosome": "chr13",
            "position": 32914437,  # BRCA2 - Breast Cancer
            "ref": "T",
            "alt": "C",
            "type": "SNP",
        },
        {
            "chromosome": "chr1",
            "position": 161599693,  # APOB - Hypercholesterolemia
            "ref": "C",
            "alt": "T",
            "type": "SNP",
        },
    ]

    genome_hv = encoder.encode_genome(variants)
    print(f"Encoded {len(variants)} variants into genome hypervector")
    print(f"Memory usage: {genome_hv.memory_bytes} bytes")
    print(f"Memory per variant: {genome_hv.memory_bytes / len(variants):.1f} bytes")

    # Example 3: Compare two genomes
    print("\n\nExample 3: Genome Similarity")
    print("-" * 30)

    # Create a similar genome with one variant difference
    variants_similar = variants[:-1]  # Remove last variant
    variants_similar.append(
        {
            "chromosome": "chr1",
            "position": 161599693,
            "ref": "C",
            "alt": "G",  # Different alt allele
            "type": "SNP",
        }
    )

    genome_hv2 = encoder.encode_genome(variants_similar)

    similarity = encoder.similarity(genome_hv, genome_hv2)
    print(f"Similarity between genomes: {similarity:.4f}")

    # Compare with completely different genome
    variants_different = [
        {"chromosome": "chr2", "position": 12345678, "ref": "A", "alt": "T", "type": "SNP"},
        {"chromosome": "chr3", "position": 87654321, "ref": "G", "alt": "C", "type": "SNP"},
    ]

    genome_hv3 = encoder.encode_genome(variants_different)
    similarity2 = encoder.similarity(genome_hv, genome_hv3)
    print(f"Similarity with different genome: {similarity2:.4f}")

    # Example 4: Memory comparison
    print("\n\nExample 4: Memory Efficiency")
    print("-" * 30)

    # Traditional float32 representation
    traditional_bytes = encoder.dimension * 4  # 4 bytes per float32
    packed_bytes = genome_hv.memory_bytes

    print(f"Traditional (float32): {traditional_bytes:,} bytes")
    print(f"Packed (bit-packed): {packed_bytes:,} bytes")
    print(f"Compression ratio: {traditional_bytes / packed_bytes:.1f}x")
    print(f"Memory saved: {(1 - packed_bytes/traditional_bytes) * 100:.1f}%")

    # Example 5: Batch processing
    print("\n\nExample 5: Batch Processing")
    print("-" * 30)

    # Generate many random variants
    n_genomes = 100
    n_variants_per_genome = 50

    print(f"Processing {n_genomes} genomes with {n_variants_per_genome} variants each...")

    genomes = []
    for i in range(n_genomes):
        # Generate random variants
        variants = []
        for j in range(n_variants_per_genome):
            variants.append(
                {
                    "chromosome": f"chr{np.random.randint(1, 23)}",
                    "position": np.random.randint(1, 250_000_000),
                    "ref": np.random.choice(["A", "T", "G", "C"]),
                    "alt": np.random.choice(["A", "T", "G", "C"]),
                    "type": "SNP",
                }
            )

        genome_hv = encoder.encode_genome(variants)
        genomes.append(genome_hv)

    total_memory = sum(g.memory_bytes for g in genomes)
    print(f"Total memory for {n_genomes} genomes: {total_memory / 1024:.2f} KB")
    print(f"Average memory per genome: {total_memory / n_genomes / 1024:.2f} KB")

    # Example 6: Hypervector operations
    print("\n\nExample 6: Hypervector Operations")
    print("-" * 30)

    # Create two hypervectors
    hv1 = genomes[0]
    hv2 = genomes[1]

    # XOR (binding) - useful for creating associations
    bound = hv1.xor(hv2)
    print(f"Created bound hypervector (hv1 XOR hv2)")

    # Majority vote (bundling) - useful for creating summaries
    bundled = hv1.majority(genomes[1:5])
    print(f"Created bundled hypervector from 5 genomes")

    # Check similarity preservation
    sim_original = encoder.similarity(hv1, hv2)
    sim_after_ops = encoder.similarity(bound, bundled)
    print(f"Similarity between original HVs: {sim_original:.4f}")
    print(f"Similarity after operations: {sim_after_ops:.4f}")

    print("\n" + "=" * 50)
    print("Packed hypervector encoding demonstration complete!")


if __name__ == "__main__":
    main()
