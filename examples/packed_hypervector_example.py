"""
Example usage of packed hypervector encoding for genomic data
"""

import numpy as np

from genomevault.hypervector.encoding import PackedGenomicEncoder


def main():
    """Demonstrate packed hypervector usage"""

    logger.info("Packed Hypervector Genomic Encoding Example")
    logger.info("=" * 50)

    # Initialize encoder with packed mode
    encoder = PackedGenomicEncoder(
        dimension=10000,
        packed=True,  # Enable bit-packing
        device="cpu",  # Use 'gpu' if CUDA available
    )

    logger.info("\nEncoder initialized:")
    logger.info(f"- Dimension: {encoder.dimension}")
    logger.info(f"- Memory efficiency: {encoder.memory_efficiency}x")
    logger.info(f"- Bits per hypervector: {encoder.dimension}")
    logger.info(f"- Bytes per hypervector: {(encoder.dimension + 63) // 64 * 8}")

    # Example 1: Encode a single variant
    logger.info("\n\nExample 1: Single Variant Encoding")
    logger.info("-" * 30)

    variant = {
        "chromosome": "chr7",
        "position": 117559590,  # CFTR gene position
        "ref": "G",
        "alt": "A",
        "type": "SNP",
    }

    hv = encoder.encode_variant(**variant)
    logger.info(f"Encoded variant: {variant}")
    logger.info(f"Hypervector type: {type(hv).__name__}")
    logger.info(f"Memory usage: {hv.memory_bytes} bytes")

    # Example 2: Encode multiple variants (a genome)
    logger.info("\n\nExample 2: Genome Encoding")
    logger.info("-" * 30)

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
    logger.info(f"Encoded {len(variants)} variants into genome hypervector")
    logger.info(f"Memory usage: {genome_hv.memory_bytes} bytes")
    logger.info(
        f"Memory per variant: {genome_hv.memory_bytes / len(variants):.1f} bytes"
    )

    # Example 3: Compare two genomes
    logger.info("\n\nExample 3: Genome Similarity")
    logger.info("-" * 30)

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
    logger.info(f"Similarity between genomes: {similarity:.4f}")

    # Compare with completely different genome
    variants_different = [
        {
            "chromosome": "chr2",
            "position": 12345678,
            "ref": "A",
            "alt": "T",
            "type": "SNP",
        },
        {
            "chromosome": "chr3",
            "position": 87654321,
            "ref": "G",
            "alt": "C",
            "type": "SNP",
        },
    ]

    genome_hv3 = encoder.encode_genome(variants_different)
    similarity2 = encoder.similarity(genome_hv, genome_hv3)
    logger.info(f"Similarity with different genome: {similarity2:.4f}")

    # Example 4: Memory comparison
    logger.info("\n\nExample 4: Memory Efficiency")
    logger.info("-" * 30)

    # Traditional float32 representation
    traditional_bytes = encoder.dimension * 4  # 4 bytes per float32
    packed_bytes = genome_hv.memory_bytes

    logger.info(f"Traditional (float32): {traditional_bytes:,} bytes")
    logger.info(f"Packed (bit-packed): {packed_bytes:,} bytes")
    logger.info(f"Compression ratio: {traditional_bytes / packed_bytes:.1f}x")
    logger.info(f"Memory saved: {(1 - packed_bytes / traditional_bytes) * 100:.1f}%")

    # Example 5: Batch processing
    logger.info("\n\nExample 5: Batch Processing")
    logger.info("-" * 30)

    # Generate many random variants
    n_genomes = 100
    n_variants_per_genome = 50

    logger.info(
        f"Processing {n_genomes} genomes with {n_variants_per_genome} variants each..."
    )

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
    logger.info(f"Total memory for {n_genomes} genomes: {total_memory / 1024:.2f} KB")
    logger.info(f"Average memory per genome: {total_memory / n_genomes / 1024:.2f} KB")

    # Example 6: Hypervector operations
    logger.info("\n\nExample 6: Hypervector Operations")
    logger.info("-" * 30)

    # Create two hypervectors
    hv1 = genomes[0]
    hv2 = genomes[1]

    # XOR (binding) - useful for creating associations
    bound = hv1.xor(hv2)
    logger.info("Created bound hypervector (hv1 XOR hv2)")

    # Majority vote (bundling) - useful for creating summaries
    bundled = hv1.majority(genomes[1:5])
    logger.info("Created bundled hypervector from 5 genomes")

    # Check similarity preservation
    sim_original = encoder.similarity(hv1, hv2)
    sim_after_ops = encoder.similarity(bound, bundled)
    logger.info(f"Similarity between original HVs: {sim_original:.4f}")
    logger.info(f"Similarity after operations: {sim_after_ops:.4f}")

    logger.info("\n" + "=" * 50)
    logger.info("Packed hypervector encoding demonstration complete!")


if __name__ == "__main__":
    main()
