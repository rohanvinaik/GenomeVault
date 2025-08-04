#!/usr/bin/env python3
"""
Example usage of the unified hypervector encoder for HV-01 implementation.
Demonstrates sparse/orthogonal projections, cross-modal binding, and performance.
"""

import time

import numpy as np

from genomevault.hypervector.encoding import create_encoder


def main():
    print("GenomeVault Unified Hypervector Encoder Demo")
    print("=" * 50)

    # Example 1: Basic variant encoding with sparse projection
    print("\n1. Encoding genomic variants with sparse projection (10k dimensions)")
    encoder_sparse = create_encoder(
        dimension=10000, projection_type="sparse", sparse_density=0.1, seed=42
    )

    # Encode some variants
    variants = [
        ("chr1", 12345, "A", "G", "SNP"),
        ("chr2", 67890, "ATG", "A", "DEL"),
        ("chrX", 11111, "C", "CGG", "INS"),
    ]

    print(f"\nEncoding {len(variants)} variants...")
    variant_vectors = []
    for chrom, pos, ref, alt, var_type in variants:
        vec = encoder_sparse.encode_variant(chrom, pos, ref, alt, var_type)
        variant_vectors.append(vec)
        print(f"  {var_type} at {chrom}:{pos} {ref}>{alt} -> vector shape: {vec.shape}")

    # Calculate similarities
    print("\nPairwise similarities between variants:")
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            sim = encoder_sparse.similarity(variant_vectors[i], variant_vectors[j])
            print(f"  Variant {i + 1} vs Variant {j + 1}: {sim:.3f}")

    # Example 2: Orthogonal projection with higher dimensions
    print("\n2. Using orthogonal projection with 15k dimensions")
    encoder_ortho = create_encoder(
        dimension=15000, projection_type="orthogonal", seed=42
    )

    # Encode DNA sequences
    sequences = ["ATCGATCG", "ATCGATCG", "GGCCTTAA", "ATATATAT"]
    print(f"\nEncoding {len(sequences)} DNA sequences...")

    seq_vectors = []
    for seq in sequences:
        vec = encoder_ortho.encode_sequence(seq)
        seq_vectors.append(vec)
        print(f"  '{seq}' -> vector shape: {vec.shape}")

    # Check that identical sequences produce identical vectors
    sim_identical = encoder_ortho.similarity(seq_vectors[0], seq_vectors[1])
    print(f"\nSimilarity between identical sequences: {sim_identical:.6f}")

    # Example 3: Cross-modal binding
    print("\n3. Cross-modal binding (genomic + clinical data)")
    encoder_multi = create_encoder(dimension=20000, seed=42)

    # Create mock genomic and clinical feature vectors
    n_genomic_features = 100
    n_clinical_features = 50

    genomic_features = np.random.randn(1, n_genomic_features)
    clinical_features = np.random.randn(1, n_clinical_features)

    # Fit separate projections
    encoder_multi.fit(n_genomic_features)
    genomic_vec = encoder_multi.encode_genomic_features(genomic_features)[0]

    # For clinical, we'd use a separate encoder in practice
    encoder_clinical = create_encoder(dimension=20000, seed=43)
    encoder_clinical.fit(n_clinical_features)
    clinical_vec = encoder_clinical.encode_genomic_features(clinical_features)[0]

    # Perform cross-modal binding
    combined_vec = encoder_multi.cross_modal_binding(
        genomic_vec, clinical_vec, modality_weights={"genomic": 0.6, "clinical": 0.4}
    )

    print(f"Genomic vector shape: {genomic_vec.shape}")
    print(f"Clinical vector shape: {clinical_vec.shape}")
    print(f"Combined vector shape: {combined_vec.shape}")

    # Example 4: Performance benchmark
    print("\n4. Performance benchmark")
    print("Testing encoding speed for 10,000 variants...")

    start_time = time.time()
    n_variants = 10000

    for i in range(n_variants):
        encoder_sparse.encode_variant(
            f"chr{(i % 22) + 1}",
            i * 1000,
            "ACGT"[i % 4],
            "TGCA"[i % 4],
            ["SNP", "INS", "DEL"][i % 3],
        )

    elapsed_time = time.time() - start_time
    variants_per_second = n_variants / elapsed_time

    print(f"\nEncoded {n_variants} variants in {elapsed_time:.2f} seconds")
    print(f"Throughput: {variants_per_second:.0f} variants/second")
    print(f"Average time per variant: {elapsed_time / n_variants * 1000:.3f} ms")

    # Example 5: Bundle and unbundle operations
    print("\n5. Bundle and unbundle operations")
    encoder_bundle = create_encoder(dimension=10000, seed=42)
    encoder_bundle.fit(100)

    # Get base vectors
    vec_a = encoder_bundle._base_vectors["A"]
    vec_snp = encoder_bundle._base_vectors["SNP"]
    vec_chr1 = encoder_bundle._base_vectors["chr1"]

    # Bundle them
    from genomevault.hypervector.operations.binding import bundle

    bundled = bundle([vec_a, vec_snp, vec_chr1])

    # Try to decode
    components = encoder_bundle.decode_components(bundled, threshold=0.2)
    print("\nDecoded components from bundled vector:")
    for name, similarity in components:
        print(f"  {name}: similarity = {similarity:.3f}")

    print("\nDemo completed!")


if __name__ == "__main__":
    main()
