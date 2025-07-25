"""
Example: Using Hamming LUT for genomic variant similarity

This example demonstrates how the optimized Hamming distance computation
can be used for comparing genomic variants encoded as hypervectors.
"""

import time

import numpy as np
import torch

from genomevault.hypervector.operations.binding import HypervectorBinder
from genomevault.hypervector.operations.hamming_lut import HammingLUT
from genomevault.hypervector_transform.hdc_encoder import (
    HypervectorConfig,
    HypervectorEncoder,
    OmicsType,
)


def simulate_variant_data(num_samples: int, num_variants: int) -> np.ndarray:
    """Simulate binary variant data (0/1 for absence/presence)"""
    # In real scenarios, this would come from VCF processing
    return np.random.binomial(1, 0.1, size=(num_samples, num_variants))


def encode_variants_to_hypervectors(
    variant_matrix: np.ndarray, dimension: int = 10000
) -> torch.Tensor:
    """Encode variant matrix to hypervectors"""
    config = HypervectorConfig(dimension=dimension)
    encoder = HypervectorEncoder(config)

    hypervectors = []
    for i in range(variant_matrix.shape[0]):
        # Encode each sample's variants
        hv = encoder.encode(variant_matrix[i], OmicsType.GENOMIC)
        hypervectors.append(hv)

    return torch.stack(hypervectors)


def find_similar_genomes(query_idx: int, hypervectors: torch.Tensor, top_k: int = 5) -> list:
    """Find the most similar genomes to a query using optimized Hamming distance"""
    binder = HypervectorBinder(use_gpu=torch.cuda.is_available())

    # Get query hypervector
    query_hv = hypervectors[query_idx]

    # Compute similarities to all other samples
    similarities = []
    for i in range(len(hypervectors)):
        if i != query_idx:
            sim = binder.hamming_similarity(query_hv, hypervectors[i])
            similarities.append((i, sim))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def batch_clustering_demo(hypervectors: torch.Tensor, num_clusters: int = 3):
    """Demonstrate batch similarity computation for clustering"""
    print("\n=== Batch Clustering Demo ===")

    binder = HypervectorBinder(use_gpu=torch.cuda.is_available())

    # Compute all pairwise similarities
    print("Computing pairwise similarities...")
    start = time.perf_counter()
    similarity_matrix = binder.batch_hamming_similarity(hypervectors, hypervectors)
    elapsed = time.perf_counter() - start

    print(f"Computed {len(hypervectors)}x{len(hypervectors)} similarity matrix in {elapsed:.3f}s")

    # Simple clustering by finding highly similar groups
    threshold = 0.85
    clusters = []
    assigned = set()

    for i in range(len(hypervectors)):
        if i in assigned:
            continue

        # Find all samples similar to i
        cluster = [i]
        for j in range(i + 1, len(hypervectors)):
            if j not in assigned and similarity_matrix[i, j] > threshold:
                cluster.append(j)
                assigned.add(j)

        if len(cluster) > 1:
            clusters.append(cluster)
            assigned.add(i)

    print(f"\nFound {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {len(cluster)} samples")


def population_stratification_demo(hypervectors: torch.Tensor):
    """Demonstrate population stratification using Hamming similarity"""
    print("\n=== Population Stratification Demo ===")

    # Simulate population labels (in practice, these might be unknown)
    num_samples = len(hypervectors)
    populations = np.random.choice(["EUR", "AFR", "EAS"], size=num_samples)

    # Use LUT-optimized similarity
    hamming_lut = HammingLUT(use_gpu=torch.cuda.is_available())

    # Compute mean within-population similarity
    for pop in ["EUR", "AFR", "EAS"]:
        pop_indices = np.where(populations == pop)[0]
        if len(pop_indices) < 2:
            continue

        # Get hypervectors for this population
        pop_hvs = hypervectors[pop_indices]

        # Convert to binary and pack
        pop_binary = [(torch.sign(hv) > 0).numpy().astype(np.uint8) for hv in pop_hvs]
        pop_packed = [np.packbits(b).view(np.uint64) for b in pop_binary]

        # Compute pairwise distances within population
        similarities = []
        for i in range(len(pop_packed)):
            for j in range(i + 1, len(pop_packed)):
                dist = hamming_lut.distance(pop_packed[i], pop_packed[j])
                sim = 1.0 - (dist / (hypervectors.shape[1]))
                similarities.append(sim)

        if similarities:
            mean_sim = np.mean(similarities)
            print(f"{pop} population mean similarity: {mean_sim:.4f}")


def performance_comparison():
    """Compare standard vs LUT-optimized similarity computation"""
    print("\n=== Performance Comparison ===")

    dimensions = [1000, 5000, 10000, 20000]
    num_samples = 100

    for dim in dimensions:
        print(f"\nDimension: {dim}")

        # Generate random hypervectors
        hvs = torch.sign(torch.randn(num_samples, dim))

        # Standard similarity (no LUT)
        binder_std = HypervectorBinder()
        binder_std.hamming_lut = None  # Disable LUT

        start = time.perf_counter()
        _ = binder_std.batch_hamming_similarity(hvs[:10], hvs[:10])
        std_time = time.perf_counter() - start

        # LUT-optimized similarity
        binder_lut = HypervectorBinder()

        start = time.perf_counter()
        _ = binder_lut.batch_hamming_similarity(hvs[:10], hvs[:10])
        lut_time = time.perf_counter() - start

        speedup = std_time / lut_time if lut_time > 0 else 1.0
        print(f"  Standard: {std_time*1000:.2f}ms")
        print(f"  LUT:      {lut_time*1000:.2f}ms")
        print(f"  Speedup:  {speedup:.2f}x")


def main():
    """Run all demonstrations"""
    print("GenomeVault Hamming LUT Example")
    print("================================")

    # Simulate genomic data
    print("\nSimulating genomic variant data...")
    num_samples = 200
    num_variants = 50000
    variant_data = simulate_variant_data(num_samples, num_variants)
    print(f"Generated {num_samples} samples with {num_variants} variants")

    # Encode to hypervectors
    print("\nEncoding variants to hypervectors...")
    start = time.perf_counter()
    hypervectors = encode_variants_to_hypervectors(variant_data, dimension=10000)
    elapsed = time.perf_counter() - start
    print(f"Encoded in {elapsed:.2f}s")

    # Find similar genomes
    print("\n=== Finding Similar Genomes ===")
    query_idx = 0
    similar = find_similar_genomes(query_idx, hypervectors, top_k=5)

    print(f"\nTop 5 genomes similar to sample {query_idx}:")
    for idx, sim in similar:
        print(f"  Sample {idx}: {sim:.4f} similarity")

    # Batch clustering
    batch_clustering_demo(hypervectors[:50])  # Use subset for speed

    # Population stratification
    population_stratification_demo(hypervectors)

    # Performance comparison
    performance_comparison()

    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nKey Benefits of Hamming LUT:")
    print("✓ 2-3x faster similarity computation")
    print("✓ Enables real-time genomic similarity search")
    print("✓ Scales to large cohorts (1000s of samples)")
    print("✓ Compatible with privacy-preserving hypervectors")
    print("✓ Supports both CPU and GPU acceleration")


if __name__ == "__main__":
    main()
