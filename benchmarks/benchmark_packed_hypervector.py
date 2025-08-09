"""
Benchmark script comparing packed vs standard hypervector implementations
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from memory_profiler import profile

from genomevault.hypervector.encoding import GenomicEncoder, PackedGenomicEncoder


def generate_random_variants(n_variants: int = 1000) -> list[dict[str, any]]:
    """Generate random genomic variants for testing"""
    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    bases = ["A", "T", "G", "C"]

    variants = []
    for _ in range(n_variants):
        chr_idx = np.random.randint(len(chromosomes))
        position = np.random.randint(1, 250_000_000)  # Human genome size
        ref_idx = np.random.randint(4)
        alt_idx = (ref_idx + np.random.randint(1, 4)) % 4  # Different from ref

        variants.append(
            {
                "chromosome": chromosomes[chr_idx],
                "position": position,
                "ref": bases[ref_idx],
                "alt": bases[alt_idx],
                "type": "SNP",
            }
        )

    return variants


def benchmark_encoding(
    encoder: any, variants: list[dict[str, any]], name: str = "Encoder"
) -> tuple[any, float, float, int]:
    """Benchmark encoding performance"""
    print(f"\n{name} Benchmark:")
    print("-" * 50)

    # Single variant encoding
    start = time.time()
    for _ in range(100):
        _ = encoder.encode_variant(chromosome="chr1", position=12345, ref="A", alt="G")
    single_time = (time.time() - start) / 100
    print(f"Single variant encoding: {single_time * 1000:.3f} ms")

    # Batch encoding
    start = time.time()
    genome_hv = encoder.encode_genome(variants)
    batch_time = time.time() - start
    print(f"Batch encoding ({len(variants)} variants): {batch_time:.3f} s")
    print(f"Average per variant: {batch_time / len(variants) * 1000:.3f} ms")

    # Memory usage
    if hasattr(genome_hv, "memory_bytes"):
        memory = genome_hv.memory_bytes
    else:
        memory = genome_hv.element_size() * genome_hv.nelement()
    print(f"Memory usage: {memory / 1024:.2f} KB")

    return genome_hv, single_time, batch_time, memory


def benchmark_similarity(encoder: any, hv1: any, hv2: any, name: str = "Encoder") -> float:
    """Benchmark similarity computation"""
    start = time.time()
    for _ in range(1000):
        _ = encoder.similarity(hv1, hv2)
    sim_time = (time.time() - start) / 1000
    print(f"{name} - Similarity computation: {sim_time * 1000:.3f} ms")
    return sim_time


def plot_results(results: dict[str, dict[str, float]]) -> None:
    """Plot benchmark results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Encoding time comparison
    ax = axes[0, 0]
    encoders = list(results.keys())
    single_times = [results[e]["single_time"] * 1000 for e in encoders]
    ax.bar(encoders, single_times)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Single Variant Encoding Time")

    # Batch encoding time
    ax = axes[0, 1]
    batch_times = [results[e]["batch_time"] for e in encoders]
    ax.bar(encoders, batch_times)
    ax.set_ylabel("Time (s)")
    ax.set_title("Batch Encoding Time (1000 variants)")

    # Memory usage
    ax = axes[1, 0]
    memory_usage = [results[e]["memory"] / 1024 for e in encoders]
    ax.bar(encoders, memory_usage)
    ax.set_ylabel("Memory (KB)")
    ax.set_title("Memory Usage")

    # Similarity computation time
    ax = axes[1, 1]
    sim_times = [results[e]["sim_time"] * 1000 for e in encoders]
    ax.bar(encoders, sim_times)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Similarity Computation Time")

    plt.tight_layout()
    plt.savefig("packed_hypervector_benchmark.png", dpi=300)
    print("\nBenchmark plot saved as 'packed_hypervector_benchmark.png'")


@profile
def memory_profile_test() -> None:
    """Profile memory usage"""
    dimension = 10000
    variants = generate_random_variants(1000)

    print("\n=== Memory Profile: Standard Encoder ===")
    standard_encoder = GenomicEncoder(dimension=dimension)
    standard_hv = standard_encoder.encode_genome(variants)

    print("\n=== Memory Profile: Packed Encoder ===")
    packed_encoder = PackedGenomicEncoder(dimension=dimension, packed=True)
    packed_hv = packed_encoder.encode_genome(variants)

    print(f"\nStandard memory: {standard_hv.element_size() * standard_hv.nelement() / 1024:.2f} KB")
    print(f"Packed memory: {packed_hv.memory_bytes / 1024:.2f} KB")


def main() -> None:
    """Run comprehensive benchmarks"""
    print("Hypervector Encoding Benchmark")
    print("=" * 50)

    # Test parameters
    dimension = 10000
    n_variants = 1000

    # Generate test data
    print(f"\nGenerating {n_variants} random variants...")
    variants = generate_random_variants(n_variants)

    # Initialize encoders
    standard_encoder = GenomicEncoder(dimension=dimension)
    packed_encoder = PackedGenomicEncoder(dimension=dimension, packed=True)

    # Run benchmarks
    results = {}

    # Standard encoder
    std_hv, std_single, std_batch, std_memory = benchmark_encoding(
        standard_encoder, variants, "Standard Encoder"
    )

    # Packed encoder
    packed_hv, packed_single, packed_batch, packed_memory = benchmark_encoding(
        packed_encoder, variants, "Packed Encoder"
    )

    # Similarity benchmarks
    print("\n\nSimilarity Computation Benchmark:")
    print("-" * 50)

    # Generate second genome for similarity
    variants2 = generate_random_variants(n_variants)
    std_hv2 = standard_encoder.encode_genome(variants2)
    packed_hv2 = packed_encoder.encode_genome(variants2)

    std_sim_time = benchmark_similarity(standard_encoder, std_hv, std_hv2, "Standard")
    packed_sim_time = benchmark_similarity(packed_encoder, packed_hv, packed_hv2, "Packed")

    # Store results
    results["Standard"] = {
        "single_time": std_single,
        "batch_time": std_batch,
        "memory": std_memory,
        "sim_time": std_sim_time,
    }

    results["Packed"] = {
        "single_time": packed_single,
        "batch_time": packed_batch,
        "memory": packed_memory,
        "sim_time": packed_sim_time,
    }

    # Summary
    print("\n\nSummary:")
    print("=" * 50)
    print(f"Encoding speedup: {std_batch / packed_batch:.2f}x")
    print(f"Memory reduction: {std_memory / packed_memory:.2f}x")
    print(f"Similarity speedup: {std_sim_time / packed_sim_time:.2f}x")

    # Accuracy check
    print("\n\nAccuracy Verification:")
    print("-" * 50)

    # Convert packed to torch for comparison
    packed_torch = packed_hv.to_torch()
    std_normalized = std_hv / torch.norm(std_hv)
    packed_normalized = packed_torch / torch.norm(packed_torch)

    # Check that encodings are similar (not identical due to different methods)
    similarity = torch.cosine_similarity(std_normalized, packed_normalized, dim=0).item()
    print(f"Encoding similarity between methods: {similarity:.4f}")

    # Plot results
    try:
        plot_results(results)
    except ImportError:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        print("\nMatplotlib not available for plotting")
        raise

    # Memory profile
    print("\n\nRunning memory profile...")
    memory_profile_test()


if __name__ == "__main__":
    main()
