#!/usr/bin/env python3
"""
Build minimal metadata index for Stage 1 filtering.

WEEK 1 IMPLEMENTATION - Multi-Stage Query Architecture

Purpose:
    Build 21 MB metadata index (64 bytes per chunk) with:
        - Genomic positions (chr, start, end)
        - Top-5 k-mer hashes (MurmurHash3)

Performance Targets:
    - Storage: 21 MB (not 3-6 GB!)
    - Filtering speed: <0.1 μs per chunk
    - Genome reduction: 40-60%

Usage:
    python build_metadata_index.py \\
        --genome-fasta data/reference_genomes/hg38_chr22.fa.gz \\
        --output-path output/metadata_index_chr22.h5 \\
        --chunk-size 1024 \\
        --stride 896

Author: Research Team
Date: November 22, 2025
"""

import h5py
import numpy as np
import mmh3
from collections import Counter
import gzip
import time
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MinimalMetadataIndexBuilder:
    """
    Build minimal metadata index for fast Stage 1 filtering.

    Storage: 64 bytes per chunk
        - chunk_id (uint32): 4 bytes
        - chr (uint8): 1 byte
        - start (uint32): 4 bytes
        - end (uint32): 4 bytes
        - kmer_hashes (5 × uint64): 40 bytes
        - padding: 11 bytes (for alignment)

    Total: 327,655 chunks × 64 bytes = 21 MB
    """

    def __init__(self, chunk_size=1024, stride=896, kmer_k=5, top_n_kmers=5):
        """
        Initialize metadata index builder.

        Args:
            chunk_size: Size of genomic chunks (default 1024bp)
            stride: Step size between chunks (default 896bp, 128bp overlap)
            kmer_k: K-mer size for hashing (default 5)
            top_n_kmers: Number of top k-mers to store (default 5)
        """
        self.chunk_size = chunk_size
        self.stride = stride
        self.kmer_k = kmer_k
        self.top_n_kmers = top_n_kmers

    def load_genome_sequence(self, fasta_path):
        """
        Load genome sequence from FASTA file.

        Args:
            fasta_path: Path to gzipped FASTA file

        Returns:
            str: Genome sequence (uppercase)
        """
        logger.info(f"Loading genome sequence from {fasta_path}...")

        with gzip.open(fasta_path, 'rt') as f:
            lines = f.readlines()

        # Skip header lines and concatenate sequence
        sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
        sequence = sequence.upper()

        logger.info(f"Genome length: {len(sequence):,} bp")

        return sequence

    def compute_kmer_hashes(self, sequence):
        """
        Compute top-n k-mer hashes using MurmurHash3.

        Args:
            sequence: Genomic sequence (string)

        Returns:
            np.ndarray: Array of uint64 hashes (length top_n_kmers)
        """
        # Extract all k-mers (skip windows with N)
        kmers = []
        for i in range(len(sequence) - self.kmer_k + 1):
            kmer = sequence[i:i+self.kmer_k]
            if 'N' not in kmer:
                kmers.append(kmer)

        if not kmers:
            # No valid k-mers (all N's) - return zeros
            return np.zeros(self.top_n_kmers, dtype=np.uint64)

        # Count k-mer frequencies
        kmer_counts = Counter(kmers)

        # Get top-n k-mers
        top_kmers = kmer_counts.most_common(self.top_n_kmers)

        # Hash each k-mer using MurmurHash3
        hashes = []
        for kmer, _ in top_kmers:
            # MurmurHash3 64-bit hash (seed=42 for reproducibility)
            hash_val = mmh3.hash64(kmer, seed=42)[0]
            hashes.append(hash_val)

        # Pad with zeros if fewer than top_n k-mers
        while len(hashes) < self.top_n_kmers:
            hashes.append(0)

        return np.array(hashes, dtype=np.uint64)

    def build_index(self, genome_sequence, chr_id=22):
        """
        Build metadata index for all chunks in genome.

        Args:
            genome_sequence: Full genome sequence (string)
            chr_id: Chromosome ID (default 22 for chr22)

        Returns:
            dict: Metadata arrays
                - chunk_id (uint32)
                - chr (uint8)
                - start (uint32)
                - end (uint32)
                - kmer_hashes (n_chunks × top_n_kmers, uint64)
        """
        # Calculate number of chunks
        n_chunks = (len(genome_sequence) - self.chunk_size) // self.stride + 1
        logger.info(f"Number of chunks: {n_chunks:,}")

        # Initialize metadata arrays
        metadata = {
            'chunk_id': np.zeros(n_chunks, dtype=np.uint32),
            'chr': np.full(n_chunks, chr_id, dtype=np.uint8),
            'start': np.zeros(n_chunks, dtype=np.uint32),
            'end': np.zeros(n_chunks, dtype=np.uint32),
            'kmer_hashes': np.zeros((n_chunks, self.top_n_kmers), dtype=np.uint64),
        }

        # Process each chunk
        logger.info("Computing metadata for all chunks...")
        t0 = time.perf_counter()

        for chunk_idx in range(n_chunks):
            if chunk_idx % 10000 == 0 and chunk_idx > 0:
                elapsed = time.perf_counter() - t0
                rate = chunk_idx / elapsed
                eta = (n_chunks - chunk_idx) / rate
                logger.info(f"  Processing chunk {chunk_idx:,} / {n_chunks:,} ({chunk_idx/n_chunks*100:.1f}%) | "
                           f"Rate: {rate:.0f} chunks/sec | ETA: {eta:.0f}s")

            start = chunk_idx * self.stride
            end = start + self.chunk_size
            chunk_seq = genome_sequence[start:end]

            # Store genomic position
            metadata['chunk_id'][chunk_idx] = chunk_idx
            metadata['start'][chunk_idx] = start
            metadata['end'][chunk_idx] = end

            # Compute k-mer hashes
            kmer_hashes = self.compute_kmer_hashes(chunk_seq)
            metadata['kmer_hashes'][chunk_idx] = kmer_hashes

        t1 = time.perf_counter()
        total_time = t1 - t0
        logger.info(f"✓ Metadata computation complete in {total_time:.1f}s "
                   f"({n_chunks/total_time:.0f} chunks/sec)")

        return metadata

    def save_index(self, metadata, output_path):
        """
        Save metadata index to HDF5 file with compression.

        Args:
            metadata: Dict of metadata arrays
            output_path: Path to output HDF5 file
        """
        logger.info(f"Saving metadata index to {output_path}...")

        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to HDF5 with gzip compression (level 9 for maximum compression)
        with h5py.File(output_path, 'w') as f:
            for key, value in metadata.items():
                f.create_dataset(key, data=value, compression='gzip', compression_opts=9)

            # Store metadata attributes
            f.attrs['chunk_size'] = self.chunk_size
            f.attrs['stride'] = self.stride
            f.attrs['kmer_k'] = self.kmer_k
            f.attrs['top_n_kmers'] = self.top_n_kmers
            f.attrs['n_chunks'] = len(metadata['chunk_id'])

        # Verify storage size
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Metadata index saved: {file_size_mb:.1f} MB")

        # Check against 21 MB target
        if file_size_mb <= 25:  # Allow 20% tolerance
            logger.info(f"  ✓ Storage target met: {file_size_mb:.1f} MB ≤ 25 MB (target: 21 MB)")
        else:
            logger.warning(f"  ⚠ Storage exceeds target: {file_size_mb:.1f} MB > 25 MB (target: 21 MB)")

        return file_size_mb


class MetadataIndexBenchmark:
    """
    Benchmark metadata index performance.

    Tests:
        1. Filtering speed (<0.1 μs per chunk)
        2. Genome reduction (40-60%)
        3. K-mer hash collision rate (<5%)
    """

    def __init__(self, metadata_path):
        """
        Initialize benchmark with metadata index.

        Args:
            metadata_path: Path to HDF5 metadata index
        """
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """Load metadata from HDF5 file."""
        logger.info(f"Loading metadata index from {self.metadata_path}...")

        with h5py.File(self.metadata_path, 'r') as f:
            metadata = {
                'chunk_id': f['chunk_id'][:],
                'chr': f['chr'][:],
                'start': f['start'][:],
                'end': f['end'][:],
                'kmer_hashes': f['kmer_hashes'][:],
            }
            attrs = dict(f.attrs)

        logger.info(f"✓ Loaded {len(metadata['chunk_id']):,} chunks")
        logger.info(f"  Chunk size: {attrs['chunk_size']} bp")
        logger.info(f"  Stride: {attrs['stride']} bp")
        logger.info(f"  K-mer k: {attrs['kmer_k']}")

        return metadata

    def benchmark_filtering_speed(self, n_queries=100):
        """
        Benchmark metadata filtering speed.

        Target: <0.1 μs per chunk

        Args:
            n_queries: Number of random queries to test

        Returns:
            dict: Benchmark results
        """
        logger.info(f"\n=== Benchmark 1: Filtering Speed ===")
        logger.info(f"Testing {n_queries} random k-mer hash queries...")

        # Generate random k-mer hash queries
        np.random.seed(42)
        query_hashes = [
            np.random.choice(self.metadata['kmer_hashes'].flatten(), 5, replace=False)
            for _ in range(n_queries)
        ]

        # Benchmark filtering
        filtering_times = []
        for query_hash_set in query_hashes:
            t0 = time.perf_counter()

            # Fast O(1) lookup per chunk using set intersection
            query_set = set(query_hash_set)
            candidates = []
            for chunk_idx in range(len(self.metadata['chunk_id'])):
                chunk_hash_set = set(self.metadata['kmer_hashes'][chunk_idx])
                if query_set & chunk_hash_set:  # Non-empty intersection
                    candidates.append(chunk_idx)

            t1 = time.perf_counter()
            filtering_time_us = (t1 - t0) * 1e6
            filtering_times.append(filtering_time_us)

        # Summary statistics
        avg_time = np.mean(filtering_times)
        median_time = np.median(filtering_times)
        p95_time = np.percentile(filtering_times, 95)

        logger.info(f"Filtering time per query:")
        logger.info(f"  Average: {avg_time:.2f} μs")
        logger.info(f"  Median: {median_time:.2f} μs")
        logger.info(f"  95th percentile: {p95_time:.2f} μs")

        # Per-chunk filtering time
        n_chunks = len(self.metadata['chunk_id'])
        per_chunk_time = avg_time / n_chunks
        logger.info(f"Per-chunk filtering time: {per_chunk_time:.4f} μs")

        # Check against target
        if per_chunk_time < 0.1:
            logger.info(f"  ✓ Target met: {per_chunk_time:.4f} μs < 0.1 μs")
        else:
            logger.warning(f"  ⚠ Target missed: {per_chunk_time:.4f} μs > 0.1 μs")

        return {
            'avg_time_us': avg_time,
            'median_time_us': median_time,
            'p95_time_us': p95_time,
            'per_chunk_time_us': per_chunk_time,
            'target_met': per_chunk_time < 0.1,
        }

    def benchmark_genome_reduction(self, n_queries=100):
        """
        Benchmark genome reduction from k-mer hash filtering.

        Target: 40-60% reduction

        Args:
            n_queries: Number of random queries to test

        Returns:
            dict: Benchmark results
        """
        logger.info(f"\n=== Benchmark 2: Genome Reduction ===")
        logger.info(f"Testing {n_queries} random k-mer hash queries...")

        # Generate random k-mer hash queries
        np.random.seed(42)
        query_hashes = [
            np.random.choice(self.metadata['kmer_hashes'].flatten(), 5, replace=False)
            for _ in range(n_queries)
        ]

        # Benchmark genome reduction
        reductions = []
        for query_hash_set in query_hashes:
            query_set = set(query_hash_set)
            candidates = []
            for chunk_idx in range(len(self.metadata['chunk_id'])):
                chunk_hash_set = set(self.metadata['kmer_hashes'][chunk_idx])
                if query_set & chunk_hash_set:
                    candidates.append(chunk_idx)

            reduction_pct = (len(self.metadata['chunk_id']) - len(candidates)) / len(self.metadata['chunk_id']) * 100
            reductions.append(reduction_pct)

        # Summary statistics
        avg_reduction = np.mean(reductions)
        median_reduction = np.median(reductions)

        logger.info(f"Genome reduction:")
        logger.info(f"  Average: {avg_reduction:.1f}%")
        logger.info(f"  Median: {median_reduction:.1f}%")
        logger.info(f"  Range: {np.min(reductions):.1f}% - {np.max(reductions):.1f}%")

        # Check against target (40-60%)
        if 40 <= avg_reduction <= 60:
            logger.info(f"  ✓ Target met: {avg_reduction:.1f}% in [40%, 60%]")
        else:
            logger.warning(f"  ⚠ Target missed: {avg_reduction:.1f}% outside [40%, 60%]")

        return {
            'avg_reduction_pct': avg_reduction,
            'median_reduction_pct': median_reduction,
            'target_met': 40 <= avg_reduction <= 60,
        }

    def benchmark_collision_rate(self):
        """
        Benchmark k-mer hash collision rate.

        Target: <5% collisions

        Returns:
            dict: Benchmark results
        """
        logger.info(f"\n=== Benchmark 3: K-mer Hash Collision Rate ===")

        # Count unique k-mer hashes
        all_hashes = self.metadata['kmer_hashes'].flatten()
        non_zero_hashes = all_hashes[all_hashes != 0]
        unique_hashes = len(set(non_zero_hashes))
        total_hashes = len(non_zero_hashes)

        collision_rate = (total_hashes - unique_hashes) / total_hashes * 100

        logger.info(f"Total k-mer hashes: {total_hashes:,}")
        logger.info(f"Unique k-mer hashes: {unique_hashes:,}")
        logger.info(f"Collision rate: {collision_rate:.2f}%")

        # Check against target
        if collision_rate < 5:
            logger.info(f"  ✓ Target met: {collision_rate:.2f}% < 5%")
        else:
            logger.warning(f"  ⚠ Target missed: {collision_rate:.2f}% > 5%")

        return {
            'total_hashes': total_hashes,
            'unique_hashes': unique_hashes,
            'collision_rate_pct': collision_rate,
            'target_met': collision_rate < 5,
        }

    def run_all_benchmarks(self):
        """
        Run all benchmarks and report summary.

        Returns:
            dict: Combined benchmark results
        """
        logger.info("\n" + "="*80)
        logger.info("METADATA INDEX BENCHMARK SUITE")
        logger.info("="*80)

        results = {}

        # Benchmark 1: Filtering speed
        results['filtering'] = self.benchmark_filtering_speed(n_queries=100)

        # Benchmark 2: Genome reduction
        results['reduction'] = self.benchmark_genome_reduction(n_queries=100)

        # Benchmark 3: Collision rate
        results['collision'] = self.benchmark_collision_rate()

        # Summary
        logger.info("\n" + "="*80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*80)

        all_targets_met = (
            results['filtering']['target_met'] and
            results['reduction']['target_met'] and
            results['collision']['target_met']
        )

        logger.info(f"Filtering speed: {'✓' if results['filtering']['target_met'] else '✗'} "
                   f"{results['filtering']['per_chunk_time_us']:.4f} μs per chunk (target: <0.1 μs)")
        logger.info(f"Genome reduction: {'✓' if results['reduction']['target_met'] else '✗'} "
                   f"{results['reduction']['avg_reduction_pct']:.1f}% (target: 40-60%)")
        logger.info(f"Collision rate: {'✓' if results['collision']['target_met'] else '✗'} "
                   f"{results['collision']['collision_rate_pct']:.2f}% (target: <5%)")

        if all_targets_met:
            logger.info("\n✓ ALL BENCHMARKS PASSED!")
        else:
            logger.warning("\n⚠ Some benchmarks failed - review results above")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Build minimal metadata index for Stage 1 filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build metadata index for chr22
    python build_metadata_index.py \\
        --genome-fasta data/reference_genomes/hg38_chr22.fa.gz \\
        --output-path output/metadata_index_chr22.h5

    # Build and benchmark
    python build_metadata_index.py \\
        --genome-fasta data/reference_genomes/hg38_chr22.fa.gz \\
        --output-path output/metadata_index_chr22.h5 \\
        --benchmark
        """
    )

    parser.add_argument('--genome-fasta', required=True,
                       help='Path to genome FASTA file (gzipped)')
    parser.add_argument('--output-path', required=True,
                       help='Path to output HDF5 file')
    parser.add_argument('--chunk-size', type=int, default=1024,
                       help='Chunk size in bp (default: 1024)')
    parser.add_argument('--stride', type=int, default=896,
                       help='Stride in bp (default: 896, 128bp overlap)')
    parser.add_argument('--kmer-k', type=int, default=5,
                       help='K-mer size for hashing (default: 5)')
    parser.add_argument('--top-n-kmers', type=int, default=5,
                       help='Number of top k-mers to store (default: 5)')
    parser.add_argument('--chr-id', type=int, default=22,
                       help='Chromosome ID (default: 22 for chr22)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark suite after building index')

    args = parser.parse_args()

    # Build metadata index
    builder = MinimalMetadataIndexBuilder(
        chunk_size=args.chunk_size,
        stride=args.stride,
        kmer_k=args.kmer_k,
        top_n_kmers=args.top_n_kmers
    )

    # Load genome
    genome_sequence = builder.load_genome_sequence(args.genome_fasta)

    # Build index
    metadata = builder.build_index(genome_sequence, chr_id=args.chr_id)

    # Save index
    file_size_mb = builder.save_index(metadata, args.output_path)

    # Run benchmarks if requested
    if args.benchmark:
        benchmarker = MetadataIndexBenchmark(args.output_path)
        results = benchmarker.run_all_benchmarks()

        # Save benchmark results
        import json
        benchmark_path = Path(args.output_path).with_suffix('.benchmark.json')
        with open(benchmark_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Benchmark results saved to {benchmark_path}")

    logger.info("\n✓ METADATA INDEX BUILD COMPLETE!")
    logger.info(f"  Output: {args.output_path}")
    logger.info(f"  Size: {file_size_mb:.1f} MB")
    logger.info(f"  Chunks: {len(metadata['chunk_id']):,}")


if __name__ == '__main__':
    main()
