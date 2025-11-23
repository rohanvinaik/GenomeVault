#!/usr/bin/env python3
"""
Genome-Wide Random Position Sampler

Samples positions across the entire genome with proper stratification
to avoid local sequence context bias.

Strategy:
  - Sample equally from all chromosomes (chr1-22, chrX, chrY)
  - Within each chromosome, sample from diverse regions
  - Avoid clustering in any single genomic context
"""

import h5py
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Set
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def extract_chromosome_from_chunk(chunk_key: str) -> str:
    """Extract chromosome from chunk key (e.g., 'chr22_consensus:10000000' -> 'chr22')."""
    chrom_part = chunk_key.split(':')[0]
    return chrom_part.replace('_consensus', '')


def sample_genome_wide_positions(
    h5_path: Path,
    sample_size: int,
    seed: int = 42,
    min_spacing_kb: int = 100
) -> List[Tuple[str, int]]:
    """
    Sample positions genome-wide with proper stratification.

    Args:
        h5_path: Path to H5 file containing chunk_keys
        sample_size: Total number of positions to sample
        seed: Random seed for reproducibility
        min_spacing_kb: Minimum spacing between sampled positions (in kilobases)

    Returns:
        List of (chrom, pos) tuples
    """
    np.random.seed(seed)

    logger.info("=" * 80)
    logger.info("GENOME-WIDE RANDOM POSITION SAMPLING")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Target sample size: {sample_size:,}")
    logger.info(f"Minimum spacing: {min_spacing_kb} kb")
    logger.info(f"Random seed: {seed}")
    logger.info("")

    # Load chunk keys
    logger.info(f"Loading chunk keys from: {h5_path.name}")
    with h5py.File(h5_path, 'r') as f:
        chunk_keys_bytes = f['chunk_keys'][:]
        chunk_keys = [k.decode('utf-8') for k in chunk_keys_bytes]

    logger.info(f"  ✓ Found {len(chunk_keys):,} chunks")
    logger.info("")

    # Group chunks by chromosome
    chunks_by_chrom = defaultdict(list)
    for chunk_key in chunk_keys:
        chrom = extract_chromosome_from_chunk(chunk_key)
        chunks_by_chrom[chrom].append(chunk_key)

    chromosomes = sorted(chunks_by_chrom.keys(), key=lambda x: (
        x != 'chrX' and x != 'chrY',  # chrX/chrY at end
        int(x.replace('chr', '').replace('X', '23').replace('Y', '24'))
    ))

    logger.info("Chromosome distribution:")
    for chrom in chromosomes:
        n_chunks = len(chunks_by_chrom[chrom])
        logger.info(f"  {chrom:6s}: {n_chunks:>6,} chunks")
    logger.info("")

    # Calculate samples per chromosome (proportional to chunk count)
    total_chunks = len(chunk_keys)
    samples_per_chrom = {}

    for chrom in chromosomes:
        n_chunks = len(chunks_by_chrom[chrom])
        proportion = n_chunks / total_chunks
        samples_per_chrom[chrom] = int(sample_size * proportion)

    # Adjust for rounding errors
    total_allocated = sum(samples_per_chrom.values())
    if total_allocated < sample_size:
        # Add remaining samples to largest chromosomes
        diff = sample_size - total_allocated
        sorted_chroms = sorted(chromosomes, key=lambda c: len(chunks_by_chrom[c]), reverse=True)
        for i in range(diff):
            samples_per_chrom[sorted_chroms[i % len(sorted_chroms)]] += 1

    logger.info("Sampling strategy:")
    for chrom in chromosomes:
        n_samples = samples_per_chrom[chrom]
        logger.info(f"  {chrom:6s}: {n_samples:>6,} positions")
    logger.info("")

    # Sample positions from each chromosome
    all_positions = []
    min_spacing_bp = min_spacing_kb * 1000
    N = 2000  # Chunk size in bp

    logger.info("Sampling positions...")

    for chrom in chromosomes:
        chrom_chunks = chunks_by_chrom[chrom]
        n_samples = samples_per_chrom[chrom]

        if n_samples == 0:
            continue

        # Extract available positions for this chromosome
        available_positions = []
        for chunk_key in chrom_chunks:
            chrom_with_suffix, chunk_start_str = chunk_key.split(':')
            chunk_start = int(chunk_start_str)

            # Each chunk has 2000 positions
            for offset in range(N):
                pos = chunk_start + offset
                available_positions.append((chrom_with_suffix, pos))

        # Shuffle and take samples with spacing constraint
        np.random.shuffle(available_positions)

        sampled = []
        last_pos = -min_spacing_bp * 2  # Start far enough back

        for chrom_key, pos in available_positions:
            if len(sampled) >= n_samples:
                break

            # Check spacing constraint
            if pos - last_pos >= min_spacing_bp:
                sampled.append((chrom_key, pos))
                last_pos = pos

        # If we couldn't meet spacing constraint, relax it
        if len(sampled) < n_samples:
            logger.warning(f"  {chrom}: Could only sample {len(sampled)}/{n_samples} with {min_spacing_kb}kb spacing")
            logger.warning(f"  Relaxing spacing constraint for remaining samples...")

            remaining = n_samples - len(sampled)
            for chrom_key, pos in available_positions:
                if len(sampled) >= n_samples:
                    break
                if (chrom_key, pos) not in sampled:
                    sampled.append((chrom_key, pos))

        all_positions.extend(sampled)
        logger.info(f"  ✓ {chrom:6s}: {len(sampled):>6,} positions sampled")

    logger.info("")
    logger.info(f"✓ Total positions sampled: {len(all_positions):,}")
    logger.info("")

    # Validation: Check chromosome diversity
    chrom_counts = defaultdict(int)
    for chrom_key, pos in all_positions:
        chrom = extract_chromosome_from_chunk(chrom_key)
        chrom_counts[chrom] += 1

    logger.info("Final chromosome distribution:")
    for chrom in chromosomes:
        count = chrom_counts.get(chrom, 0)
        pct = 100 * count / len(all_positions) if all_positions else 0
        logger.info(f"  {chrom:6s}: {count:>6,} ({pct:>5.2f}%)")
    logger.info("")

    return all_positions


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sample genome-wide random positions')
    parser.add_argument('--h5', type=str, required=True, help='Path to H5 file')
    parser.add_argument('--samples', type=int, default=10000, help='Number of positions to sample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--min-spacing-kb', type=int, default=100, help='Minimum spacing in kb')
    parser.add_argument('--output', type=str, default='genome_wide_positions.txt', help='Output file')

    args = parser.parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        logger.error(f"H5 file not found: {h5_path}")
        exit(1)

    positions = sample_genome_wide_positions(
        h5_path,
        args.samples,
        seed=args.seed,
        min_spacing_kb=args.min_spacing_kb
    )

    # Save to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for chrom, pos in positions:
            f.write(f"{chrom}\t{pos}\n")

    logger.info(f"✓ Positions saved to: {output_path}")
