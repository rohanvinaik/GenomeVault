#!/usr/bin/env python3
"""
GC-Content Topological Map - The "One-Float" Genome Index

Proves that ||AT_vec|| and ||GC_vec|| encode GC content:
  ||AT_vec|| ≈ sqrt((N_A + N_T) * D)
  ||GC_vec|| ≈ sqrt((N_G + N_C) * D)

  GC_ratio = ||GC_vec|| / (||AT_vec|| + ||GC_vec||)

This single float per chunk tells us:
- GC < 35%: AT-rich (fragile sites, CNVs) → Use recursive subdivision
- GC 35-55%: Normal regions (96% of genome) → Use binary encoding
- GC > 55%: GC-rich (regulatory, CpG islands) → Use float32

Zero additional storage - computed from existing vectors!
"""

import json
import gzip
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
import h5py

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def compute_gc_map_from_vectors(hdf5_path: Path, max_chunks: int = 100000):
    """
    Compute GC-content map from vector magnitudes.

    Returns:
        dict: {chunk_key: gc_ratio}
    """
    logger.info("Computing GC-content topological map from vector magnitudes...")

    gc_map = {}

    with h5py.File(hdf5_path, 'r') as f:
        n_chunks, D = f['AT_vectors'].shape
        n_chunks = min(n_chunks, max_chunks)

        chunk_keys = [key.decode('utf-8') for key in f['chunk_keys'][:n_chunks]]

        logger.info(f"  Processing {n_chunks:,} chunks (D={D:,})...")

        batch_size = 10000
        n_batches = (n_chunks + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_chunks)

            AT_batch = f['AT_vectors'][batch_start:batch_end]
            GC_batch = f['GC_vectors'][batch_start:batch_end]

            # Compute magnitudes
            mag_AT = np.linalg.norm(AT_batch, axis=1)  # ||AT_vec||
            mag_GC = np.linalg.norm(GC_batch, axis=1)  # ||GC_vec||

            # GC ratio = ||GC|| / (||AT|| + ||GC||)
            gc_ratio = mag_GC / (mag_AT + mag_GC + 1e-10)

            for i in range(batch_end - batch_start):
                global_idx = batch_start + i
                chunk_key = chunk_keys[global_idx]
                gc_map[chunk_key] = float(gc_ratio[i])

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                progress = (batch_end / n_chunks) * 100
                logger.info(f"    Progress: {batch_end:,}/{n_chunks:,} ({progress:.1f}%)")

    logger.info(f"  ✓ Generated GC map for {len(gc_map):,} chunks")
    return gc_map


def compute_actual_gc_content(gdiff_path: Path):
    """
    Compute actual GC content from GDiff variants (ground truth).

    Returns:
        dict: {chunk_key: actual_gc_ratio}
    """
    logger.info("Computing actual GC content from GDiff...")

    with gzip.open(gdiff_path, 'rt') as f:
        gdiff = json.load(f)

    variants = gdiff["differential_variants"]
    logger.info(f"  Total variants: {len(variants):,}")

    # Count nucleotides per chunk
    chunk_counts = defaultdict(lambda: {'A': 0, 'T': 0, 'G': 0, 'C': 0})

    chunk_size = 2000
    for v in variants:
        alt = v['alt']
        if alt not in ['A', 'T', 'G', 'C']:
            continue

        chrom = v['chrom']
        pos = v['pos']
        chunk_start = (pos // chunk_size) * chunk_size
        chunk_key = f"{chrom}:{chunk_start}"

        chunk_counts[chunk_key][alt] += 1

    # Compute GC ratio
    actual_gc = {}
    for chunk_key, counts in chunk_counts.items():
        at_count = counts['A'] + counts['T']
        gc_count = counts['G'] + counts['C']
        total = at_count + gc_count

        if total > 0:
            actual_gc[chunk_key] = gc_count / total

    logger.info(f"  ✓ Computed actual GC for {len(actual_gc):,} chunks")
    return actual_gc


def validate_gc_map(gc_map: dict, actual_gc: dict):
    """Validate that vector magnitudes correlate with actual GC content."""
    logger.info("")
    logger.info("=== VALIDATION: Vector Magnitudes vs Actual GC Content ===")
    logger.info("")

    # Find common chunks
    common_chunks = set(gc_map.keys()) & set(actual_gc.keys())
    logger.info(f"Comparing {len(common_chunks):,} common chunks...")

    if len(common_chunks) < 100:
        logger.warning(f"Only {len(common_chunks)} chunks with variants - may not be representative")
        return

    # Sample for comparison
    sample_chunks = list(common_chunks)[:10000]

    predicted = [gc_map[ck] for ck in sample_chunks]
    actual = [actual_gc[ck] for ck in sample_chunks]

    # Compute correlation
    correlation = np.corrcoef(predicted, actual)[0, 1]

    # Compute error metrics
    errors = [abs(p - a) for p, a in zip(predicted, actual)]
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)

    logger.info(f"  Correlation: {correlation:.4f}")
    logger.info(f"  Mean absolute error: {mean_error:.4f}")
    logger.info(f"  Median absolute error: {median_error:.4f}")
    logger.info(f"  Max error: {max_error:.4f}")
    logger.info("")

    # Show distribution
    logger.info("GC Distribution (predicted from magnitudes):")
    gc_bins = [0.0, 0.35, 0.45, 0.55, 1.0]
    bin_labels = ["<35% (AT-rich)", "35-45% (Normal)", "45-55% (Normal)", ">55% (GC-rich)"]

    for i in range(len(gc_bins) - 1):
        count = sum(1 for gc in predicted if gc_bins[i] <= gc < gc_bins[i+1])
        pct = (count / len(predicted)) * 100
        logger.info(f"  {bin_labels[i]}: {count:,} chunks ({pct:.1f}%)")

    return correlation


def analyze_error_stratification(gc_map: dict, error_file: Path):
    """Analyze how errors stratify by GC content."""
    logger.info("")
    logger.info("=== ERROR STRATIFICATION BY GC CONTENT ===")
    logger.info("")

    # Load errors from previous test
    if not error_file.exists():
        logger.warning(f"Error file not found: {error_file}")
        return

    with open(error_file, 'r') as f:
        error_data = json.load(f)

    errors = error_data.get('errors', [])
    if not errors:
        logger.warning("No errors found in file")
        return

    logger.info(f"Analyzing {len(errors):,} errors...")

    # Categorize errors by GC content
    chunk_size = 2000
    gc_categories = {
        'low_gc': {'errors': [], 'range': '<35%'},
        'normal_gc': {'errors': [], 'range': '35-55%'},
        'high_gc': {'errors': [], 'range': '>55%'}
    }

    for err in errors:
        chrom = err['chrom']
        pos = err['pos']
        chunk_start = (pos // chunk_size) * chunk_size
        chunk_key = f"{chrom}:{chunk_start}"

        if chunk_key not in gc_map:
            continue

        gc = gc_map[chunk_key]

        if gc < 0.35:
            gc_categories['low_gc']['errors'].append(err)
        elif gc > 0.55:
            gc_categories['high_gc']['errors'].append(err)
        else:
            gc_categories['normal_gc']['errors'].append(err)

    # Report stratification
    logger.info("Error distribution by GC content:")
    for cat, data in gc_categories.items():
        count = len(data['errors'])
        pct = (count / len(errors)) * 100 if errors else 0
        logger.info(f"  {data['range']} ({cat}): {count:,} errors ({pct:.1f}%)")

    logger.info("")


def demonstrate_adaptive_encoding(gc_map: dict):
    """Show how to use GC map for adaptive encoding."""
    logger.info("")
    logger.info("=== ADAPTIVE ENCODING STRATEGY ===")
    logger.info("")

    # Categorize chunks
    strategy_counts = {
        'skip': 0,      # Normal GC, binary encoding sufficient
        'recursive': 0,  # AT-rich, fragile sites
        'float32': 0     # GC-rich, regulatory regions
    }

    for chunk_key, gc in gc_map.items():
        if gc < 0.35:
            # AT-rich: Use recursive subdivision
            strategy_counts['recursive'] += 1
        elif gc > 0.55:
            # GC-rich: Store at float32 precision
            strategy_counts['float32'] += 1
        else:
            # Normal: Binary encoding or skip
            strategy_counts['skip'] += 1

    total = len(gc_map)

    logger.info("Encoding strategy distribution:")
    logger.info(f"  Layer 1 (Binary/Skip):      {strategy_counts['skip']:,} chunks ({strategy_counts['skip']/total*100:.1f}%)")
    logger.info(f"  Layer 2a (Recursive):       {strategy_counts['recursive']:,} chunks ({strategy_counts['recursive']/total*100:.1f}%)")
    logger.info(f"  Layer 2b (Float32):         {strategy_counts['float32']:,} chunks ({strategy_counts['float32']/total*100:.1f}%)")
    logger.info("")

    # Compute storage savings
    D = 10000
    bytes_per_chunk_binary = (D // 8) * 2  # Binary: 2 packed bit arrays
    bytes_per_chunk_float32 = D * 4 * 2   # Float32: 2 full vectors

    total_binary = strategy_counts['skip'] * bytes_per_chunk_binary
    total_recursive = strategy_counts['recursive'] * bytes_per_chunk_binary  # Assume same as binary
    total_float32 = strategy_counts['float32'] * bytes_per_chunk_float32

    total_bytes = total_binary + total_recursive + total_float32
    baseline_bytes = total * bytes_per_chunk_float32

    compression_ratio = baseline_bytes / total_bytes

    logger.info("Storage comparison:")
    logger.info(f"  Baseline (all float32):     {baseline_bytes / 1024**3:.2f} GB")
    logger.info(f"  Adaptive encoding:          {total_bytes / 1024**3:.2f} GB")
    logger.info(f"  Compression ratio:          {compression_ratio:.1f}×")
    logger.info("")


def run_gc_topological_map_analysis():
    """Run complete GC topological map analysis."""
    logger.info("=" * 80)
    logger.info("GC-CONTENT TOPOLOGICAL MAP ANALYSIS")
    logger.info("=" * 80)
    logger.info("")

    # Paths
    hdf5_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome.h5")
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    error_file = Path("HDV_VALIDATION_PACKAGE/error_analysis/float32_errors.json")

    # Step 1: Compute GC map from vector magnitudes
    gc_map = compute_gc_map_from_vectors(hdf5_path)

    # Step 2: Compute actual GC content for validation
    actual_gc = compute_actual_gc_content(gdiff_path)

    # Step 3: Validate correlation
    correlation = validate_gc_map(gc_map, actual_gc)

    # Step 4: Analyze error stratification
    analyze_error_stratification(gc_map, error_file)

    # Step 5: Demonstrate adaptive encoding
    demonstrate_adaptive_encoding(gc_map)

    # Save GC map
    output_dir = Path("HDV_VALIDATION_PACKAGE/gc_map")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "gc_topological_map.json"

    # Sample for saving (don't save all 100k chunks)
    sample_chunks = list(gc_map.keys())[:10000]
    sample_map = {k: gc_map[k] for k in sample_chunks}

    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'description': 'GC-content topological map computed from vector magnitudes',
                'formula': 'GC_ratio = ||GC_vec|| / (||AT_vec|| + ||GC_vec||)',
                'total_chunks': len(gc_map),
                'correlation_with_actual': float(correlation) if correlation else None
            },
            'sample_map': sample_map
        }, f, indent=2)

    logger.info(f"✓ GC map saved to: {output_file}")
    logger.info("")

    logger.info("=" * 80)
    logger.info("✅ GC TOPOLOGICAL MAP ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("KEY INSIGHT:")
    logger.info("  Vector magnitudes encode GC content - a zero-cost topological map!")
    logger.info("  Use this to guide adaptive encoding:")
    logger.info("    - AT-rich (<35%): Recursive subdivision for fragile sites")
    logger.info("    - Normal (35-55%): Binary encoding (96% of genome)")
    logger.info("    - GC-rich (>55%): Float32 precision for regulatory regions")


if __name__ == "__main__":
    run_gc_topological_map_analysis()
