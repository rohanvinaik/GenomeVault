#!/usr/bin/env python3
"""
Float32 Instability Analysis

Investigates why float32 streaming HDV shows pattern reversal between runs
when using identical code and data.
"""

import json
import gzip
import numpy as np
from pathlib import Path
import h5py
from collections import defaultdict

print("=" * 80)
print("FLOAT32 INSTABILITY ROOT CAUSE ANALYSIS")
print("=" * 80)
print()

# Load current run errors
print("Loading current run error data...")
with open('HDV_VALIDATION_PACKAGE/error_analysis/float32_errors.json', 'r') as f:
    current_errors = json.load(f)

print(f"  Current run (5:34 PM): {current_errors['metadata']['accuracy']:.2f}%")
print(f"    AT: {current_errors['metadata']['at_accuracy']:.2f}%")
print(f"    GC: {current_errors['metadata']['gc_accuracy']:.2f}%")
print(f"    Errors: {current_errors['metadata']['error_count']}")
print()

# Load ground truth to analyze error distribution
print("Loading ground truth GDiff...")
gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
with gzip.open(gdiff_path, 'rt') as f:
    gdiff = json.load(f)

variants = gdiff["differential_variants"]
print(f"  Total variants: {len(variants):,}")
print()

# Analyze error patterns by chromosome
print("Analyzing error distribution by chromosome...")
error_by_chrom = defaultdict(lambda: {'AT': 0, 'GC': 0})

for err in current_errors['errors']:
    pair = 'AT' if err['truth'] in ['A', 'T'] else 'GC'
    error_by_chrom[err['chrom']][pair] += 1

print(f"\n{'Chromosome':<20} {'AT Errors':<12} {'GC Errors':<12} {'Ratio (AT/GC)':<15}")
print("-" * 60)
for chrom in sorted(error_by_chrom.keys()):
    at_err = error_by_chrom[chrom]['AT']
    gc_err = error_by_chrom[chrom]['GC']
    ratio = at_err / gc_err if gc_err > 0 else float('inf')
    print(f"{chrom:<20} {at_err:<12} {gc_err:<12} {ratio:<15.2f}")

print()

# Analyze HDF5 magnitude patterns
print("Analyzing HDF5 vector magnitudes...")
hdf5_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome.h5")

with h5py.File(hdf5_path, 'r') as f:
    # Sample 1000 random chunks
    num_chunks = f['AT_vectors'].shape[0]
    sample_indices = np.random.choice(num_chunks, size=min(1000, num_chunks), replace=False)

    at_norms = []
    gc_norms = []

    for idx in sample_indices:
        at_vec = f['AT_vectors'][idx]
        gc_vec = f['GC_vectors'][idx]

        at_norms.append(np.linalg.norm(at_vec))
        gc_norms.append(np.linalg.norm(gc_vec))

    at_norms = np.array(at_norms)
    gc_norms = np.array(gc_norms)

print(f"  AT vector norms (1000 samples):")
print(f"    Mean: {np.mean(at_norms):.2f}")
print(f"    Std:  {np.std(at_norms):.2f}")
print(f"    Min:  {np.min(at_norms):.2f}")
print(f"    Max:  {np.max(at_norms):.2f}")
print()
print(f"  GC vector norms (1000 samples):")
print(f"    Mean: {np.mean(gc_norms):.2f}")
print(f"    Std:  {np.std(gc_norms):.2f}")
print(f"    Min:  {np.min(gc_norms):.2f}")
print(f"    Max:  {np.max(gc_norms):.2f}")
print()
print(f"  GC/AT magnitude ratio: {np.mean(gc_norms) / np.mean(at_norms):.2f}×")
print()

# Analyze sampling hypothesis
print("Testing sampling dependency hypothesis...")
print("  Checking if error chunks have different magnitude patterns...")

# Get error positions
error_positions = [(err['chrom'], err['pos']) for err in current_errors['errors']]

# Load chunk index
with h5py.File(hdf5_path, 'r') as f:
    chunk_keys = [key.decode('utf-8') for key in f['chunk_keys'][:]]
    chunk_to_idx = {key: idx for idx, key in enumerate(chunk_keys)}

# Get error chunk magnitudes
error_chunk_at_norms = []
error_chunk_gc_norms = []

for chrom, pos in error_positions[:100]:  # Sample first 100 errors
    chunk_start = (pos // 2000) * 2000
    chunk_key = f"{chrom}:{chunk_start}"

    if chunk_key in chunk_to_idx:
        idx = chunk_to_idx[chunk_key]
        with h5py.File(hdf5_path, 'r') as f:
            at_vec = f['AT_vectors'][idx]
            gc_vec = f['GC_vectors'][idx]
            error_chunk_at_norms.append(np.linalg.norm(at_vec))
            error_chunk_gc_norms.append(np.linalg.norm(gc_vec))

error_chunk_at_norms = np.array(error_chunk_at_norms)
error_chunk_gc_norms = np.array(error_chunk_gc_norms)

print(f"\n  Error chunks (first 100 errors):")
print(f"    AT norms - Mean: {np.mean(error_chunk_at_norms):.2f}, Std: {np.std(error_chunk_at_norms):.2f}")
print(f"    GC norms - Mean: {np.mean(error_chunk_gc_norms):.2f}, Std: {np.std(error_chunk_gc_norms):.2f}")
print(f"    GC/AT ratio in error chunks: {np.mean(error_chunk_gc_norms) / np.mean(error_chunk_at_norms):.2f}×")
print()

# Statistical analysis
print("=" * 80)
print("HYPOTHESIS TESTING")
print("=" * 80)
print()

print("Hypothesis 1: Random sampling caused different chunk selections")
print("  - 10,000 samples from 7.4M variants")
print("  - Expected: ~0.13% sampling rate")
print("  - Probability of hitting systematically different chunks: LOW")
print()

print("Hypothesis 2: Vector magnitude imbalance + normalization")
print(f"  - GC vectors are {np.mean(gc_norms) / np.mean(at_norms):.2f}× larger than AT")
print("  - Normalization: sim / ||vec||")
print("  - Effect: Lower magnitude vectors get relatively boosted")
print("  - This should be SYSTEMATIC, not random between runs")
print()

print("Hypothesis 3: Different random seeds")
print("  - Both runs should use np.random.seed(42) for position codebook")
print("  - Sample selection uses np.random.choice() without explicit seed")
print("  - This explains different test positions between runs")
print()

print("=" * 80)
print("ROOT CAUSE DIAGNOSIS")
print("=" * 80)
print()

print("CRITICAL FINDING:")
print("  The sample selection in both test scripts uses:")
print("    np.random.choice(len(variants), size=10000, replace=False)")
print("  WITHOUT setting a seed!")
print()
print("  This means each run selects DIFFERENT random positions from the")
print("  7.4M variants, which can hit chunks with different AT/GC magnitude")
print("  distributions.")
print()
print("SOLUTION:")
print("  1. Add np.random.seed(42) BEFORE np.random.choice()")
print("  2. This ensures reproducible sampling across runs")
print("  3. Error profiles should then be stable")
print()

print("DEEPER ISSUE:")
print(f"  GC vectors systematically {np.mean(gc_norms) / np.mean(at_norms):.2f}× larger magnitude")
print("  This magnitude imbalance may indicate an encoding issue in")
print("  genomevault/hypervector_transform/complementary_pair_encoder.py")
print()
print("  Expected: AT and GC should have similar magnitude distributions")
print("  Observed: GC vectors ~2× larger")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. Re-run all error profiling tests with fixed random seed")
print("2. Investigate ComplementaryPairEncoder for magnitude bias")
print("3. Consider magnitude normalization during encoding (not just query)")
print()
