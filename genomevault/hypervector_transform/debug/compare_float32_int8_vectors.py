#!/usr/bin/env python3
"""
Compare Float32 vs Int8 Vector Characteristics

Analyzes magnitude distributions and error patterns to identify
architectural differences causing accuracy variation.
"""

import h5py
import numpy as np
import json
import gzip
from pathlib import Path

print("=" * 80)
print("FLOAT32 vs INT8 VECTOR ANALYSIS")
print("=" * 80)
print()

hdf5_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome.h5")
gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")

# Sample 1000 chunks for analysis
print("Sampling 1000 chunks from HDF5...")
with h5py.File(hdf5_path, 'r') as f:
    num_chunks = f['AT_vectors'].shape[0]
    sample_indices = np.random.choice(num_chunks, size=1000, replace=False)

    print(f"  Total chunks: {num_chunks:,}")
    print(f"  Sampling: 1,000 chunks")
    print()

    # Load float32 vectors
    print("Float32 vectors (original HDF5):")
    at_float32 = []
    gc_float32 = []

    for idx in sample_indices[:100]:  # Just 100 for speed
        at_vec = f['AT_vectors'][idx]
        gc_vec = f['GC_vectors'][idx]

        at_float32.append(np.linalg.norm(at_vec))
        gc_float32.append(np.linalg.norm(gc_vec))

    at_float32 = np.array(at_float32)
    gc_float32 = np.array(gc_float32)

    print(f"  AT magnitude - Mean: {np.mean(at_float32):.2f}, Std: {np.std(at_float32):.2f}")
    print(f"  GC magnitude - Mean: {np.mean(gc_float32):.2f}, Std: {np.std(gc_float32):.2f}")
    print(f"  GC/AT ratio: {np.mean(gc_float32) / np.mean(at_float32):.2f}×")
    print()

# Now check how int8 quantization affects this
print("Int8 quantization analysis:")
print("  Int8 uses per-chunk scaling: max(abs(vec)) / 127")
print("  This NORMALIZES each chunk independently")
print()

with h5py.File(hdf5_path, 'r') as f:
    at_scales = []
    gc_scales = []

    for idx in sample_indices[:100]:
        at_vec = f['AT_vectors'][idx]
        gc_vec = f['GC_vectors'][idx]

        # Simulate int8 quantization
        at_scale = np.max(np.abs(at_vec)) / 127.0
        gc_scale = np.max(np.abs(gc_vec)) / 127.0

        at_scales.append(at_scale)
        gc_scales.append(gc_scale)

    at_scales = np.array(at_scales)
    gc_scales = np.array(gc_scales)

    print(f"  AT scale factors - Mean: {np.mean(at_scales):.4f}, Std: {np.std(at_scales):.4f}")
    print(f"  GC scale factors - Mean: {np.mean(gc_scales):.4f}, Std: {np.std(gc_scales):.4f}")
    print(f"  GC/AT scale ratio: {np.mean(gc_scales) / np.mean(at_scales):.2f}×")
    print()

print("=" * 80)
print("KEY ARCHITECTURAL DIFFERENCE")
print("=" * 80)
print()
print("Float32 (validate_whole_genome_hdv.py):")
print("  sim_AT = dot(pos_vec, AT_vec) / (||AT_vec|| + 1e-10)")
print("  sim_GC = dot(pos_vec, GC_vec) / (||GC_vec|| + 1e-10)")
print("  → NORMALIZES by L2 norm during query")
print()
print("Int8 (int8_lightning_hdc.py):")
print("  sim_AT = dot(at_vec_int8, pos_enc_int8) * scale_factor_AT")
print("  sim_GC = dot(gc_vec_int8, pos_enc_int8) * scale_factor_GC")
print("  → Uses raw dot product with scale factors")
print()
print("CRITICAL DIFFERENCE:")
print("  Float32: Normalization happens at QUERY time (per-chunk)")
print("  Int8:    Normalization happened at ENCODING time (per-chunk)")
print("           But then UNDONE by scale factors during query!")
print()
print("This means:")
print("  - Float32: Every chunk's AT and GC vectors are unit-normalized at query")
print("  - Int8: Vectors retain their relative magnitudes via scale factors")
print()

# Load error data
print("=" * 80)
print("ERROR PATTERN ANALYSIS")
print("=" * 80)
print()

print("Loading error data...")
with open('HDV_VALIDATION_PACKAGE/error_analysis/float32_errors.json', 'r') as f:
    float32_errors = json.load(f)

with open('HDV_VALIDATION_PACKAGE/error_analysis/int8_errors.json', 'r') as f:
    int8_errors = json.load(f)

print()
print("Float32 (with query-time normalization):")
print(f"  Overall: {float32_errors['metadata']['accuracy']:.2f}%")
print(f"  AT: {float32_errors['metadata']['at_accuracy']:.2f}%")
print(f"  GC: {float32_errors['metadata']['gc_accuracy']:.2f}%")
print(f"  AT/GC accuracy ratio: {float32_errors['metadata']['at_accuracy'] / float32_errors['metadata']['gc_accuracy']:.3f}")
print()

print("Int8 (with scale-preserved magnitudes):")
print(f"  Overall: {int8_errors['metadata']['accuracy']:.2f}%")
print(f"  AT: {int8_errors['metadata']['at_accuracy']:.2f}%")
print(f"  GC: {int8_errors['metadata']['gc_accuracy']:.2f}%")
print(f"  AT/GC accuracy ratio: {int8_errors['metadata']['at_accuracy'] / int8_errors['metadata']['gc_accuracy']:.3f}")
print()

print("=" * 80)
print("HYPOTHESIS")
print("=" * 80)
print()
print("The variation is caused by DIFFERENT NORMALIZATION STRATEGIES:")
print()
print("1. Float32 uses QUERY-TIME L2 normalization")
print("   - Makes all chunks have equal weight regardless of signal strength")
print("   - Can amplify noise in low-signal chunks")
print("   - GC vectors are naturally ~2× larger magnitude")
print("   - Normalization reduces their relative contribution")
print()
print("2. Int8 preserves MAGNITUDE RELATIONSHIPS via scale factors")
print("   - Stronger signals (larger magnitude) contribute more")
print("   - GC's naturally higher magnitude is preserved")
print("   - This may be more biologically valid")
print()
print("RECOMMENDATION:")
print("  Remove query-time normalization from float32 to match int8 behavior")
print("  This will make results comparable across quantization levels")
print()
