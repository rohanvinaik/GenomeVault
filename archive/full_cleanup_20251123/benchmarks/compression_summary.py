#!/usr/bin/env python3
"""
Simple Compression Summary - Extract and Document Compression Ratios

Uses existing benchmark data to document the compression calculation.
"""

import json
from pathlib import Path
from datetime import datetime

# Load latest benchmarks
latest_results_path = Path("/Users/rohanvinaik/genomevault/benchmark_results/differential_encoding/latest_results.json")

with open(latest_results_path) as f:
    data = json.load(f)

# Extract metrics
diff_ratio = data['summary']['key_metrics']['differential_encoding']['compression_ratio']
hv_ratio = data['summary']['key_metrics']['hypervector_projection']['compression_ratio']

# Calculate combined
combined_ratio = diff_ratio * hv_ratio

print("=" * 80)
print("GenomeVault Compression Calculation - Summary")
print("=" * 80)
print()
print("Data Source:")
print(f"  File: benchmark_results/differential_encoding/latest_results.json")
print(f"  Timestamp: {data['metadata']['timestamp']}")
print(f"  Platform: {data['metadata'].get('architecture', 'N/A')}")
print()

print("COMPRESSION BREAKDOWN:")
print("-" * 80)
print(f"  Stage 1: Differential Encoding     {diff_ratio}×")
print(f"  Stage 2: Hypervector Projection    {hv_ratio}×")
print(f"  Combined: {diff_ratio}× × {hv_ratio}× = {combined_ratio}×")
print()

print("SIZE ESTIMATION (10,000 variants):")
print("-" * 80)

# Typical VCF is ~100 bytes per variant (header + data)
variants = 10000
bytes_per_variant = 100
raw_size_bytes = variants * bytes_per_variant
raw_size_mb = raw_size_bytes / (1024 * 1024)

# After differential
diff_size_bytes = raw_size_bytes / diff_ratio
diff_size_kb = diff_size_bytes / 1024

# After hypervector
hv_size_bytes = diff_size_bytes / hv_ratio
hv_size_kb = hv_size_bytes / 1024

# Final after compression
final_size_bytes = raw_size_bytes / combined_ratio
final_size_kb = final_size_bytes / 1024

print(f"  Raw VCF:                 {raw_size_mb:.2f} MB ({raw_size_bytes:,} bytes)")
print(f"  After Differential:      {diff_size_kb:.1f} KB ({diff_size_bytes:.0f} bytes, {diff_ratio}× compression)")
print(f"  After Hypervector:       {hv_size_kb:.1f} KB ({hv_size_bytes:.0f} bytes, {hv_ratio}× compression)")
print(f"  Final Size:              {final_size_kb:.1f} KB ({final_size_bytes:.0f} bytes)")
print()
print(f"  Total Compression Ratio: {combined_ratio}×")
print()

# Create output JSON
output = {
    "timestamp": datetime.now().isoformat(),
    "source_data": {
        "file": "benchmark_results/differential_encoding/latest_results.json",
        "timestamp": data['metadata']['timestamp'],
    },
    "compression_ratios": {
        "differential_encoding": diff_ratio,
        "hypervector_projection": hv_ratio,
        "combined_multiplicative": combined_ratio,
    },
    "size_calculation_example": {
        "input_variants": variants,
        "raw_vcf_mb": round(raw_size_mb, 2),
        "after_differential_kb": round(diff_size_kb, 1),
        "after_hypervector_kb": round(hv_size_kb, 1),
        "final_size_kb": round(final_size_kb, 1),
    },
    "paper_claim": {
        "claimed_compression": "264×",
        "calculated_compression": f"{combined_ratio}×",
        "method": f"{diff_ratio}× (differential) × {hv_ratio}× (hypervector)",
        "status": "VERIFIED" if combined_ratio == 264 else "DISCREPANCY",
    }
}

output_file = Path("compression_summary.json")
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Saved to: {output_file}")
print()

print("PAPER CLAIM VERIFICATION:")
print("-" * 80)
print(f"  Claimed:    264× compression")
print(f"  Calculated: {combined_ratio}× compression")
print(f"  Status:     {'✓ VERIFIED' if combined_ratio == 264 else '✗ NEEDS UPDATE IN PAPER'}")
print()

if combined_ratio != 264:
    print("RECOMMENDATION:")
    print("-" * 80)
    print(f"  Update paper to reflect actual compression: {combined_ratio}×")
    print(f"  Or provide additional compression stage to reach 264×")
    print()

print("=" * 80)
