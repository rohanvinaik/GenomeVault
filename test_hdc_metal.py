#!/usr/bin/env python3
"""
Test FULL_HDC tier with Metal acceleration.
"""

import time
import numpy as np
from genomevault.compression.tiered_compression import TieredCompressor, CompressionTier
from genomevault.core.constants import OmicsType

def test_hdc_with_metal():
    """Test FULL_HDC tier which should use Metal for hypervector encoding."""
    print("\n" + "="*60)
    print("Testing FULL_HDC Tier with Metal Acceleration")
    print("="*60)
    
    # Initialize compressor
    compressor = TieredCompressor()
    
    # Create test data with many variants to trigger HDC encoding
    test_data = {
        "sample_id": "TEST_HDC_001",
        "variants": {}
    }
    
    # Add 200,000 test variants - use regular int not numpy int64
    print("Creating test data with 200,000 variants...")
    for i in range(200000):
        test_data["variants"][f"rs{i}"] = int(i % 3)  # genotype 0, 1, or 2 - force regular int
    
    print(f"Test data: {len(test_data['variants'])} variants")
    
    # Run compression for FULL_HDC tier which uses hypervector encoding
    print("\nRunning FULL_HDC compression (should use Metal)...")
    start = time.time()
    compressed, metrics = compressor.compress_to_target(
        test_data,
        CompressionTier.FULL_HDC,
        OmicsType.GENOMIC
    )
    elapsed = time.time() - start
    
    print(f"\n✅ Results:")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Original size: {metrics.original_size:,} bytes")
    print(f"  Compressed size: {metrics.compressed_size:,} bytes")
    print(f"  Compression ratio: {metrics.compression_ratio:.2f}x")
    print(f"  Target size: {CompressionTier.FULL_HDC.target_bytes:,} bytes")
    
    # Check if we met the target
    if metrics.compressed_size <= CompressionTier.FULL_HDC.target_bytes:
        print(f"  ✅ Met target size!")
    else:
        print(f"  ⚠️  Exceeded target by {metrics.compressed_size - CompressionTier.FULL_HDC.target_bytes:,} bytes")
    
    # Check logs for Metal usage
    print("\nChecking logs for Metal acceleration...")
    with open("/Users/rohanvinaik/genomevault/logs/genomevault.log", "r") as f:
        lines = f.readlines()[-100:]  # Last 100 lines
        metal_lines = [l for l in lines if "Metal" in l or "🍎" in l or "hypervector using HDC" in l]
        if metal_lines:
            print("  Found Metal acceleration evidence:")
            for line in metal_lines[-5:]:
                print(f"    {line.strip()}")
        else:
            print("  ⚠️  No direct Metal evidence in recent logs")
    
    return metrics

if __name__ == "__main__":
    test_hdc_with_metal()