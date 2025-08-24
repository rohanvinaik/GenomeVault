#!/usr/bin/env python3
"""
Debug test for multi-core and Metal acceleration.
"""

import time
from genomevault.compression.tiered_compression import TieredCompressor, CompressionTier
from genomevault.core.constants import OmicsType

def test_clinical_multicore():
    """Test CLINICAL tier with multi-core processing."""
    print("\n" + "="*60)
    print("Testing CLINICAL Tier Multi-Core Processing")
    print("="*60)
    
    # Initialize compressor
    compressor = TieredCompressor()
    
    # Create test data with variants
    test_data = {
        "sample_id": "TEST_001",
        "variants": {}
    }
    
    # Add 50,000 test variants (smaller for faster testing)
    for i in range(50000):
        test_data["variants"][f"rs{i}"] = i % 3  # genotype 0, 1, or 2
    
    print(f"Test data: {len(test_data['variants'])} variants")
    
    # Run compression
    start = time.time()
    compressed, metrics = compressor.compress_to_target(
        test_data,
        CompressionTier.CLINICAL,
        OmicsType.GENOMIC
    )
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Original size: {metrics.original_size:,} bytes")
    print(f"  Compressed size: {metrics.compressed_size:,} bytes")
    print(f"  Compression ratio: {metrics.compression_ratio:.2f}x")
    
    return metrics

if __name__ == "__main__":
    test_clinical_multicore()