#!/usr/bin/env python3
"""
Example script for encoding genomic data using the simplified hypervector engine.
"""

from genomevault.hypervector.engine import HypervectorEngine, HypervectorConfig
from genomevault.hypervector.featurizers.variants import featurize_variants
import json
import numpy as np


def main():
    # Example 1: Encode numeric features directly
    print("Example 1: Encoding numeric features")
    print("-" * 40)

    config = HypervectorConfig(dim=10000, seed=42, normalize=True)
    engine = HypervectorEngine(config)

    # Some example numeric features
    features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    hv = engine.encode_numeric(features)

    print(f"Input features: {features}")
    print(f"Hypervector shape: {hv.shape}")
    print(f"Hypervector norm: {np.linalg.norm(hv):.4f}")
    print(f"First 5 values: {hv[:5]}")

    # Save the vector
    hv.tofile("sample_numeric.vec")
    print("Saved to sample_numeric.vec\n")

    # Example 2: Encode genomic variants
    print("Example 2: Encoding genomic variants")
    print("-" * 40)

    # Create some fake genomic variant data
    variants = [
        {"chrom": "chr1", "pos": 12345, "ref": "A", "alt": "G", "impact": "HIGH"},
        {"chrom": "chr2", "pos": 67890, "ref": "T", "alt": "C", "impact": "MODERATE"},
        {"chrom": "chr3", "pos": 99999, "ref": "G", "alt": "A", "impact": "LOW"},
    ]

    # Featurize the variants
    features = featurize_variants(variants)
    print(f"Variant features: {features}")

    # Encode to hypervector
    hv = engine.encode_numeric(features)

    print(f"Hypervector shape: {hv.shape}")
    print(f"Hypervector norm: {np.linalg.norm(hv):.4f}")
    print(f"First 5 values: {hv[:5]}")

    # Save the vector
    hv.tofile("sample_variants.vec")
    print("Saved to sample_variants.vec\n")

    # Example 3: Binary encoding
    print("Example 3: Binary encoding")
    print("-" * 40)

    config_binary = HypervectorConfig(dim=10000, seed=42, binary=True)
    engine_binary = HypervectorEngine(config_binary)

    hv_binary = engine_binary.encode_numeric(features)

    print(f"Binary hypervector shape: {hv_binary.shape}")
    print(f"Unique values: {np.unique(hv_binary)}")
    print(f"First 20 binary values: {hv_binary[:20]}")

    # Save the binary vector
    hv_binary.tofile("sample_binary.vec")
    print("Saved to sample_binary.vec\n")

    # Example 4: Load from JSON file
    print("Example 4: Loading from JSON file")
    print("-" * 40)

    # Create a fake genome JSON file
    fake_genome = {"sample_id": "SAMPLE_001", "variants": variants}

    with open("fake_genome.json", "w") as f:
        json.dump(fake_genome, f, indent=2)
    print("Created fake_genome.json")

    # Load and process
    with open("fake_genome.json") as f:
        data = json.load(f)

    # Extract and featurize variants
    if "variants" in data:
        features = featurize_variants(data["variants"])
        hv = engine.encode_numeric(features)

        print(f"Sample ID: {data.get('sample_id', 'Unknown')}")
        print(f"Number of variants: {len(data['variants'])}")
        print(f"Encoded vector length: {hv.shape[0]}")

        # Save with sample ID
        output_file = f"{data.get('sample_id', 'unknown')}.vec"
        hv.tofile(output_file)
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
