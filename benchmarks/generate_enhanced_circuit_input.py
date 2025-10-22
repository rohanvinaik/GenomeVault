#!/usr/bin/env python3
"""
Generate test input for enhanced variant_presence circuit.

The enhanced circuit requires:
- 10 variants (batch processing)
- Each with full 20-level Merkle tree path
- Comprehensive fields (chromosome, position, alleles, genotype, quality, AF)
"""

import json
import hashlib
import random
from pathlib import Path
from typing import Dict, List

def poseidon_hash_simulation(*inputs):
    """Simulate Poseidon hash using SHA256 (for testing only)"""
    data = "".join(str(i) for i in inputs)
    return int(hashlib.sha256(data.encode()).hexdigest()[:16], 16)

def generate_merkle_path(leaf_index: int, tree_depth: int = 20) -> tuple:
    """Generate a valid Merkle path for testing"""
    path_elements = []
    path_indices = []

    current_index = leaf_index
    for level in range(tree_depth):
        # Generate sibling hash
        sibling = poseidon_hash_simulation(level, current_index)
        path_elements.append(str(sibling))

        # Determine if we're left or right child
        path_indices.append(str(current_index % 2))

        # Move up the tree
        current_index = current_index // 2

    return path_elements, path_indices

def generate_variant(
    variant_id: int,
    chromosome: int = None,
    position: int = None
) -> Dict:
    """Generate a single variant with all required fields"""

    if chromosome is None:
        chromosome = random.randint(1, 22)  # Autosomes

    if position is None:
        position = random.randint(1000000, 100000000)

    # Alleles (encoded as integers for circuit)
    # A=65, C=67, G=71, T=84 (ASCII values)
    allele_map = {0: 65, 1: 67, 2: 71, 3: 84}  # A, C, G, T
    ref_allele = random.choice([65, 67, 71, 84])
    alt_options = [a for a in [65, 67, 71, 84] if a != ref_allele]
    alt_allele = random.choice(alt_options)

    # Genotype: 0=0/0, 1=0/1, 2=1/0, 3=1/1
    genotype = random.choice([1, 2, 3])  # Exclude 0/0 (no variant)

    # Quality score (PHRED scale, 20-99)
    quality_score = random.randint(25, 99)

    # Allele frequency (0-100, representing percentage)
    allele_frequency = random.randint(1, 50)

    # Witness randomness (for privacy)
    witness_randomness = random.randint(1, 2**64)

    # Generate Merkle path
    merkle_path, merkle_indices = generate_merkle_path(variant_id)

    return {
        "chromosome": str(chromosome),
        "position": str(position),
        "ref_allele": str(ref_allele),
        "alt_allele": str(alt_allele),
        "genotype": str(genotype),
        "quality_score": str(quality_score),
        "allele_frequency": str(allele_frequency),
        "merkle_path": merkle_path,
        "merkle_indices": merkle_indices,
        "witness_randomness": str(witness_randomness)
    }

def generate_batch_input(
    num_variants: int = 10,
    num_valid: int = None
) -> Dict:
    """Generate full input for batch circuit"""

    if num_valid is None:
        num_valid = num_variants

    # Generate variants
    variants = [generate_variant(i) for i in range(num_variants)]

    # Calculate commitment root (simplified for testing)
    # In production, this would be computed from all variants in the database
    variant_hashes = []
    for v in variants:
        h = poseidon_hash_simulation(
            v["chromosome"],
            v["position"],
            v["ref_allele"],
            v["alt_allele"],
            v["genotype"],
            v["quality_score"],
            v["allele_frequency"],
            v["witness_randomness"]
        )
        variant_hashes.append(h)

    commitment_root = poseidon_hash_simulation(*variant_hashes)

    # Build input structure (arrays for batch)
    input_data = {
        "commitment_root": str(commitment_root),
        "expected_num_valid": str(num_valid),
        "chromosomes": [v["chromosome"] for v in variants],
        "positions": [v["position"] for v in variants],
        "ref_alleles": [v["ref_allele"] for v in variants],
        "alt_alleles": [v["alt_allele"] for v in variants],
        "genotypes": [v["genotype"] for v in variants],
        "quality_scores": [v["quality_score"] for v in variants],
        "allele_frequencies": [v["allele_frequency"] for v in variants],
        "merkle_paths": [v["merkle_path"] for v in variants],
        "merkle_indices": [v["merkle_indices"] for v in variants],
        "witness_randomness": [v["witness_randomness"] for v in variants]
    }

    return input_data

def main():
    """Generate and save test input"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate test input for enhanced variant_presence circuit"
    )
    parser.add_argument(
        "--num-variants", type=int, default=10,
        help="Number of variants in batch (default: 10)"
    )
    parser.add_argument(
        "--output", default="input_enhanced.json",
        help="Output file path (default: input_enhanced.json)"
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    print(f"Generating test input for {args.num_variants} variants...")

    input_data = generate_batch_input(args.num_variants)

    # Save to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        if args.pretty:
            json.dump(input_data, f, indent=2)
        else:
            json.dump(input_data, f)

    print(f"✅ Saved to: {output_path}")
    print(f"   Variants: {args.num_variants}")
    print(f"   Expected valid: {input_data['expected_num_valid']}")
    print(f"   Commitment root: {input_data['commitment_root'][:16]}...")
    print(f"   File size: {output_path.stat().st_size:,} bytes")
    print("")
    print("To generate a witness:")
    print(f"  cd genomevault/zk/circuits/variant_presence/build")
    print(f"  node variant_presence_enhanced_js/generate_witness.js \\")
    print(f"    variant_presence_enhanced_js/variant_presence_enhanced.wasm \\")
    print(f"    {output_path.absolute()} \\")
    print(f"    witness.wtns")

if __name__ == "__main__":
    main()
