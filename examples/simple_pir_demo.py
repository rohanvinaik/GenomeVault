from typing import Any, Dict

import numpy as np

from genomevault.hypervector.encoding import GenomicEncoder

#!/usr/bin/env python3
"""
Simple PIR (Private Information Retrieval) Demo
Demonstrates basic PIR concepts without complex imports
"""

import sys
from pathlib import Path

# Add genomevault to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def simple_pir_demo() -> None:
       """TODO: Add docstring for simple_pir_demo"""
     """Demonstrate basic PIR concepts"""

    print("Simple PIR Demo")
    print("=" * 40)

    # Create a simple "database" of genomic variants
    print("1. Creating variant database...")
    encoder = GenomicEncoder(dimension=10000)

    # Database of variants
    database = []
    for i in range(10):
        variant = {
            "chromosome": f"chr{i+1}",
            "position": 100000 + i * 1000,
            "ref": "ACGT"[i % 4],
            "alt": "TGCA"[i % 4],
            "type": "SNP",
        }
        hv = encoder.encode_variant(**variant)
        database.append((variant, hv))

    print(f"   Created database with {len(database)} variants")

    # Query for a specific variant (privately)
    print("\n2. Creating private query...")
    query_variant = {
        "chromosome": "chr3",
        "position": 102000,
        "ref": "G",
        "alt": "C",
        "type": "SNP",
    }
    query_hv = encoder.encode_variant(**query_variant)

    # Find similar variants using hypervector similarity
    print("\n3. Finding similar variants (privately)...")
    results = []
    for var, hv in database:
        similarity = encoder.similarity(query_hv, hv)
        results.append((similarity, var))

    # Sort by similarity
    results.sort(reverse=True)

    print("\nTop 3 most similar variants:")
    for sim, var in results[:3]:
        print(
            f"   Similarity: {sim:.3f} - {var['chromosome']}:{var['position']} {var['ref']}>{var['alt']}"
        )

    print("\n✓ Demo completed!")
    print("\nNote: In real PIR, the server wouldn't know which variant was queried.")
    print("This demo shows the basic concept using hypervector similarity.")


if __name__ == "__main__":
    simple_pir_demo()
