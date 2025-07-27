from genomevault.local_processing import DifferentialStorage
from genomevault.hypervector.encoding import GenomicEncoder
from typing import Any, Dict

#!/usr/bin/env python3
"""
Simple GenomeVault demo that doesn't require pysam
"""

import sys
from pathlib import Path

# Add genomevault to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
        """TODO: Add docstring for main"""
    print("GenomeVault Simple Demo")
    print("=" * 40)

    # Create encoder
    encoder = GenomicEncoder(dimension=10000)
    print("✓ Created genomic encoder")

    # Encode a variant
    variant_hv = encoder.encode_variant(
        chromosome="chr1", position=123456, ref="A", alt="G", variant_type="SNP"
    )
    print("✓ Encoded variant")

    # Create a simple genome with a few variants
    variants = [
        {"chromosome": "chr1", "position": 123456, "ref": "A", "alt": "G", "type": "SNP"},
        {"chromosome": "chr2", "position": 234567, "ref": "C", "alt": "T", "type": "SNP"},
        {"chromosome": "chr3", "position": 345678, "ref": "G", "alt": "A", "type": "SNP"},
    ]

    genome_hv = encoder.encode_genome(variants)
    print(f"✓ Encoded genome with {len(variants)} variants")

    # Test similarity
    variant2_hv = encoder.encode_variant(
        chromosome="chr1",
        position=123457,  # Adjacent position
        ref="A",
        alt="G",
        variant_type="SNP",
    )

    similarity = encoder.similarity(variant_hv, variant2_hv)
    print(f"✓ Similarity between adjacent variants: {similarity:.3f}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
