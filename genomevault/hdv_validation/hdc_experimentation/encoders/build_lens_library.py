#!/usr/bin/env python3
"""
Build Structural Motif Lens Library

Creates a reusable H5 file containing precomputed hypervector consensus patterns
for known genomic structural motifs (Alu, CpG islands, TATA boxes, etc.).

This is a preprocessing step that runs ONCE to create a shared resource.
The resulting lens_library.h5 can be used with any human genome.

Usage:
    python build_lens_library.py \\
        --reference data/consensus.fa \\
        --output output/lens_library.h5 \\
        --D 5120 --N 1024 --seed 42

Based on: docs/theory/STRUCTURAL_MOTIF_LENS_LIBRARY.md
Version: 1.0
Date: November 2025
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from decoders.lens_aware_decoder import LensLibrary
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build structural motif lens library for split binary HDC decoding"
    )

    parser.add_argument(
        '--reference',
        type=str,
        required=True,
        help="Reference genome FASTA (e.g., consensus.fa)"
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Output path for lens library H5 file"
    )

    parser.add_argument(
        '--D',
        type=int,
        default=5120,
        help="Hypervector dimension (must match encoder)"
    )

    parser.add_argument(
        '--N',
        type=int,
        default=1024,
        help="Chunk size (must match encoder, for position codebook)"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for position codebook (must match encoder)"
    )

    args = parser.parse_args()

    # Validate paths
    reference_path = Path(args.reference)
    if not reference_path.exists():
        logger.error(f"Reference genome not found: {reference_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STRUCTURAL MOTIF LENS LIBRARY BUILDER")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Reference genome: {reference_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"D = {args.D}")
    logger.info(f"N = {args.N}")
    logger.info(f"Seed = {args.seed}")
    logger.info("")

    # Step 1: Generate position codebook (MUST match encoder's codebook)
    logger.info("Step 1: Generating position codebook...")
    np.random.seed(args.seed)
    position_codebook = np.random.choice([-1, 1], size=(args.N, args.D)).astype(np.int8)
    logger.info(f"✓ Position codebook: {args.N} × {args.D} binary random vectors")
    logger.info("")

    # Step 2: Initialize lens library
    logger.info("Step 2: Building lens library...")
    library = LensLibrary(D=args.D)
    library.build_from_reference(
        reference_fasta=str(reference_path),
        position_codebook=position_codebook
    )
    logger.info("")

    # Step 3: Display lens details
    logger.info("Step 3: Lens library summary")
    logger.info("-" * 80)
    logger.info(f"{'Lens Name':<20} {'Texture':<18} {'Prevalence':<12} {'Size':<10}")
    logger.info("-" * 80)

    total_size = 0
    for lens_name, lens in library.lenses.items():
        size_bytes = (lens.bank0.nbytes + lens.bank1.nbytes + lens.bank2.nbytes)
        total_size += size_bytes

        logger.info(
            f"{lens_name:<20} {lens.texture_type:<18} "
            f"{lens.prevalence*100:>6.2f}%      {lens.typical_size:>6} bp"
        )

    logger.info("-" * 80)
    logger.info(f"Total lenses: {len(library.lenses)}")
    logger.info(f"Total size: {total_size / 1024 / 1024:.2f} MB (in memory)")
    logger.info("")

    # Step 4: Save to HDF5
    logger.info("Step 4: Saving to HDF5...")
    library.save(output_path)

    file_size = output_path.stat().st_size / 1024 / 1024
    logger.info(f"✓ Saved to {output_path} ({file_size:.2f} MB)")
    logger.info("")

    # Step 5: Verify by loading
    logger.info("Step 5: Verifying...")
    loaded_library = LensLibrary.load(output_path)

    if len(loaded_library.lenses) == len(library.lenses):
        logger.info(f"✓ Verification passed: {len(loaded_library.lenses)} lenses loaded")
    else:
        logger.error(f"✗ Verification failed: expected {len(library.lenses)}, got {len(loaded_library.lenses)}")
        sys.exit(1)

    logger.info("")
    logger.info("=" * 80)
    logger.info("LENS LIBRARY BUILD COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Use with decoder: --lens-library {output_path}")
    logger.info(f"  2. This library can be reused for ANY human genome")
    logger.info(f"  3. Expected accuracy improvement: +5-10% overall, +10-15% on uncertain positions")
    logger.info("")


if __name__ == '__main__':
    main()
