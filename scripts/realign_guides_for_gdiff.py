#!/usr/bin/env python3
"""
Re-align Guide FASTQs to Own FASTAs (GDiff Coordinate System Fix)

CRITICAL ARCHITECTURE FIX:
Guide BAMs originally aligned to consensus are in consensus coordinate space.
Experimental BAM aligned to guide FASTAs is in guide FASTA coordinate space.
GDiff encoder needs BOTH in the SAME coordinate system.

This script re-aligns all 12 guide FASTQs to their own guide FASTAs,
creating ref*_gdiff.bam files in guide FASTA coordinate space.

Usage:
    python3 scripts/realign_guides_for_gdiff.py

Output:
    data/guide_strands/ref1_gdiff.bam
    data/guide_strands/ref2_gdiff.bam
    ...
    data/guide_strands/ref12_gdiff.bam

Time estimate: ~2-3 hours per guide (30-36 hours total for k=12)
"""

import sys
import logging
from pathlib import Path

# Add genomevault to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.differential_encoding.align_to_reference_pool import PrivacyPreservingReferencePoolAligner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Re-align all 12 guide FASTQs to their own guide FASTAs."""

    logger.info("=" * 80)
    logger.info("GUIDE RE-ALIGNMENT FOR GDIFF COORDINATE SYSTEM FIX")
    logger.info("=" * 80)
    logger.info("This creates guide BAMs in guide FASTA coordinate space")
    logger.info("Required for GDiff encoder to compare experimental vs guide BAMs")
    logger.info("=" * 80)

    # Map ref numbers to ERR IDs (excluding ERR3239334 which is experimental)
    # Based on the 13 ERR samples found, using 12 as guides
    ERR_MAPPING = {
        1: ("ERR3239276", "data/downloaded/fastq/ERR3239276"),
        2: ("ERR3239454", "data/downloaded/fastq/ERR3239454"),
        3: ("ERR3239475", "data/downloaded/fastq/ERR3239475"),
        4: ("ERR3239548", "data/downloaded/fastq/european/ERR3239548/ERR3239548"),
        5: ("ERR3239590", "data/downloaded/fastq/european/ERR3239590/ERR3239590"),
        6: ("ERR3239920", "data/downloaded/fastq/european/ERR3239920/ERR3239920"),
        7: ("ERR3239578", "data/downloaded/fastq/east_asian/ERR3239578/ERR3239578"),
        8: ("ERR3239612", "data/downloaded/fastq/east_asian/ERR3239612/ERR3239612"),
        9: ("ERR3239756", "data/downloaded/fastq/african/european/ERR3239756/ERR3239756"),
        10: ("ERR3239778", "data/downloaded/fastq/african/european/ERR3239778/ERR3239778"),
        11: ("ERR3239912", "data/downloaded/fastq/south_asian/european/ERR3239912/ERR3239912"),
        12: ("ERR3239934", "data/downloaded/fastq/south_asian/european/ERR3239934/ERR3239934"),
    }

    guide_dir = Path("/Volumes/1TBStorage/guide_strands")  # SD card location
    guide_data = []

    # Verify all inputs exist before starting
    logger.info("\nVerifying inputs...")
    missing_files = []

    for ref_num, (err_id, fastq_base) in ERR_MAPPING.items():
        guide_fasta = guide_dir / f"ref{ref_num}.fa.gz"
        fastq_r1 = Path(f"{fastq_base}_1.fastq.gz")
        fastq_r2 = Path(f"{fastq_base}_2.fastq.gz")
        output_bam = guide_dir / f"ref{ref_num}_gdiff.bam"

        # Check guide FASTA
        if not guide_fasta.exists():
            missing_files.append(f"Guide FASTA missing: {guide_fasta}")
            continue

        # Check FASTQ files
        if not fastq_r1.exists():
            missing_files.append(f"FASTQ R1 missing: {fastq_r1}")
            continue
        if not fastq_r2.exists():
            missing_files.append(f"FASTQ R2 missing: {fastq_r2}")
            continue

        logger.info(f"  ✓ ref{ref_num}: {err_id}")
        guide_data.append((guide_fasta, fastq_r1, fastq_r2, output_bam))

    if missing_files:
        logger.error("\n❌ Missing files detected:")
        for f in missing_files:
            logger.error(f"  - {f}")
        logger.error("\nCannot proceed. Please ensure all guide FASTAs and FASTQs exist.")
        return 1

    logger.info(f"\n✅ All {len(guide_data)} guide inputs verified")
    logger.info(f"\nStarting re-alignment...")
    logger.info(f"  Output directory: {guide_dir}")
    logger.info(f"  Threads: 10")
    logger.info(f"  Estimated time: ~2-3 hours per guide (~{len(guide_data)*2.5:.1f} hours total)")
    logger.info("=" * 80)

    # Re-align all 12 guides to their own FASTAs
    gdiff_bams = PrivacyPreservingReferencePoolAligner.align_guides_to_own_fastas(
        guide_data=guide_data,
        threads=10
    )

    logger.info("\n" + "=" * 80)
    logger.info("✅ RE-ALIGNMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Created {len(gdiff_bams)} guide BAMs in guide FASTA coordinate space:")
    for bam in gdiff_bams:
        size_gb = bam.stat().st_size / (1024**3)
        logger.info(f"  ✓ {bam.name} ({size_gb:.2f} GB)")

    logger.info("\nThese BAMs are now ready for GDiff encoding!")
    logger.info("Next step: Run the k=12 GDiff pipeline:")
    logger.info("  python3 benchmarks/run_k12_gdiff_pipeline.py")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
