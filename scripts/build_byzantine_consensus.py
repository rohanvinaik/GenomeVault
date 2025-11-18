#!/usr/bin/env python3
"""
Build Byzantine Consensus Reference

Command-line interface for building consensus reference genomes from multiple
public references with positional uncertainty and privacy guarantees.

Usage:
    python scripts/build_byzantine_consensus.py \
        --references data/reference_genomes/hg38.fa.gz \
                     data/reference_genomes/hg19.fa.gz \
                     data/reference_genomes/chm13v2.0.fa.gz \
        --output data/reference_genomes/consensus \
        --chromosomes chr22 \
        --confidence-threshold 0.9 \
        --threads 8

For whole genome (long-running):
    python scripts/build_byzantine_consensus.py \
        --references data/reference_genomes/hg38.fa.gz \
                     data/reference_genomes/hg19.fa.gz \
                     data/reference_genomes/chm13v2.0.fa.gz \
        --output data/reference_genomes/consensus \
        --threads 8
"""

import sys
import argparse
import logging
from pathlib import Path

# Add genomevault to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.reference import build_consensus_reference


def main():
    parser = argparse.ArgumentParser(
        description="Build Byzantine Consensus Reference from multiple public references",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--references',
        nargs='+',
        required=True,
        metavar='FASTA',
        help='Paths to reference FASTA files (.fa or .fa.gz). '
             'Typically: hg38.fa.gz hg19.fa.gz chm13v2.0.fa.gz'
    )

    parser.add_argument(
        '--output',
        required=True,
        metavar='DIR',
        help='Output directory for consensus files'
    )

    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.9,
        metavar='FLOAT',
        help='Minimum confidence for unambiguous base (default: 0.9). '
             'Lower values = more positional uncertainty = stronger privacy'
    )

    parser.add_argument(
        '--chromosomes',
        nargs='+',
        metavar='CHR',
        help='Specific chromosomes to process (e.g., chr22 chr21). '
             'Default: all chromosomes (LONG RUNNING)'
    )

    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        metavar='N',
        help='Number of threads for parallel processing (default: 1)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode: only process chr22 (for testing)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # Quick test mode
    if args.quick_test and not args.chromosomes:
        args.chromosomes = ['chr22']
        logger.info("Quick test mode: processing chr22 only")

    # Validate inputs
    references = [Path(r) for r in args.references]
    for ref in references:
        if not ref.exists():
            logger.error(f"Reference file not found: {ref}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("Byzantine Consensus Reference Builder")
    logger.info("=" * 70)
    logger.info(f"References: {len(references)}")
    for ref in references:
        logger.info(f"  - {ref.name}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Confidence threshold: {args.confidence_threshold}")
    if args.chromosomes:
        logger.info(f"Chromosomes: {', '.join(args.chromosomes)}")
    else:
        logger.info("Chromosomes: ALL (this will take several hours)")
    logger.info(f"Threads: {args.threads}")
    logger.info("=" * 70)

    # Build consensus
    try:
        output_files = build_consensus_reference(
            references=references,
            output_dir=Path(args.output),
            confidence_threshold=args.confidence_threshold,
            chromosomes=args.chromosomes,
            threads=args.threads
        )

        # Summary
        logger.info("=" * 70)
        logger.info("SUCCESS! Byzantine Consensus Reference built")
        logger.info("=" * 70)
        logger.info("Output files:")
        for file_type, file_path in output_files.items():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  {file_type:20s}: {file_path} ({size_mb:.2f} MB)")
        logger.info("=" * 70)

        logger.info("\nNext steps:")
        logger.info("  1. Use consensus.fa as reference for aligning your 4 genomes")
        logger.info("  2. Run: python benchmarks/run_byzantine_privacy_stack.py")
        logger.info("  3. See: docs/guides/BYZANTINE_CONSENSUS_PRIVACY_STACK.md")

        return 0

    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
