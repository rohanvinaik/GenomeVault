"""
Generate BED files for UCSC LiftOver/Collision testing from HDV error analysis.

This module creates BED files comparing HDV biophysical predictions against
BAM ground truth for positions where high-precision quantizations disagree.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def parse_position(position_str: str) -> Tuple[str, int, int]:
    """
    Parse position string to BED coordinates.

    Args:
        position_str: Format "chrX_consensus:81532277"

    Returns:
        (chrom, start, end) where start is 0-based, end is 1-based (BED format)

    Example:
        >>> parse_position("chr1_consensus:12345")
        ("chr1", 12344, 12345)
    """
    if ':' not in position_str:
        raise ValueError(f"Invalid position format: {position_str}")

    chrom_raw, pos_str = position_str.split(':')
    chrom = chrom_raw.replace('_consensus', '')  # "chrX_consensus" -> "chrX"

    pos = int(pos_str)
    start = pos - 1  # BED is 0-based start
    end = pos        # BED is 1-based end

    return chrom, start, end


def generate_collision_beds(
    input_json: Path,
    output_dir: Path,
    quantization: str = 'float32',
    fallback_quantization: str = 'int8',
    prefix: str = None
) -> Tuple[Path, Path]:
    """
    Generate BED files for collision testing from HDV error analysis.

    Creates two BED files:
    1. genomevault_predictions.bed - HDV biophysical predictions
    2. bam_ground_truth.bed - BAM reference calls

    Args:
        input_json: Path to error JSON (e.g., high_precision_errors.json)
        output_dir: Directory to save BED files
        quantization: Primary quantization to use for HDV calls (default: float32)
        fallback_quantization: Fallback if primary missing (default: int8)

    Returns:
        (hdc_bed_path, bam_bed_path) - Paths to generated BED files

    Raises:
        FileNotFoundError: If input JSON doesn't exist
        ValueError: If JSON format is invalid
    """
    logger.info("=" * 80)
    logger.info("GENERATING COLLISION BED FILES")
    logger.info("=" * 80)
    logger.info("")

    # Validate input
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load error data
    logger.info(f"Loading error data from: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)

    if 'errors' not in data:
        raise ValueError(f"Invalid JSON format - missing 'errors' key in {input_json}")

    errors = data['errors']
    logger.info(f"  Loaded {len(errors)} error positions")
    logger.info("")

    # Output file paths
    if prefix:
        hdc_bed_path = output_dir / f"{prefix}_genomevault_predictions.bed"
        bam_bed_path = output_dir / f"{prefix}_bam_ground_truth.bed"
    else:
        hdc_bed_path = output_dir / "genomevault_predictions.bed"
        bam_bed_path = output_dir / "bam_ground_truth.bed"

    # Counters for statistics
    valid_positions = 0
    skipped_positions = 0
    missing_hdc_call = 0

    logger.info(f"Generating BED files...")
    logger.info(f"  Primary quantization: {quantization}")
    logger.info(f"  Fallback quantization: {fallback_quantization}")
    logger.info("")

    with open(hdc_bed_path, 'w') as f_hdc, \
         open(bam_bed_path, 'w') as f_bam:

        # Write UCSC custom track headers
        f_hdc.write('track name="GenomeVault_HDC" description="HDC Biophysical Predictions" color=0,128,0\n')
        f_bam.write('track name="BAM_Reference" description="Standard BAM Calls" color=128,0,0\n')

        for error in errors:
            try:
                # Parse position
                position_str = error['position']
                chrom, start, end = parse_position(position_str)

                # Get BAM ground truth
                bam_call = error['ground_truth']

                # Get HDV prediction (prefer primary, fallback to secondary)
                predictions = error['predictions']
                hdc_call = predictions.get(quantization, predictions.get(fallback_quantization, None))

                if hdc_call is None:
                    logger.warning(f"  No {quantization}/{fallback_quantization} prediction for {position_str}")
                    missing_hdc_call += 1
                    continue

                # Get confidence score
                confidences = error.get('confidences', {})
                confidence = confidences.get(quantization, confidences.get(fallback_quantization, 0))

                # Write BED lines
                # Format: chrom, start, end, name
                hdc_name = f"{hdc_call}_conf_{confidence:.2f}"
                bam_name = f"{bam_call}_reference"

                f_hdc.write(f"{chrom}\t{start}\t{end}\t{hdc_name}\n")
                f_bam.write(f"{chrom}\t{start}\t{end}\t{bam_name}\n")

                valid_positions += 1

            except Exception as e:
                logger.warning(f"  Skipped position {error.get('position', 'unknown')}: {e}")
                skipped_positions += 1
                continue

    logger.info("")
    logger.info("BED File Generation Complete:")
    logger.info(f"  Valid positions: {valid_positions}")
    logger.info(f"  Skipped (parse errors): {skipped_positions}")
    logger.info(f"  Missing HDC calls: {missing_hdc_call}")
    logger.info("")
    logger.info(f"✓ GenomeVault predictions: {hdc_bed_path}")
    logger.info(f"✓ BAM ground truth:        {bam_bed_path}")
    logger.info("")
    logger.info("UCSC Genome Browser Upload:")
    logger.info(f"  1. Go to: https://genome.ucsc.edu/cgi-bin/hgCustom")
    logger.info(f"  2. Upload both BED files as custom tracks")
    logger.info(f"  3. Compare overlapping regions to identify collision sites")
    logger.info("")

    return hdc_bed_path, bam_bed_path


def main():
    """
    Command-line interface for BED file generation.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate collision BED files from HDV error analysis"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("genomevault/hdv_validation/results/bed_files"),
        help="Output directory for BED files"
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path("genomevault/hdv_validation/results/comparison_results"),
        help="Directory containing error JSON files"
    )
    parser.add_argument(
        '--quantization',
        type=str,
        default='float32',
        choices=['float32', 'int8', 'int4', 'binary'],
        help="Primary quantization for HDV predictions (default: float32)"
    )
    parser.add_argument(
        '--fallback',
        type=str,
        default='int8',
        choices=['float32', 'int8', 'int4', 'binary'],
        help="Fallback quantization (default: int8)"
    )
    parser.add_argument(
        '--high-precision',
        action='store_true',
        help="Generate BED files for high-precision-only errors (float32 ∩ int8)"
    )
    parser.add_argument(
        '--low-precision',
        action='store_true',
        help="Generate BED files for low-precision-only errors (int4 ∩ binary)"
    )
    parser.add_argument(
        '--common',
        action='store_true',
        help="Generate BED files for common errors (all quantizations)"
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help="Generate BED files for all error categories"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )

    # Determine which categories to generate
    generate_categories = []
    if args.all:
        generate_categories = ['high_precision', 'low_precision', 'common']
    else:
        if args.high_precision:
            generate_categories.append('high_precision')
        if args.low_precision:
            generate_categories.append('low_precision')
        if args.common:
            generate_categories.append('common')

    # If no flags specified, default to high-precision only
    if not generate_categories:
        generate_categories = ['high_precision']
        logger.info("No category specified, defaulting to --high-precision")
        logger.info("")

    try:
        for category in generate_categories:
            if category == 'high_precision':
                input_json = args.results_dir / "high_precision_errors.json"
                prefix = "high_precision"
            elif category == 'low_precision':
                input_json = args.results_dir / "low_precision_errors.json"
                prefix = "low_precision"
            elif category == 'common':
                input_json = args.results_dir / "common_errors.json"
                prefix = "common"

            generate_collision_beds(
                input_json=input_json,
                output_dir=args.output_dir,
                quantization=args.quantization,
                fallback_quantization=args.fallback,
                prefix=prefix
            )
    except Exception as e:
        logger.error(f"Failed to generate BED files: {e}")
        raise


if __name__ == "__main__":
    main()
