"""
Clean BED files for UCSC Genome Browser compatibility.

This script processes BED files from liftOver output and creates
UCSC-compatible 4-column BED files by:
1. Removing header lines (track, [], etc.)
2. Removing empty lines
3. Extracting only 4 columns: chrom, start, end, name
4. Organizing into raw/ and ucsc_cleaned/ subdirectories
"""

import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


def clean_bed_file(input_path: Path, output_path: Path) -> int:
    """
    Clean a single BED file for UCSC compatibility.

    Args:
        input_path: Path to input BED file
        output_path: Path to output cleaned BED file

    Returns:
        Number of valid lines written

    Removes:
        - Lines starting with '[' (liftOver annotations)
        - Lines starting with 'track' (track headers)
        - Empty lines
        - Extra columns beyond first 4 (chrom, start, end, name)
    """
    valid_lines = 0

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            # Skip headers or empty lines
            if line.startswith('[') or line.startswith('track') or not line.strip():
                continue

            # Split by whitespace (handles tabs or spaces)
            parts = line.strip().split()

            # We need exactly 4 columns: Chrom, Start, End, Name
            if len(parts) >= 4:
                chrom = parts[0]
                start = parts[1]
                end = parts[2]
                name = parts[3]

                # Write tab-separated 4-column BED
                f_out.write(f"{chrom}\t{start}\t{end}\t{name}\n")
                valid_lines += 1

    return valid_lines


def clean_bed_directory(
    input_dir: Path,
    create_subdirs: bool = True
) -> None:
    """
    Clean all BED files in a directory for UCSC compatibility.

    Args:
        input_dir: Directory containing BED files to clean
        create_subdirs: If True, create raw/ and ucsc_cleaned/ subdirectories
                       and organize files accordingly. If False, clean files
                       in place with '_cleaned' suffix.

    Structure created (if create_subdirs=True):
        input_dir/
        ├── raw/                    # Original files (backup)
        │   ├── file1.bed
        │   └── file2.bed
        └── ucsc_cleaned/           # Cleaned files (UCSC-ready)
            ├── file1.bed
            └── file2.bed
    """
    logger.info("=" * 80)
    logger.info("CLEANING BED FILES FOR UCSC GENOME BROWSER")
    logger.info("=" * 80)
    logger.info("")

    # Validate input directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all BED files
    bed_files = list(input_dir.glob("*.bed"))

    if not bed_files:
        logger.warning(f"No BED files found in {input_dir}")
        return

    logger.info(f"Found {len(bed_files)} BED file(s) to clean:")
    for bed_file in bed_files:
        logger.info(f"  - {bed_file.name}")
    logger.info("")

    if create_subdirs:
        # Create subdirectories
        raw_dir = input_dir / "raw"
        cleaned_dir = input_dir / "ucsc_cleaned"

        raw_dir.mkdir(exist_ok=True)
        cleaned_dir.mkdir(exist_ok=True)

        logger.info(f"Created subdirectories:")
        logger.info(f"  Raw files: {raw_dir}")
        logger.info(f"  Cleaned files: {cleaned_dir}")
        logger.info("")

        # Process each BED file
        for bed_file in bed_files:
            logger.info(f"Processing: {bed_file.name}")

            # Copy to raw/ (backup)
            raw_path = raw_dir / bed_file.name
            shutil.copy2(bed_file, raw_path)
            logger.info(f"  ✓ Backed up to: raw/{bed_file.name}")

            # Clean and save to ucsc_cleaned/
            cleaned_path = cleaned_dir / bed_file.name
            valid_lines = clean_bed_file(bed_file, cleaned_path)
            logger.info(f"  ✓ Cleaned: {valid_lines} valid lines → ucsc_cleaned/{bed_file.name}")
            logger.info("")

    else:
        # Clean in place with _cleaned suffix
        for bed_file in bed_files:
            logger.info(f"Processing: {bed_file.name}")

            cleaned_path = bed_file.parent / f"{bed_file.stem}_cleaned.bed"
            valid_lines = clean_bed_file(bed_file, cleaned_path)
            logger.info(f"  ✓ Cleaned: {valid_lines} valid lines → {cleaned_path.name}")
            logger.info("")

    logger.info("=" * 80)
    logger.info("CLEANING COMPLETE")
    logger.info("=" * 80)
    logger.info("")

    if create_subdirs:
        logger.info(f"✓ Original files backed up in: {raw_dir}")
        logger.info(f"✓ UCSC-ready files saved in: {cleaned_dir}")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. Navigate to: {cleaned_dir}")
        logger.info("  2. Copy contents of cleaned BED files")
        logger.info("  3. Paste into UCSC 'Define Regions' box at:")
        logger.info("     https://genome.ucsc.edu/cgi-bin/hgTables")
    else:
        logger.info("✓ Cleaned files saved with '_cleaned' suffix")

    logger.info("")


def main():
    """
    Command-line interface for BED file cleaning.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean BED files for UCSC Genome Browser compatibility"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path("genomevault/hdv_validation/results/bed_files/liftover_bed"),
        help="Directory containing BED files to clean"
    )
    parser.add_argument(
        '--no-subdirs',
        action='store_true',
        help="Don't create subdirectories, clean in place with '_cleaned' suffix"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    try:
        clean_bed_directory(
            input_dir=args.input_dir,
            create_subdirs=not args.no_subdirs
        )
    except Exception as e:
        logger.error(f"Error cleaning BED files: {e}")
        raise


if __name__ == "__main__":
    main()
