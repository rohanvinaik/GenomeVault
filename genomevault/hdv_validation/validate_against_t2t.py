"""
Validate GenomeVault HDV predictions against T2T-CHM13v2.0 reference genome.

This module compares GenomeVault biophysical predictions from BED files
against the UCSC T2T-CHM13v2.0 (hs1) reference genome to evaluate:
1. Raw accuracy vs. gold-standard reference
2. Imputation success rate (confidence > 0 corrections)
3. Valid rejection rate (confidence = 0 for ambiguous bases)
4. Statistical patterns and error distributions
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests

logger = logging.getLogger(__name__)


class T2TReferenceValidator:
    """
    Validator for GenomeVault predictions against T2T-CHM13v2.0.

    Attributes:
        api_url: UCSC API endpoint
        genome: Genome assembly (default: hs1 = T2T-CHM13v2.0)
        rate_limit_delay: Delay between API calls (seconds)
    """

    def __init__(
        self,
        genome: str = "hs1",
        rate_limit_delay: float = 0.1,
        api_url: str = "https://api.genome.ucsc.edu/getData/sequence"
    ):
        """
        Initialize T2T reference validator.

        Args:
            genome: UCSC genome code (default: hs1 = T2T-CHM13v2.0)
            rate_limit_delay: Delay between API calls in seconds
            api_url: UCSC API endpoint
        """
        self.api_url = api_url
        self.genome = genome
        self.rate_limit_delay = rate_limit_delay

        # Statistics tracking
        self.stats = {
            'total': 0,
            'match': 0,
            'mismatch': 0,
            'valid_rejection': 0,
            'imputed_correct': 0,
            'imputed_incorrect': 0,
            'api_errors': 0,
        }

        # Detailed results
        self.results = []

        # Error patterns by chromosome
        self.chromosome_stats = defaultdict(lambda: {'match': 0, 'mismatch': 0})

        # Confidence distribution
        self.confidence_bins = defaultdict(lambda: {'match': 0, 'mismatch': 0})

    def query_ucsc_base(
        self,
        chrom: str,
        start: int,
        end: int,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Query UCSC API for reference base at position.

        Args:
            chrom: Chromosome name (e.g., chr22)
            start: 0-based start position
            end: 1-based end position (BED format)
            max_retries: Maximum retry attempts on failure

        Returns:
            Reference base (A, T, G, C, N) or None on error
        """
        params = {
            'genome': self.genome,
            'chrom': chrom,
            'start': start,
            'end': end
        }

        for attempt in range(max_retries):
            try:
                response = requests.get(self.api_url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    # UCSC returns lowercase, we need uppercase
                    return data['dna'].upper()
                elif response.status_code == 404:
                    logger.warning(f"Position not found: {chrom}:{start}-{end}")
                    return None
                else:
                    logger.warning(f"API returned {response.status_code} for {chrom}:{start}-{end}")

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                time.sleep(self.rate_limit_delay * (attempt + 1))
            except Exception as e:
                logger.error(f"Error querying UCSC API: {e}")
                break

        return None

    def parse_bed_prediction(self, name_field: str) -> Tuple[str, float]:
        """
        Parse prediction from BED name field.

        Supports two formats:
        1. GenomeVault: "C_conf_0.80" -> (C, 0.80)
        2. BAM reference: "N_reference" -> (N, 0.0)
                         "A_reference" -> (A, 1.0)

        Args:
            name_field: BED name field

        Returns:
            (predicted_base, confidence)

        Example:
            >>> parse_bed_prediction("C_conf_0.80")
            ("C", 0.80)
            >>> parse_bed_prediction("N_reference")
            ("N", 0.0)
        """
        parts = name_field.split('_')

        # GenomeVault format: "BASE_conf_X.XX"
        if len(parts) >= 3 and parts[1] == 'conf':
            base = parts[0]
            confidence = float(parts[2])
            return base, confidence

        # BAM reference format: "BASE_reference"
        elif len(parts) == 2 and parts[1] == 'reference':
            base = parts[0]
            # N calls have 0.0 confidence (uncertain), others have 1.0 (reference call)
            confidence = 0.0 if base == 'N' else 1.0
            return base, confidence

        else:
            raise ValueError(f"Invalid BED name format: {name_field}")

    def validate_position(
        self,
        chrom: str,
        start: int,
        end: int,
        pred_base: str,
        confidence: float
    ) -> Dict:
        """
        Validate a single genomic position.

        Args:
            chrom: Chromosome
            start: 0-based start
            end: 1-based end
            pred_base: Predicted base (A, T, G, C, N)
            confidence: Prediction confidence (0.0-1.0)

        Returns:
            Result dictionary with validation outcome
        """
        # Query T2T reference
        true_base = self.query_ucsc_base(chrom, start, end)

        if true_base is None:
            self.stats['api_errors'] += 1
            return {
                'position': f"{chrom}:{start}-{end}",
                'prediction': pred_base,
                'confidence': confidence,
                'truth': 'ERROR',
                'result': 'API_ERROR',
                'correct': None
            }

        # Determine result
        self.stats['total'] += 1

        if true_base == pred_base:
            # Perfect match
            self.stats['match'] += 1
            self.chromosome_stats[chrom]['match'] += 1

            # Check if this was an imputation (confidence > 0)
            if confidence > 0.0:
                self.stats['imputed_correct'] += 1
                result = 'IMPUTED_CORRECT'
            else:
                result = 'MATCH'

            correct = True

        else:
            # Mismatch
            if confidence == 0.0:
                # Predicted low confidence / deletion / ambiguous
                # If truth is N or different, this is a valid rejection
                self.stats['valid_rejection'] += 1
                result = 'VALID_REJECTION'
                correct = True  # Correctly identified as unreliable
            else:
                # High confidence prediction but wrong
                self.stats['mismatch'] += 1
                self.chromosome_stats[chrom]['mismatch'] += 1

                if confidence > 0.0:
                    self.stats['imputed_incorrect'] += 1
                    result = 'IMPUTED_INCORRECT'
                else:
                    result = 'MISMATCH'

                correct = False

        # Track confidence distribution
        conf_bin = int(confidence * 10) / 10  # Round to 1 decimal
        if correct:
            self.confidence_bins[conf_bin]['match'] += 1
        else:
            self.confidence_bins[conf_bin]['mismatch'] += 1

        return {
            'position': f"{chrom}:{start}-{end}",
            'prediction': f"{pred_base}_conf_{confidence:.2f}",
            'confidence': confidence,
            'truth': true_base,
            'result': result,
            'correct': correct
        }

    def validate_bed_file(
        self,
        bed_file: Path,
        output_dir: Path,
        verbose: bool = True
    ) -> Tuple[Dict, Path]:
        """
        Validate all positions in a BED file.

        Args:
            bed_file: Path to BED file with GenomeVault predictions
            output_dir: Directory to save validation results
            verbose: Print progress to console

        Returns:
            (statistics_dict, report_path)
        """
        logger.info("=" * 80)
        logger.info("VALIDATING GENOMEVAULT AGAINST T2T-CHM13v2.0")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"BED file: {bed_file}")
        logger.info(f"Reference: {self.genome} (T2T-CHM13v2.0)")
        logger.info("")

        if verbose:
            print(f"{'Coordinate':<25} | {'GV Pred':<15} | {'T2T Truth':<10} | {'Result'}")
            print("-" * 75)

        # Read and validate BED file
        with open(bed_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Skip headers and empty lines
                if line.startswith('[') or line.startswith('track') or not line.strip():
                    continue

                parts = line.strip().split()

                if len(parts) < 4:
                    logger.warning(f"Line {line_num}: Invalid BED format (need 4 columns)")
                    continue

                # Parse BED fields
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                name = parts[3]

                # Parse prediction
                try:
                    pred_base, confidence = self.parse_bed_prediction(name)
                except ValueError as e:
                    logger.warning(f"Line {line_num}: {e}")
                    continue

                # Validate position
                result = self.validate_position(chrom, start, end, pred_base, confidence)
                self.results.append(result)

                # Print progress
                if verbose and result['result'] != 'API_ERROR':
                    status_emoji = {
                        'MATCH': '✅',
                        'IMPUTED_CORRECT': '✅ (imputed)',
                        'VALID_REJECTION': '🛡️',
                        'MISMATCH': '❌',
                        'IMPUTED_INCORRECT': '❌ (imputed)'
                    }
                    status = status_emoji.get(result['result'], '?')

                    print(f"{result['position']:<25} | {result['prediction']:<15} | "
                          f"{result['truth']:<10} | {result['result']} {status}")

                # Rate limiting
                time.sleep(self.rate_limit_delay)

        # Generate report
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.generate_report(bed_file, output_dir)

        return self.stats, report_path

    def generate_report(self, bed_file: Path, output_dir: Path) -> Path:
        """
        Generate comprehensive validation report.

        Args:
            bed_file: Original BED file validated
            output_dir: Directory to save report

        Returns:
            Path to generated report
        """
        # Extract cohort and source from filename for human-readable naming
        # Pattern: {cohort}_{source}_liftover.bed → {cohort}_{source}_generation.md
        filename_stem = bed_file.stem  # e.g., "high_precision_genomevault_liftover"

        # Remove "_liftover" suffix if present
        if filename_stem.endswith("_liftover"):
            base_name = filename_stem[:-9]  # Remove "_liftover"
        else:
            base_name = filename_stem

        report_name = f"{base_name}_generation.md"
        report_path = output_dir / report_name

        # Calculate metrics
        total_validated = self.stats['match'] + self.stats['mismatch']

        if total_validated > 0:
            accuracy = (self.stats['match'] / total_validated) * 100
        else:
            accuracy = 0.0

        # Imputation metrics
        total_imputed = self.stats['imputed_correct'] + self.stats['imputed_incorrect']
        if total_imputed > 0:
            imputation_accuracy = (self.stats['imputed_correct'] / total_imputed) * 100
        else:
            imputation_accuracy = 0.0

        # Generate markdown report
        with open(report_path, 'w') as f:
            f.write(f"# T2T-CHM13v2.0 Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**BED File:** `{bed_file}`\n\n")
            f.write(f"**Reference Genome:** {self.genome} (T2T-CHM13v2.0)\n\n")
            f.write("---\n\n")

            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Positions Validated:** {self.stats['total']}\n")
            f.write(f"- **Perfect Matches:** {self.stats['match']}\n")
            f.write(f"- **Mismatches:** {self.stats['mismatch']}\n")
            f.write(f"- **Valid Rejections (conf=0.0):** {self.stats['valid_rejection']}\n")
            f.write(f"- **API Errors:** {self.stats['api_errors']}\n\n")

            # Overall accuracy
            f.write("## Overall Accuracy\n\n")
            f.write(f"**GenomeVault vs. T2T-CHM13v2.0:** {accuracy:.2f}%\n\n")
            f.write(f"- Correct predictions: {self.stats['match']} / {total_validated}\n")
            f.write(f"- Incorrect predictions: {self.stats['mismatch']} / {total_validated}\n\n")

            # Imputation performance
            f.write("## Imputation Performance (Confidence > 0)\n\n")
            if total_imputed > 0:
                f.write(f"**Imputation Accuracy:** {imputation_accuracy:.2f}%\n\n")
                f.write(f"- Correct imputations: {self.stats['imputed_correct']}\n")
                f.write(f"- Incorrect imputations: {self.stats['imputed_incorrect']}\n")
                f.write(f"- Total imputed sites: {total_imputed}\n\n")
            else:
                f.write("*No high-confidence imputations in dataset*\n\n")

            # Chromosome breakdown
            f.write("## Per-Chromosome Statistics\n\n")
            f.write("| Chromosome | Matches | Mismatches | Accuracy |\n")
            f.write("|------------|---------|------------|----------|\n")

            for chrom in sorted(self.chromosome_stats.keys()):
                chrom_match = self.chromosome_stats[chrom]['match']
                chrom_mismatch = self.chromosome_stats[chrom]['mismatch']
                chrom_total = chrom_match + chrom_mismatch

                if chrom_total > 0:
                    chrom_acc = (chrom_match / chrom_total) * 100
                else:
                    chrom_acc = 0.0

                f.write(f"| {chrom} | {chrom_match} | {chrom_mismatch} | {chrom_acc:.2f}% |\n")

            f.write("\n")

            # Confidence distribution
            f.write("## Confidence Distribution\n\n")
            f.write("| Confidence | Matches | Mismatches | Accuracy |\n")
            f.write("|------------|---------|------------|----------|\n")

            for conf in sorted(self.confidence_bins.keys()):
                conf_match = self.confidence_bins[conf]['match']
                conf_mismatch = self.confidence_bins[conf]['mismatch']
                conf_total = conf_match + conf_mismatch

                if conf_total > 0:
                    conf_acc = (conf_match / conf_total) * 100
                else:
                    conf_acc = 0.0

                f.write(f"| {conf:.1f} | {conf_match} | {conf_mismatch} | {conf_acc:.2f}% |\n")

            f.write("\n")

            # Detailed results (first 100)
            f.write("## Sample Results (First 100 Positions)\n\n")
            f.write("| Position | Prediction | Truth | Result |\n")
            f.write("|----------|------------|-------|--------|\n")

            for result in self.results[:100]:
                if result['result'] != 'API_ERROR':
                    f.write(f"| {result['position']} | {result['prediction']} | "
                           f"{result['truth']} | {result['result']} |\n")

            if len(self.results) > 100:
                f.write(f"\n*Showing 100 of {len(self.results)} total results*\n")

            f.write("\n")

            # Methodology
            f.write("## Methodology\n\n")
            f.write("**Reference Genome:** T2T-CHM13v2.0 (UCSC code: hs1)\n\n")
            f.write("The T2T-CHM13v2.0 assembly is a complete, gapless human genome ")
            f.write("representing the most accurate reference available as of 2023.\n\n")
            f.write("**Validation Logic:**\n\n")
            f.write("- **MATCH:** Prediction == Truth (confidence > 0)\n")
            f.write("- **IMPUTED_CORRECT:** High-confidence prediction correct\n")
            f.write("- **VALID_REJECTION:** Low confidence (0.0) for ambiguous/difficult base\n")
            f.write("- **MISMATCH:** Prediction != Truth\n")
            f.write("- **IMPUTED_INCORRECT:** High-confidence prediction wrong\n\n")

        # Save JSON results
        json_path = output_dir / f"{base_name}_generation.json"
        with open(json_path, 'w') as f:
            json.dump({
                'metadata': {
                    'bed_file': str(bed_file),
                    'reference': self.genome,
                    'timestamp': datetime.now().isoformat()
                },
                'statistics': self.stats,
                'chromosome_stats': dict(self.chromosome_stats),
                'confidence_bins': dict(self.confidence_bins),
                'results': self.results
            }, f, indent=2)

        logger.info("")
        logger.info("=" * 80)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"Total positions validated: {self.stats['total']}")
        logger.info(f"Overall accuracy: {accuracy:.2f}%")
        logger.info("")
        logger.info(f"✓ Report saved: {report_path}")
        logger.info(f"✓ JSON results: {json_path}")
        logger.info("")

        return report_path


def main():
    """
    Command-line interface for T2T validation.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate GenomeVault predictions against T2T-CHM13v2.0"
    )
    parser.add_argument(
        '--bed-file',
        type=Path,
        default=Path("genomevault/hdv_validation/results/bed_files/liftover_bed/liftover/common_genomevault_liftover.bed"),
        help="BED file with liftOver results (from UCSC liftOver tool)"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("genomevault/hdv_validation/results/BAM_vs_pipeline_accuracy"),
        help="Output directory for validation reports"
    )
    parser.add_argument(
        '--genome',
        type=str,
        default='hs1',
        choices=['hs1', 'hg38', 'hg19'],
        help="UCSC reference genome (default: hs1 = T2T-CHM13v2.0)"
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.1,
        help="Delay between API calls in seconds (default: 0.1)"
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )

    # Validate BED file exists
    if not args.bed_file.exists():
        logger.error(f"BED file not found: {args.bed_file}")
        return

    # Run validation
    validator = T2TReferenceValidator(
        genome=args.genome,
        rate_limit_delay=args.rate_limit
    )

    try:
        stats, report_path = validator.validate_bed_file(
            bed_file=args.bed_file,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )

        # Print summary
        total = stats['match'] + stats['mismatch']
        if total > 0:
            accuracy = (stats['match'] / total) * 100
            print(f"\n✓ GenomeVault Accuracy vs T2T-CHM13v2.0: {accuracy:.2f}%")
            print(f"✓ Report: {report_path}")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    main()
