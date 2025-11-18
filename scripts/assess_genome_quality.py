#!/usr/bin/env python3
"""
Genome Quality Assessment CLI Tool

Assess FASTQ quality and provide clinical-grade recommendations for GenomeVault use.

Based on Accuracy-Efficiency-Privacy Decision Matrix V2.0
(Section 7: Clinical Error Bounds and Decision Rules)

Usage:
    python scripts/assess_genome_quality.py \\
        --fastq path/to/genome.fastq.gz \\
        --use-case diagnostic \\
        --output quality_report.json

Privacy: All analysis is LOCAL (no network calls).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add genomevault to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.quality_control import (
    validate_input_quality,
    compute_min_input_quality,
    select_optimal_configuration_clinical,
    ERROR_BOUNDS_CLINICAL,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def assess_genome_quality(
    fastq_path: str,
    use_case: str,
    output_path: str = None,
    k: int = 3,
    D: int = 10000
):
    """
    Assess genome quality and provide recommendations.

    Args:
        fastq_path: Path to FASTQ file (.fastq or .fastq.gz)
        use_case: Clinical use case (screening, diagnostic, life_critical, regulatory)
        output_path: Optional path to save JSON report
        k: k-anonymity level (default: 3)
        D: Hypervector dimension (default: 10000)
    """
    logger.info("=" * 80)
    logger.info("GenomeVault Genome Quality Assessment")
    logger.info("=" * 80)
    logger.info(f"FASTQ: {fastq_path}")
    logger.info(f"Use case: {use_case}")
    logger.info(f"Configuration: k={k}, D={D}")
    logger.info("")

    # Get target error bound for use case
    if use_case not in ERROR_BOUNDS_CLINICAL:
        logger.error(f"Invalid use case '{use_case}'")
        logger.error(f"Valid use cases: {list(ERROR_BOUNDS_CLINICAL.keys())}")
        sys.exit(1)

    target_epsilon = ERROR_BOUNDS_CLINICAL[use_case]['max_total_error']
    min_confidence = ERROR_BOUNDS_CLINICAL[use_case]['min_confidence']
    recommended_runs = ERROR_BOUNDS_CLINICAL[use_case]['recommended_runs']

    logger.info(f"Clinical Requirements for '{use_case}':")
    logger.info(f"  Maximum total error (ε_max): {target_epsilon:.4f} ({target_epsilon*100:.1f}%)")
    logger.info(f"  Minimum confidence: {min_confidence:.4f} ({min_confidence*100:.2f}%)")
    logger.info(f"  Recommended runs: {recommended_runs}")
    logger.info("")

    # Step 1: Compute minimum required quality
    logger.info("Step 1: Computing minimum quality requirements...")
    min_quality_info = compute_min_input_quality(target_epsilon, k, D)

    logger.info(f"  Required Q_input_min: {min_quality_info['Q_input_min']:.4f} "
                f"({min_quality_info['Q_input_min']*100:.2f}%)")
    logger.info(f"  Error budget breakdown:")
    logger.info(f"    - Input (sequencing): {min_quality_info['epsilon_breakdown']['input_max']:.4f}")
    logger.info(f"    - Pipeline (processing): {min_quality_info['epsilon_breakdown']['pipeline']:.4f}")
    logger.info(f"    - Query (single run): {min_quality_info['epsilon_breakdown']['query']:.4f}")
    logger.info(f"  Recommended platform: {min_quality_info['sequencing_recommendation']}")
    logger.info("")

    # Step 2: Validate actual input quality
    logger.info("Step 2: Assessing input FASTQ quality...")
    logger.info("  (Sampling 100,000 reads...)")

    try:
        quality_report = validate_input_quality(
            fastq_path=fastq_path,
            target_epsilon=target_epsilon,
            k=k,
            D=D
        )
    except Exception as e:
        logger.error(f"Failed to assess FASTQ quality: {e}")
        sys.exit(1)

    logger.info(f"  Measured Q_input: {quality_report['Q_input']:.4f} "
                f"({quality_report['Q_input']*100:.2f}%)")
    logger.info(f"  Sequencing error rate: {quality_report['epsilon_input']:.4f} "
                f"({quality_report['epsilon_input']*100:.2f}%)")

    metrics = quality_report['quality_metrics']
    logger.info(f"  Quality metrics:")
    logger.info(f"    - Average Q-score: {metrics['average_q_score']:.1f}")
    logger.info(f"    - Median Q-score: {metrics['median_q_score']:.1f}")
    logger.info(f"    - Q30 fraction: {metrics['q30_fraction']:.2%}")
    logger.info(f"    - Coverage uniformity: {metrics['coverage_uniformity']:.2f}")
    logger.info(f"    - Bases sampled: {metrics['total_bases_sampled']:,}")
    logger.info("")

    # Step 3: Quality verdict
    logger.info("Step 3: Quality Verdict")
    logger.info("-" * 80)

    if quality_report['meets_target']:
        logger.info(f"✅ PASS: Input quality is SUFFICIENT for '{use_case}' use case")
        logger.info(f"   Q_input ({quality_report['Q_input']:.4f}) ≥ Q_min ({quality_report['Q_input_min']:.4f})")
        verdict = "PASS"
    else:
        logger.warning(f"❌ FAIL: Input quality is INSUFFICIENT for '{use_case}' use case")
        logger.warning(f"   Q_input ({quality_report['Q_input']:.4f}) < Q_min ({quality_report['Q_input_min']:.4f})")
        logger.warning(f"   Recommendation: {quality_report['recommendation']}")
        verdict = "FAIL"

    logger.info("")

    # Step 4: Optimal configuration (if quality is sufficient)
    config_report = None
    if quality_report['meets_target']:
        logger.info("Step 4: Selecting optimal GenomeVault configuration...")
        try:
            config_report = select_optimal_configuration_clinical(
                use_case=use_case,
                epsilon_max=target_epsilon,
                Q_input=quality_report['Q_input'],
                compute_budget_hours=10.0,
                storage_budget_mb=100.0
            )

            logger.info(f"  Optimal configuration:")
            logger.info(f"    - k (anonymity): {config_report['configuration']['k']}")
            logger.info(f"    - D (dimension): {config_report['configuration']['D']}")
            logger.info(f"    - B (batch size): {config_report['configuration']['B']}")

            logger.info(f"  Expected performance:")
            logger.info(f"    - Total error (ε_total): {config_report['error_bounds']['epsilon_total']:.4f}")
            logger.info(f"    - Meets requirement: {config_report['error_bounds']['meets_requirement']}")
            logger.info(f"    - Privacy (P): {config_report['performance']['privacy']:.3f}")
            logger.info(f"    - Query time: {config_report['performance']['query_time_seconds']:.2f}s")
            logger.info(f"    - Setup time: {config_report['performance']['setup_time_hours']:.1f} hours")

            logger.info(f"  Recommendations:")
            logger.info(f"    - Recommended runs: {config_report['recommendations']['recommended_runs']}")

        except Exception as e:
            logger.error(f"Failed to select optimal configuration: {e}")

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"VERDICT: {verdict}")
    logger.info("=" * 80)

    # Compile full report
    full_report = {
        'verdict': verdict,
        'use_case': use_case,
        'fastq_path': str(fastq_path),
        'target_requirements': {
            'epsilon_max': target_epsilon,
            'min_confidence': min_confidence,
            'recommended_runs': recommended_runs,
        },
        'minimum_quality_requirements': min_quality_info,
        'measured_quality': quality_report,
        'optimal_configuration': config_report,
    }

    # Save report if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(full_report, f, indent=2)

        logger.info(f"Quality report saved to: {output_path}")
        logger.info("")

    return full_report


def main():
    parser = argparse.ArgumentParser(
        description="Assess genome quality for GenomeVault clinical use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Assess for diagnostic use (5% error tolerance)
  python scripts/assess_genome_quality.py \\
      --fastq data/patient_genome.fastq.gz \\
      --use-case diagnostic \\
      --output reports/quality_report.json

  # Assess for life-critical use (0.1% error tolerance)
  python scripts/assess_genome_quality.py \\
      --fastq data/emergency_genome.fastq.gz \\
      --use-case life_critical

Use Cases:
  - screening: 30% error (any platform)
  - diagnostic: 5% error (NovaSeq X+ recommended)
  - life_critical: 0.1% error (PacBio HiFi required)
  - regulatory: 0.01% error (multiple platforms + consensus)

Privacy: All analysis is LOCAL (no network calls).
        """
    )

    parser.add_argument(
        '--fastq',
        type=str,
        required=True,
        help='Path to FASTQ file (.fastq or .fastq.gz)'
    )

    parser.add_argument(
        '--use-case',
        type=str,
        required=True,
        choices=['screening', 'diagnostic', 'life_critical', 'regulatory'],
        help='Clinical use case'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save JSON quality report (optional)'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='k-anonymity level (default: 3)'
    )

    parser.add_argument(
        '--dimension',
        type=int,
        default=10000,
        dest='D',
        help='Hypervector dimension (default: 10000)'
    )

    args = parser.parse_args()

    # Validate FASTQ exists
    fastq_path = Path(args.fastq)
    if not fastq_path.exists():
        logger.error(f"FASTQ file not found: {fastq_path}")
        sys.exit(1)

    # Run assessment
    try:
        assess_genome_quality(
            fastq_path=str(fastq_path),
            use_case=args.use_case,
            output_path=args.output,
            k=args.k,
            D=args.D
        )
    except KeyboardInterrupt:
        logger.info("\nAssessment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Assessment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
