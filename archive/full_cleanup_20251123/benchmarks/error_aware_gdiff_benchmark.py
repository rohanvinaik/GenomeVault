"""
Error-Aware GDiff Benchmark

Demonstrates complete pipeline with clinical-grade error tracking:
1. Quality assessment from FASTQ input
2. GDiff encoding with error bounds
3. Multi-run consensus for different use cases
4. Error reporting and recommendations

Based on Decision Matrix V2.0, Sections 7.3, 8, and 11.

Usage:
    python benchmarks/error_aware_gdiff_benchmark.py --use-case screening
    python benchmarks/error_aware_gdiff_benchmark.py --use-case diagnostic
    python benchmarks/error_aware_gdiff_benchmark.py --use-case life_critical
    python benchmarks/error_aware_gdiff_benchmark.py --all-use-cases
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add genomevault to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.differential_encoding.gdiff.schema import ErrorBounds
from genomevault.differential_encoding.gdiff.error_reporting import (
    generate_error_report,
    format_error_report,
    compute_epsilon_query_multirun,
    CLINICAL_THRESHOLDS
)
from genomevault.query.multi_run_consensus import (
    compute_multi_run_confidence,
    get_recommended_runs_for_use_case,
    print_use_case_summary,
    USE_CASE_PRESETS
)
from genomevault.quality_control import (
    compute_min_input_quality,
    validate_input_quality
)


def simulate_quality_assessment(Q_input: float) -> Dict[str, Any]:
    """
    Simulate quality assessment from FASTQ input.

    In production, this would parse actual FASTQ Q-scores.
    For benchmarking, we simulate different quality levels.

    Args:
        Q_input: Simulated input quality (0-1)

    Returns:
        Quality report dict
    """
    epsilon_input = 1 - Q_input

    return {
        'Q_input': Q_input,
        'epsilon_input': epsilon_input,
        'mean_phred_score': -10 * (Q_input - 1),  # Approximate Phred score
        'sequencing_platform': _infer_platform(Q_input),
        'total_reads': 1_000_000,
        'total_bases': 150_000_000
    }


def _infer_platform(Q_input: float) -> str:
    """Infer sequencing platform from quality."""
    if Q_input >= 0.999:
        return "PacBio HiFi"
    elif Q_input >= 0.99:
        return "Illumina NovaSeq X+"
    elif Q_input >= 0.95:
        return "Illumina HiSeq X"
    elif Q_input >= 0.90:
        return "Oxford Nanopore Q20+"
    elif Q_input >= 0.70:
        return "Ion Torrent"
    else:
        return "Low-quality platform"


def compute_pipeline_error(k: int = 3, D: int = 10000) -> float:
    """
    Compute GenomeVault pipeline error.

    Args:
        k: k-anonymity level
        D: Hypervector dimension

    Returns:
        epsilon_pipeline
    """
    # Component fidelities from Decision Matrix V2.0, Section 7.3
    F_gdiff = 0.999      # GDiff encoding fidelity
    F_hdc = 0.9999       # HDC transformation fidelity
    F_zk = 1 - 2**-128   # ZK proof soundness (essentially 1.0)
    F_pir = 1.0          # PIR correctness (information-theoretic)

    F_pipeline = F_gdiff * F_hdc * F_zk * F_pir
    epsilon_pipeline = 1 - F_pipeline

    return epsilon_pipeline


def run_benchmark_for_use_case(
    use_case: str,
    Q_input: float,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run complete benchmark for a clinical use case.

    Args:
        use_case: Clinical use case (screening, diagnostic, life_critical, regulatory)
        Q_input: Input quality level
        output_dir: Output directory for results

    Returns:
        Benchmark results dict
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {use_case.upper()} USE CASE")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Get clinical threshold
    threshold_info = CLINICAL_THRESHOLDS[use_case]
    target_epsilon = threshold_info['max_total_error']

    print(f"Target error: {target_epsilon:.4f} ({target_epsilon*100:.2f}%)")
    print(f"Required confidence: {threshold_info['min_confidence']:.6f}")
    print(f"Recommended runs: {threshold_info['recommended_runs']}")
    print()

    # Step 1: Quality Assessment
    print("STEP 1: Quality Assessment")
    print("-" * 80)
    quality_report = simulate_quality_assessment(Q_input)
    print(f"  Input quality: Q_input = {quality_report['Q_input']:.4f}")
    print(f"  Platform: {quality_report['sequencing_platform']}")
    print(f"  Error rate: ε_input = {quality_report['epsilon_input']:.4f} ({quality_report['epsilon_input']*100:.2f}%)")
    print()

    # Step 2: Compute Minimum Quality Requirement
    print("STEP 2: Minimum Quality Requirement")
    print("-" * 80)
    try:
        min_quality_info = compute_min_input_quality(target_epsilon, k=3, D=10000)
        print(f"  Minimum required: Q_input_min = {min_quality_info['Q_input_min']:.4f}")
        print(f"  Recommendation: {min_quality_info['sequencing_recommendation']}")

        # Check if input quality meets requirement
        if Q_input >= min_quality_info['Q_input_min']:
            print(f"  ✓ Input quality MEETS requirement")
        else:
            print(f"  ✗ Input quality BELOW requirement (gap: {min_quality_info['Q_input_min'] - Q_input:.4f})")
    except ValueError as e:
        print(f"  ⚠️  Cannot meet target: {e}")
        min_quality_info = None
    print()

    # Step 3: Pipeline Error
    print("STEP 3: Pipeline Error Computation")
    print("-" * 80)
    epsilon_pipeline = compute_pipeline_error(k=3, D=10000)
    print(f"  ε_pipeline = {epsilon_pipeline:.6f} ({epsilon_pipeline*100:.4f}%)")
    print(f"  Components:")
    print(f"    F_gdiff  = 0.999  (99.9%)")
    print(f"    F_hdc    = 0.9999 (99.99%)")
    print(f"    F_zk     = 1-2^-128 (essentially 1.0)")
    print(f"    F_pir    = 1.0 (information-theoretic)")
    print()

    # Step 4: Multi-Run Consensus
    print("STEP 4: Multi-Run Consensus")
    print("-" * 80)
    n_runs = get_recommended_runs_for_use_case(use_case)
    consensus_stats = compute_multi_run_confidence(n_runs)
    epsilon_query = consensus_stats['epsilon_query']

    print(f"  Runs: n = {n_runs}")
    print(f"  Confidence: {consensus_stats['confidence']:.8f} ({consensus_stats['confidence']*100:.6f}%)")
    print(f"  ε_query: {epsilon_query:.10f}")
    print(f"  Query time: {consensus_stats['query_time_seconds']:.2f}s")
    print(f"  Privacy cost: {consensus_stats['privacy_cost_bits']:.2f} bits (k=3)")
    print()

    # Step 5: Total Error Computation
    print("STEP 5: Total Error Computation")
    print("-" * 80)
    epsilon_input = quality_report['epsilon_input']
    epsilon_total = epsilon_input + epsilon_pipeline + epsilon_query

    print(f"  ε_input    = {epsilon_input:.6f} ({epsilon_input*100:.4f}%)")
    print(f"  ε_pipeline = {epsilon_pipeline:.6f} ({epsilon_pipeline*100:.4f}%)")
    print(f"  ε_query    = {epsilon_query:.10f} ({epsilon_query*100:.8f}%)")
    print(f"  {'─'*40}")
    print(f"  ε_total    = {epsilon_total:.6f} ({epsilon_total*100:.4f}%)")
    print()

    # Check if meets target
    meets_target = epsilon_total <= target_epsilon
    if meets_target:
        margin = target_epsilon - epsilon_total
        margin_pct = (margin / target_epsilon) * 100
        print(f"  ✅ PASS: Within target ({margin:.6f} margin, {margin_pct:.1f}%)")
    else:
        excess = epsilon_total - target_epsilon
        excess_pct = (excess / target_epsilon) * 100
        print(f"  ❌ FAIL: Exceeds target by {excess:.6f} ({excess_pct:.1f}%)")
    print()

    # Step 6: Error Bounds Creation
    print("STEP 6: Error Bounds and Reporting")
    print("-" * 80)
    error_bounds = ErrorBounds(
        epsilon_input_corrected=epsilon_input,
        epsilon_pipeline=epsilon_pipeline,
        epsilon_query=epsilon_query,
        epsilon_total=epsilon_total,
        Q_input_measured=Q_input,
        use_case=use_case,
        meets_target=meets_target
    )

    # Generate detailed error report
    error_report = generate_error_report(error_bounds, detailed=True)

    # Print summary
    print(f"  Status: {error_report['clinical_assessment']['status']}")
    if 'recommendations' in error_report:
        print(f"  Recommendations: {len(error_report['recommendations'])} actions")
        for i, rec in enumerate(error_report['recommendations'], 1):
            print(f"    {i}. [{rec['priority']}] {rec['category']}")
    else:
        print(f"  No recommendations needed - meets target")
    print()

    # Save detailed report
    use_case_dir = output_dir / use_case
    use_case_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    report_json_path = use_case_dir / "error_report.json"
    with open(report_json_path, 'w') as f:
        json.dump(error_report, f, indent=2)
    print(f"  ✓ Saved JSON report: {report_json_path}")

    # Save markdown report
    report_md = format_error_report(error_report, markdown=True)
    report_md_path = use_case_dir / "error_report.md"
    with open(report_md_path, 'w') as f:
        f.write(report_md)
    print(f"  ✓ Saved Markdown report: {report_md_path}")

    elapsed_time = time.time() - start_time
    print(f"\nBenchmark completed in {elapsed_time:.2f}s")

    # Build result summary
    result = {
        'use_case': use_case,
        'timestamp': time.time(),
        'input': {
            'Q_input': Q_input,
            'platform': quality_report['sequencing_platform'],
            'epsilon_input': epsilon_input
        },
        'pipeline': {
            'epsilon_pipeline': epsilon_pipeline,
            'k_anonymity': 3,
            'hypervector_dimension': 10000
        },
        'consensus': {
            'n_runs': n_runs,
            'confidence': consensus_stats['confidence'],
            'epsilon_query': epsilon_query,
            'query_time_seconds': consensus_stats['query_time_seconds'],
            'privacy_cost_bits': consensus_stats['privacy_cost_bits']
        },
        'total': {
            'epsilon_total': epsilon_total,
            'target_epsilon': target_epsilon,
            'meets_target': meets_target,
            'status': 'PASS' if meets_target else 'FAIL'
        },
        'error_bounds': {
            'epsilon_input_corrected': error_bounds.epsilon_input_corrected,
            'epsilon_pipeline': error_bounds.epsilon_pipeline,
            'epsilon_query': error_bounds.epsilon_query,
            'epsilon_total': error_bounds.epsilon_total,
            'Q_input_measured': error_bounds.Q_input_measured
        },
        'recommendations': error_report.get('recommendations', []),
        'elapsed_time_seconds': elapsed_time
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Error-Aware GDiff Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--use-case',
        choices=['screening', 'diagnostic', 'life_critical', 'regulatory'],
        help='Run benchmark for specific use case'
    )
    parser.add_argument(
        '--all-use-cases',
        action='store_true',
        help='Run benchmark for all 4 use cases'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('benchmark_results/error_aware_gdiff'),
        help='Output directory (default: benchmark_results/error_aware_gdiff)'
    )
    parser.add_argument(
        '--quality',
        type=float,
        help='Override input quality (0-1). If not specified, uses recommended quality for use case.'
    )
    parser.add_argument(
        '--print-summary',
        action='store_true',
        help='Print use case summary table'
    )

    args = parser.parse_args()

    # Print summary if requested
    if args.print_summary:
        print_use_case_summary()
        return

    # Determine which use cases to run
    if args.all_use_cases:
        use_cases_to_run = ['screening', 'diagnostic', 'life_critical', 'regulatory']
    elif args.use_case:
        use_cases_to_run = [args.use_case]
    else:
        print("Error: Must specify --use-case or --all-use-cases")
        print("Use --help for usage information")
        return

    # Recommended quality levels for each use case
    recommended_quality = {
        'screening': 0.70,        # Low quality acceptable
        'diagnostic': 0.95,       # High quality required
        'life_critical': 0.999,   # Ultra-high quality required
        'regulatory': 0.9999      # Extreme quality required
    }

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    all_results = []
    for use_case in use_cases_to_run:
        # Determine quality level
        if args.quality is not None:
            Q_input = args.quality
        else:
            Q_input = recommended_quality[use_case]

        result = run_benchmark_for_use_case(use_case, Q_input, args.output_dir)
        all_results.append(result)

    # Save summary
    summary_path = args.output_dir / "benchmark_summary.json"
    summary = {
        'timestamp': time.time(),
        'num_use_cases': len(all_results),
        'results': all_results
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total use cases: {len(all_results)}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary: {summary_path}")
    print()

    # Print table
    print(f"{'Use Case':<20} {'Q_input':>10} {'ε_total':>12} {'Target':>12} {'Status':>10}")
    print("-" * 80)
    for result in all_results:
        use_case = result['use_case']
        Q_input = result['input']['Q_input']
        epsilon_total = result['total']['epsilon_total']
        target = result['total']['target_epsilon']
        status = result['total']['status']

        print(f"{use_case:<20} {Q_input:>10.4f} {epsilon_total:>12.6f} {target:>12.6f} {status:>10}")
    print()


if __name__ == "__main__":
    main()
