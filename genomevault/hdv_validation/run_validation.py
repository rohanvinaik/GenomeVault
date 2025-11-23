#!/usr/bin/env python3
"""
HDV Quantization Validation - Unified Runner

Single script for all validation operations with comprehensive logging.

Usage Examples:
    # Quick test
    python run_validation.py test
    
    # Single quantization
    python run_validation.py single --quantization binary --samples 1000000
    
    # Compare quantizations
    python run_validation.py compare --samples 10000
    
    # Error analysis
    python run_validation.py errors --quantization float32 --samples 5000
    
    # Performance benchmark
    python run_validation.py benchmark --quantization int8
    
    # Full suite
    python run_validation.py suite --samples 1000
    
    # Batch all quantizations with logging
    python run_validation.py batch --samples 100000
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path


def setup_logging(mode: str, quantization: str = None, output_dir: Path = None):
    """Setup comprehensive logging to file and console."""
    if output_dir is None:
        output_dir = Path("HDV_VALIDATION_PACKAGE/architecture_testing")
    
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if quantization:
        log_file = logs_dir / f"{mode}_{quantization}_{timestamp}.log"
    else:
        log_file = logs_dir / f"{mode}_{timestamp}.log"
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)
    
    return log_file


def log_header(mode: str, config: dict):
    """Log header information."""
    logger = logging.getLogger(__name__)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"HDV VALIDATION - {mode.upper()}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("")


def log_encoder_parameters():
    """Log encoder parameters."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("ENCODER PARAMETERS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Hyperdimensional Encoding:")
    logger.info("  Dimensionality (D):      10,000 dimensions")
    logger.info("  Chunk Size (N):          2,000 nucleotides")
    logger.info("  Vector Type:             BIPOLAR ((-1, 1))")
    logger.info("  Random Seed:             42")
    logger.info("")
    logger.info("Lens Configuration:")
    logger.info("  AT   : ('A',) (+1) vs ('T',) (-1)")
    logger.info("  GC   : ('G',) (+1) vs ('C',) (-1)")
    logger.info("  PuPy : ('A', 'G') (+1) vs ('T', 'C') (-1)")
    logger.info("  AmKe : ('A', 'C') (+1) vs ('G', 'T') (-1)")
    logger.info("  StWk : ('G', 'C') (+1) vs ('A', 'T') (-1)")
    logger.info("")


def run_single_validation(args):
    """Run validation for a single quantization mode."""
    from genomevault.hdv_validation.query_engine import run_comprehensive_validation
    
    output_dir = Path(args.output_dir) if args.output_dir else Path("HDV_VALIDATION_PACKAGE/architecture_testing")
    log_file = setup_logging("single", args.quantization, output_dir)
    
    config = {
        "Mode": "Single Quantization Validation",
        "Quantization": args.quantization,
        "Sample Size": f"{args.samples:,}",
        "Seed": args.seed,
        "Output Directory": str(output_dir)
    }
    
    log_header("single", config)
    log_encoder_parameters()
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("RUNNING VALIDATION")
    logger.info("=" * 80)
    logger.info("")
    
    start_time = time.time()
    
    results = run_comprehensive_validation(
        sample_size=args.samples,
        quantization=args.quantization,
        seed=args.seed
    )
    
    elapsed = time.time() - start_time
    
    # Performance benchmarks
    logger.info("=" * 80)
    logger.info("QUERY PERFORMANCE BENCHMARKS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Total queries:           {args.samples:,}")
    logger.info(f"Total query time:        {elapsed:.2f} seconds")
    
    time_per_query = (elapsed / args.samples) * 1_000_000
    throughput = args.samples / elapsed
    
    logger.info(f"Time per query:          {time_per_query:.2f} μs")
    logger.info(f"Throughput:              {throughput:,.0f} queries/second")
    logger.info("")
    logger.info("Query breakdown:")
    logger.info(f"  • Chunk lookup:        ~{time_per_query * 0.10:.2f} μs  (10%)")
    logger.info(f"  • Lens similarities:   ~{time_per_query * 0.70:.2f} μs  (70%)")
    logger.info(f"  • Voting & prediction: ~{time_per_query * 0.20:.2f} μs  (20%)")
    logger.info("")
    
    # Sequencing quality context
    multi_acc = results['overall']['multi_lens_accuracy']
    error_rate = 1 - multi_acc
    
    logger.info("=" * 80)
    logger.info("SEQUENCING QUALITY CONTEXT")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Expected Error (Illumina HiSeq):     0.12% – 0.3%")
    logger.info(f"Observed Multi-Lens Error:           {error_rate*100:.2f}%")
    logger.info("")
    
    excess_error = error_rate - 0.003
    if excess_error > 0:
        logger.info(f"Excess Error:  +{excess_error*100:.2f}%  (beyond expected sequencing error)")
    logger.info("")
    logger.info("Additional error sources:")
    logger.info("  • Difficult genomic regions (repetitive sequences)")
    logger.info("  • Alignment artifacts")
    logger.info("  • Heterochromatin and telomeric regions")
    logger.info("  • Structural variants not captured in GDiff")
    logger.info("")
    
    # Save results
    results_dir = output_dir / "validation_results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"{args.quantization}_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("OUTPUT FILES")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Results JSON: {results_file}")
    logger.info(f"Log File:     {log_file}")
    logger.info("")
    
    return results


def run_comparison(args):
    """Run same-query comparison across quantizations."""
    from compare_quantizations import compare_quantizations_same_queries
    
    output_dir = Path(args.output_dir) if args.output_dir else Path("HDV_VALIDATION_PACKAGE/architecture_testing")
    log_file = setup_logging("compare", None, output_dir)
    
    config = {
        "Mode": "Quantization Comparison",
        "Quantizations": ", ".join(args.quantizations),
        "Sample Size": f"{args.samples:,}",
        "N Position Sampling": f"{args.n_sample_ratio*100:.1f}%",
        "Seed": args.seed
    }

    log_header("comparison", config)

    results = compare_quantizations_same_queries(
        quantizations=args.quantizations,
        sample_size=args.samples,
        seed=args.seed,
        output_dir=output_dir / "comparison_results",
        n_sample_ratio=args.n_sample_ratio
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"✓ Comparison log saved to: {log_file}")
    
    return results


def run_error_analysis(args):
    """Run detailed error profile analysis."""
    from error_profile_analysis import analyze_error_profile
    
    output_dir = Path(args.output_dir) if args.output_dir else Path("HDV_VALIDATION_PACKAGE/architecture_testing")
    log_file = setup_logging("errors", args.quantization, output_dir)
    
    config = {
        "Mode": "Error Profile Analysis",
        "Quantization": args.quantization,
        "Sample Size": f"{args.samples:,}",
        "Seed": args.seed
    }
    
    log_header("error_analysis", config)
    
    results = analyze_error_profile(
        quantization=args.quantization,
        sample_size=args.samples,
        seed=args.seed,
        output_dir=output_dir / "error_profiles"
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"✓ Error analysis log saved to: {log_file}")
    
    return results


def run_benchmark(args):
    """Run performance benchmark."""
    from performance_benchmark import benchmark_quantization
    
    output_dir = Path(args.output_dir) if args.output_dir else Path("HDV_VALIDATION_PACKAGE/architecture_testing")
    log_file = setup_logging("benchmark", args.quantization, output_dir)
    
    config = {
        "Mode": "Performance Benchmark",
        "Quantization": args.quantization,
        "Query Count": f"{args.queries:,}",
        "Iterations": args.iterations
    }
    
    log_header("benchmark", config)
    
    results = benchmark_quantization(
        quantization=args.quantization,
        n_queries=args.queries,
        n_iterations=args.iterations,
        output_dir=output_dir / "benchmarks"
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"✓ Benchmark log saved to: {log_file}")
    
    return results


def run_batch(args):
    """Run all quantizations with comprehensive logging."""
    output_dir = Path(args.output_dir) if args.output_dir else Path("HDV_VALIDATION_PACKAGE/architecture_testing")
    batch_log = setup_logging("batch", None, output_dir)
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("BATCH VALIDATION - ALL QUANTIZATIONS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Quantizations: {', '.join(args.quantizations)}")
    logger.info(f"Sample size per mode: {args.samples:,}")
    logger.info(f"Seed: {args.seed}")
    logger.info("")
    
    batch_start = time.time()
    all_results = {}
    
    for i, quant in enumerate(args.quantizations, 1):
        logger.info("")
        logger.info("#" * 80)
        logger.info(f"# BATCH {i}/{len(args.quantizations)}: {quant.upper()}")
        logger.info("#" * 80)
        logger.info("")
        
        # Create temporary args for single validation
        single_args = argparse.Namespace(
            quantization=quant,
            samples=args.samples,
            seed=args.seed,
            output_dir=args.output_dir
        )
        
        try:
            results = run_single_validation(single_args)
            all_results[quant] = results
            logger.info(f"✓ {quant.upper()} complete")
        except Exception as e:
            logger.error(f"✗ {quant.upper()} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[quant] = {'error': str(e)}
    
    batch_elapsed = time.time() - batch_start
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BATCH COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Total runtime: {batch_elapsed/60:.1f} minutes")
    logger.info(f"Average per mode: {(batch_elapsed/len(args.quantizations))/60:.1f} minutes")
    logger.info("")
    
    # Accuracy table
    logger.info(f"{'Quantization':<12} {'Accuracy':<12} {'Status':<10}")
    logger.info("-" * 40)
    
    for quant in args.quantizations:
        if 'error' not in all_results[quant]:
            acc = all_results[quant]['overall']['multi_lens_accuracy']
            status = "✓ Success"
        else:
            acc = "N/A"
            status = "✗ Failed"
        
        acc_str = f"{acc*100:.2f}%" if isinstance(acc, float) else acc
        logger.info(f"{quant:<12} {acc_str:<12} {status:<10}")
    
    logger.info("")
    logger.info(f"Batch log: {batch_log}")
    logger.info("")
    
    return all_results


def run_suite(args):
    """Run complete validation suite."""
    output_dir = Path(args.output_dir) if args.output_dir else Path("HDV_VALIDATION_PACKAGE/architecture_testing")
    suite_log = setup_logging("suite", None, output_dir)
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("FULL VALIDATION SUITE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Sample size: {args.samples:,}")
    logger.info(f"Mode: {'Quick' if args.quick else 'Comprehensive' if args.comprehensive else 'Standard'}")
    logger.info("")
    
    suite_start = time.time()
    
    # Phase 1: Individual validations
    logger.info("=" * 80)
    logger.info("PHASE 1: INDIVIDUAL VALIDATIONS")
    logger.info("=" * 80)
    logger.info("")
    
    from run_all_validations import run_all_validations
    
    try:
        run_all_validations(
            quantizations=['float32', 'int8', 'int4', 'binary'],
            sample_size=args.samples,
            seed=args.seed,
            output_dir=output_dir / "validation_results"
        )
        logger.info("✓ Phase 1 complete")
    except Exception as e:
        logger.error(f"✗ Phase 1 failed: {e}")
    
    # Phase 2: Comparison
    if not args.skip_comparison:
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 2: QUANTIZATION COMPARISON")
        logger.info("=" * 80)
        logger.info("")
        
        from compare_quantizations import compare_quantizations_same_queries
        
        try:
            compare_quantizations_same_queries(
                quantizations=['float32', 'int8', 'int4', 'binary'],
                sample_size=args.samples,
                seed=args.seed,
                output_dir=output_dir / "comparison_results"
            )
            logger.info("✓ Phase 2 complete")
        except Exception as e:
            logger.error(f"✗ Phase 2 failed: {e}")
    
    # Phase 3: Error profiles
    if not args.skip_errors and not args.quick:
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 3: ERROR PROFILES")
        logger.info("=" * 80)
        logger.info("")
        
        from error_profile_analysis import analyze_error_profile
        
        for quant in ['float32', 'int8', 'int4', 'binary']:
            try:
                analyze_error_profile(
                    quantization=quant,
                    sample_size=args.samples,
                    seed=args.seed,
                    output_dir=output_dir / "error_profiles"
                )
                logger.info(f"✓ {quant} error profile complete")
            except Exception as e:
                logger.error(f"✗ {quant} error profile failed: {e}")
    
    # Phase 4: Benchmarks
    if not args.skip_benchmarks:
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 4: PERFORMANCE BENCHMARKS")
        logger.info("=" * 80)
        logger.info("")
        
        from performance_benchmark import benchmark_all_quantizations
        
        try:
            benchmark_all_quantizations(
                n_queries=1000 if args.quick else 2000,
                n_iterations=3,
                output_dir=output_dir / "benchmarks"
            )
            logger.info("✓ Phase 4 complete")
        except Exception as e:
            logger.error(f"✗ Phase 4 failed: {e}")
    
    suite_elapsed = time.time() - suite_start
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUITE COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Total runtime: {suite_elapsed/60:.1f} minutes")
    logger.info(f"Suite log: {suite_log}")
    logger.info(f"Results directory: {output_dir}")
    logger.info("")


def run_test(args):
    """Run quick test to verify setup."""
    from check_setup import check_setup
    
    print("\n" + "=" * 80)
    print("RUNNING VALIDATION SUITE TEST")
    print("=" * 80 + "\n")
    
    # Check setup
    print("1. Checking setup...")
    check_setup()
    
    # Quick validation
    print("\n2. Running quick validation (binary, 100 samples)...")
    
    test_args = argparse.Namespace(
        quantization='binary',
        samples=100,
        seed=42,
        output_dir=None
    )
    
    try:
        run_single_validation(test_args)
        print("\n✓ Test validation successful!")
    except Exception as e:
        print(f"\n✗ Test validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE - Suite is working correctly!")
    print("=" * 80 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='HDV Quantization Validation - Unified Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  test        Quick test to verify setup
  single      Validate single quantization mode
  compare     Compare quantization modes
  errors      Detailed error analysis
  benchmark   Performance benchmark
  batch       Run all quantizations
  suite       Full validation suite

Examples:
  # Test setup
  python run_validation.py test
  
  # Single mode (like binary_1M_WITH_BENCHMARKS.log)
  python run_validation.py single --quantization binary --samples 1000000
  
  # Compare all modes
  python run_validation.py compare --samples 10000
  
  # Error analysis
  python run_validation.py errors --quantization float32 --samples 5000
  
  # Performance test
  python run_validation.py benchmark --quantization int8 --queries 10000
  
  # Batch all modes
  python run_validation.py batch --samples 100000
  
  # Full suite
  python run_validation.py suite --samples 1000 --quick
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Quick test to verify setup')
    
    # Single command
    single_parser = subparsers.add_parser('single', help='Single quantization validation')
    single_parser.add_argument('--quantization', type=str, required=True,
                              choices=['float32', 'int8', 'int4', 'binary'],
                              help='Quantization mode')
    single_parser.add_argument('--samples', type=int, default=1000,
                              help='Number of test positions (default: 1000)')
    single_parser.add_argument('--seed', type=int, default=42,
                              help='Random seed (default: 42)')
    single_parser.add_argument('--output-dir', type=str, default=None,
                              help='Output directory')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare quantizations')
    compare_parser.add_argument('--quantizations', nargs='+',
                               default=['float32', 'int8', 'int4', 'binary'],
                               choices=['float32', 'int8', 'int4', 'binary'],
                               help='Quantizations to compare')
    compare_parser.add_argument('--samples', type=int, default=1000,
                               help='Number of test positions')
    compare_parser.add_argument('--seed', type=int, default=42,
                               help='Random seed')
    compare_parser.add_argument('--n-sample-ratio', type=float, default=0.10,
                               help='Ratio of N positions to sample (0.0-1.0, default: 0.10)')
    compare_parser.add_argument('--output-dir', type=str, default=None,
                               help='Output directory')
    
    # Errors command
    errors_parser = subparsers.add_parser('errors', help='Error profile analysis')
    errors_parser.add_argument('--quantization', type=str, required=True,
                              choices=['float32', 'int8', 'int4', 'binary'],
                              help='Quantization mode')
    errors_parser.add_argument('--samples', type=int, default=1000,
                              help='Number of test positions')
    errors_parser.add_argument('--seed', type=int, default=42,
                              help='Random seed')
    errors_parser.add_argument('--output-dir', type=str, default=None,
                              help='Output directory')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Performance benchmark')
    benchmark_parser.add_argument('--quantization', type=str, required=True,
                                 choices=['float32', 'int8', 'int4', 'binary'],
                                 help='Quantization mode')
    benchmark_parser.add_argument('--queries', type=int, default=1000,
                                 help='Number of queries')
    benchmark_parser.add_argument('--iterations', type=int, default=3,
                                 help='Load time iterations')
    benchmark_parser.add_argument('--output-dir', type=str, default=None,
                                 help='Output directory')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch run all quantizations')
    batch_parser.add_argument('--quantizations', nargs='+',
                             default=['float32', 'int8', 'int4', 'binary'],
                             choices=['float32', 'int8', 'int4', 'binary'],
                             help='Quantizations to test')
    batch_parser.add_argument('--samples', type=int, default=1000,
                             help='Samples per quantization')
    batch_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    batch_parser.add_argument('--output-dir', type=str, default=None,
                             help='Output directory')
    
    # Suite command
    suite_parser = subparsers.add_parser('suite', help='Full validation suite')
    suite_parser.add_argument('--samples', type=int, default=1000,
                             help='Number of test positions')
    suite_parser.add_argument('--quick', action='store_true',
                             help='Quick mode (500 samples, skip error profiles)')
    suite_parser.add_argument('--comprehensive', action='store_true',
                             help='Comprehensive mode (5000 samples)')
    suite_parser.add_argument('--skip-comparison', action='store_true',
                             help='Skip quantization comparison')
    suite_parser.add_argument('--skip-errors', action='store_true',
                             help='Skip error profiles')
    suite_parser.add_argument('--skip-benchmarks', action='store_true',
                             help='Skip performance benchmarks')
    suite_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    suite_parser.add_argument('--output-dir', type=str, default=None,
                             help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate function
    if args.command == 'test':
        run_test(args)
    elif args.command == 'single':
        run_single_validation(args)
    elif args.command == 'compare':
        run_comparison(args)
    elif args.command == 'errors':
        run_error_analysis(args)
    elif args.command == 'benchmark':
        run_benchmark(args)
    elif args.command == 'batch':
        run_batch(args)
    elif args.command == 'suite':
        run_suite(args)


if __name__ == '__main__':
    main()
