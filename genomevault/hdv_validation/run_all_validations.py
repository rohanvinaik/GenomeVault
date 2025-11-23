#!/usr/bin/env python3
"""
Main orchestrator for comprehensive HDV validation across all quantization levels.

This script runs the full validation suite for each quantization mode and generates
comprehensive reports.

Usage:
    python run_all_validations.py --sample-size 1000 --seed 42
    python run_all_validations.py --quantizations float32 int8 --sample-size 5000
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path

from genomevault.hdv_validation.query_engine import run_comprehensive_validation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def run_all_validations(
    quantizations=['float32', 'int8', 'int4', 'binary'],
    sample_size=1000,
    seed=42,
    output_dir=None
):
    """
    Run comprehensive validation for all specified quantization levels.
    
    Args:
        quantizations: List of quantization modes to test
        sample_size: Number of test positions
        seed: Random seed for reproducibility
        output_dir: Output directory for results
    """
    if output_dir is None:
        base_dir = Path("HDV_VALIDATION_PACKAGE/architecture_testing")
        output_dir = base_dir / "validation_results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = output_dir.parent
    
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE HDV VALIDATION SUITE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Quantization modes: {', '.join(quantizations)}")
    logger.info(f"Sample size: {sample_size:,}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    # Store results for each quantization
    all_results = {}
    timing_results = {}
    
    for i, quant in enumerate(quantizations, 1):
        logger.info("=" * 80)
        logger.info(f"VALIDATION {i}/{len(quantizations)}: {quant.upper()}")
        logger.info("=" * 80)
        logger.info("")
        
        start_time = time.time()
        
        try:
            # Run validation
            results = run_comprehensive_validation(
                sample_size=sample_size,
                quantization=quant
            )
            
            elapsed_time = time.time() - start_time
            
            # Store results
            all_results[quant] = results
            timing_results[quant] = {
                'total_time_seconds': elapsed_time,
                'time_per_query_ms': (elapsed_time / sample_size) * 1000,
                'queries_per_second': sample_size / elapsed_time
            }
            
            logger.info("")
            logger.info(f"✓ Validation completed in {elapsed_time:.2f}s")
            logger.info(f"  Time per query: {timing_results[quant]['time_per_query_ms']:.3f}ms")
            logger.info(f"  Queries per second: {timing_results[quant]['queries_per_second']:.1f}")
            logger.info("")
            
            # Save individual results
            output_file = output_dir / f"{quant}_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"✓ Individual results saved to: {output_file}")
            logger.info("")
            
        except Exception as e:
            logger.error(f"✗ Validation failed for {quant}: {e}")
            import traceback
            traceback.print_exc()
            all_results[quant] = {'error': str(e)}
            timing_results[quant] = {'error': str(e)}
    
    # Generate summary report
    logger.info("=" * 80)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("=" * 80)
    logger.info("")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'quantizations': quantizations,
            'sample_size': sample_size,
            'seed': seed
        },
        'timing': timing_results,
        'accuracy_summary': {}
    }
    
    # Extract key metrics
    for quant, results in all_results.items():
        if 'error' in results:
            summary['accuracy_summary'][quant] = {'error': results['error']}
            continue
        
        summary['accuracy_summary'][quant] = {
            'overall_accuracy': results['overall']['multi_lens_accuracy'],
            'observed_accuracy': results['observed_vs_theoretical']['observed_accuracy'],
            'theoretical_accuracy': results['observed_vs_theoretical'].get('validated_theoretical_accuracy', 0),
            'combined_theoretical_accuracy': results['observed_vs_theoretical'].get('combined_theoretical_accuracy', 0),
            'per_nucleotide': {
                nuc: stats['multi_lens_accuracy']
                for nuc, stats in results['per_nucleotide'].items()
            },
            'per_lens': {
                lens: stats['accuracy']
                for lens, stats in results['per_lens'].items()
            }
        }
    
    # Save summary to base directory (root)
    summary_file = base_dir / "validation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✓ Summary saved to: {summary_file}")
    logger.info("")
    
    # Print summary table
    logger.info("=" * 80)
    logger.info("ACCURACY COMPARISON")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"{'Quantization':<12} {'Overall':<10} {'Observed':<10} {'Theoretical':<12} {'Time/Query':<12}")
    logger.info("-" * 80)
    
    for quant in quantizations:
        if quant not in all_results or 'error' in all_results[quant]:
            logger.info(f"{quant:<12} {'ERROR':<10}")
            continue
        
        acc_sum = summary['accuracy_summary'][quant]
        timing = timing_results[quant]
        
        logger.info(
            f"{quant:<12} "
            f"{acc_sum['overall_accuracy']*100:>9.2f}% "
            f"{acc_sum['observed_accuracy']*100:>9.2f}% "
            f"{acc_sum['theoretical_accuracy']*100:>11.2f}% "
            f"{timing['time_per_query_ms']:>11.3f}ms"
        )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATION SUITE COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    
    return all_results, summary


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive HDV validation across all quantization levels'
    )
    parser.add_argument(
        '--quantizations',
        nargs='+',
        default=['float32', 'int8', 'int4', 'binary'],
        choices=['float32', 'int8', 'int4', 'binary'],
        help='Quantization modes to test (default: all)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Number of test positions (default: 1000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: HDV_VALIDATION_PACKAGE/architecture_testing)'
    )
    
    args = parser.parse_args()
    
    run_all_validations(
        quantizations=args.quantizations,
        sample_size=args.sample_size,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
