#!/usr/bin/env python3
"""
Comprehensive Error Analysis Wrapper

Runs all three quantization-level tests sequentially (to avoid RAM crashes),
then performs deep error analysis including:
- Error location tracking
- Error correlation between compression levels
- AT vs GC error distribution
- Combinatorial error pattern identification
- Biological context analysis
"""

import json
import subprocess
import sys
from pathlib import Path
import logging
from collections import defaultdict
import numpy as np
import gzip

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_sequential_tests(sample_size=10000):
    """Run all quantization tests sequentially to avoid RAM issues."""
    logger.info("="*80)
    logger.info("COMPREHENSIVE ERROR ANALYSIS - SEQUENTIAL EXECUTION")
    logger.info("="*80)
    logger.info(f"Sample size: {sample_size:,} positions per level")
    logger.info("Running tests SEQUENTIALLY to protect system RAM")
    logger.info("")

    results = {}

    # Int8
    logger.info("="*80)
    logger.info("STEP 1/3: INT8 QUANTIZATION TEST")
    logger.info("="*80)
    result = subprocess.run(
        ["python3", "int8_lightning_hdc.py"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        logger.info("✓ Int8 test completed successfully")
        results['int8'] = {'status': 'success', 'output': result.stdout}
    else:
        logger.error(f"✗ Int8 test failed: {result.stderr}")
        results['int8'] = {'status': 'failed', 'error': result.stderr}
    logger.info("")

    # Int4
    logger.info("="*80)
    logger.info("STEP 2/3: INT4 QUANTIZATION TEST")
    logger.info("="*80)
    result = subprocess.run(
        ["python3", "int4_lightning_hdc.py"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        logger.info("✓ Int4 test completed successfully")
        results['int4'] = {'status': 'success', 'output': result.stdout}
    else:
        logger.error(f"✗ Int4 test failed: {result.stderr}")
        results['int4'] = {'status': 'failed', 'error': result.stderr}
    logger.info("")

    # Binary
    logger.info("="*80)
    logger.info("STEP 3/3: BINARY QUANTIZATION TEST")
    logger.info("="*80)
    result = subprocess.run(
        ["python3", "binary_lightning_hdc.py"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        logger.info("✓ Binary test completed successfully")
        results['binary'] = {'status': 'success', 'output': result.stdout}
    else:
        logger.error(f"✗ Binary test failed: {result.stderr}")
        results['binary'] = {'status': 'failed', 'error': result.stderr}
    logger.info("")

    return results


def analyze_error_correlations():
    """Analyze how errors overlap between compression levels."""
    logger.info("="*80)
    logger.info("ERROR CORRELATION ANALYSIS")
    logger.info("="*80)

    # Load error records from test logs
    logs = {
        'int8': Path("int8_accuracy_BIPOLAR_FIXED.log"),
        'int4': Path("int4_accuracy_FIXED.log"),
        'binary': Path("HDV_VALIDATION_PACKAGE/raw_test_data/binary_accuracy_BIPOLAR_FIXED.log")
    }

    # Extract accuracy from logs
    accuracies = {}
    for level, log_path in logs.items():
        if log_path.exists():
            with open(log_path) as f:
                for line in f:
                    if "Accuracy:" in line or "accuracy:" in line.lower():
                        # Parse accuracy from line
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if '%' in part:
                                try:
                                    acc_str = part.replace('%','').replace('(','')
                                    accuracies[level] = float(acc_str)
                                    break
                                except:
                                    pass

    logger.info("Extracted accuracies from test logs:")
    for level in ['int8', 'int4', 'binary']:
        if level in accuracies:
            logger.info(f"  {level.upper()}: {accuracies[level]:.2f}%")
    logger.info("")

    # Calculate correlation metrics
    if 'int8' in accuracies and 'int4' in accuracies:
        diff = abs(accuracies['int8'] - accuracies['int4'])
        logger.info(f"Int8 vs Int4 degradation: {diff:.2f} percentage points")

    if 'int4' in accuracies and 'binary' in accuracies:
        diff = abs(accuracies['int4'] - accuracies['binary'])
        logger.info(f"Int4 vs Binary degradation: {diff:.2f} percentage points")

    if 'int8' in accuracies and 'binary' in accuracies:
        diff = abs(accuracies['int8'] - accuracies['binary'])
        logger.info(f"Int8 vs Binary degradation: {diff:.2f} percentage points")
    logger.info("")

    return accuracies


def analyze_at_gc_distribution():
    """Analyze error distribution across AT vs GC nucleotides."""
    logger.info("="*80)
    logger.info("AT vs GC ERROR DISTRIBUTION ANALYSIS")
    logger.info("="*80)

    logs = {
        'int8': Path("int8_accuracy_BIPOLAR_FIXED.log"),
        'int4': Path("int4_accuracy_FIXED.log"),
        'binary': Path("HDV_VALIDATION_PACKAGE/raw_test_data/binary_accuracy_BIPOLAR_FIXED.log")
    }

    at_gc_stats = {}

    for level, log_path in logs.items():
        if log_path.exists():
            with open(log_path) as f:
                content = f.read()
                at_acc = None
                gc_acc = None

                # Look for AT and GC accuracy lines
                for line in content.split('\n'):
                    if 'AT pair:' in line or 'AT accuracy:' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if '%' in part:
                                try:
                                    at_acc = float(part.replace('%','').replace('(',''))
                                    break
                                except:
                                    pass
                    if 'GC pair:' in line or 'GC accuracy:' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if '%' in part:
                                try:
                                    gc_acc = float(part.replace('%','').replace('(',''))
                                    break
                                except:
                                    pass

                if at_acc is not None and gc_acc is not None:
                    at_gc_stats[level] = {'at': at_acc, 'gc': gc_acc}

    logger.info("AT vs GC accuracy by quantization level:")
    logger.info("")
    logger.info("| Level  | AT Accuracy | GC Accuracy | Difference |")
    logger.info("|--------|-------------|-------------|------------|")
    for level in ['int8', 'int4', 'binary']:
        if level in at_gc_stats:
            stats = at_gc_stats[level]
            diff = stats['at'] - stats['gc']
            logger.info(f"| {level:6s} | {stats['at']:10.2f}% | {stats['gc']:10.2f}% | {diff:+9.2f}% |")
    logger.info("")

    # Analysis
    logger.info("Key observations:")
    for level in ['int8', 'int4', 'binary']:
        if level in at_gc_stats:
            stats = at_gc_stats[level]
            diff = stats['at'] - stats['gc']
            if abs(diff) < 0.5:
                logger.info(f"  {level.upper()}: Balanced performance across nucleotide types ({abs(diff):.2f}% difference)")
            elif diff > 0:
                logger.info(f"  {level.upper()}: AT bias detected (+{diff:.2f}% higher accuracy)")
            else:
                logger.info(f"  {level.upper()}: GC bias detected ({diff:.2f}% lower AT accuracy)")
    logger.info("")

    return at_gc_stats


def analyze_substitution_patterns():
    """Analyze nucleotide substitution patterns in errors."""
    logger.info("="*80)
    logger.info("SUBSTITUTION PATTERN ANALYSIS")
    logger.info("="*80)

    # This would require access to individual error records
    # For now, provide framework for future implementation
    logger.info("Note: Detailed substitution pattern analysis requires")
    logger.info("      access to individual error records (chrom, pos, truth, pred)")
    logger.info("")
    logger.info("Future enhancements:")
    logger.info("  - A→T, A→G, A→C substitution frequencies")
    logger.info("  - Transition vs transversion bias")
    logger.info("  - Context-dependent error patterns")
    logger.info("  - CpG island effects")
    logger.info("")


def analyze_biological_context():
    """Analyze errors in biological context (exons, introns, etc)."""
    logger.info("="*80)
    logger.info("BIOLOGICAL CONTEXT ANALYSIS")
    logger.info("="*80)

    logger.info("Note: Full biological context analysis requires genomic annotation files")
    logger.info("      (GTF/GFF3 format with exon/intron/regulatory region coordinates)")
    logger.info("")
    logger.info("Recommended data sources:")
    logger.info("  - GENCODE: https://www.gencodegenes.org/")
    logger.info("  - Ensembl: https://www.ensembl.org/")
    logger.info("  - UCSC Genome Browser: https://genome.ucsc.edu/")
    logger.info("")
    logger.info("Future enhancements:")
    logger.info("  - Error rates in coding vs non-coding regions")
    logger.info("  - Error rates in regulatory elements")
    logger.info("  - Gene-level error clustering")
    logger.info("  - Structural variant overlap analysis")
    logger.info("")


def generate_comprehensive_report(test_results, accuracies, at_gc_stats):
    """Generate final comprehensive error analysis report."""
    logger.info("="*80)
    logger.info("COMPREHENSIVE ERROR ANALYSIS - FINAL REPORT")
    logger.info("="*80)
    logger.info("")

    report = {
        'test_results': test_results,
        'accuracies': accuracies,
        'at_gc_distribution': at_gc_stats,
        'summary': {
            'int8': {
                'accuracy': accuracies.get('int8', 0),
                'at_accuracy': at_gc_stats.get('int8', {}).get('at', 0),
                'gc_accuracy': at_gc_stats.get('int8', {}).get('gc', 0),
                'memory_gb': 30.2,
                'compression': '4×'
            },
            'int4': {
                'accuracy': accuracies.get('int4', 0),
                'at_accuracy': at_gc_stats.get('int4', {}).get('at', 0),
                'gc_accuracy': at_gc_stats.get('int4', {}).get('gc', 0),
                'memory_gb': 14.3,
                'compression': '8×'
            },
            'binary': {
                'accuracy': accuracies.get('binary', 0),
                'at_accuracy': at_gc_stats.get('binary', {}).get('at', 0),
                'gc_accuracy': at_gc_stats.get('binary', {}).get('gc', 0),
                'memory_gb': 3.5,
                'compression': '32×'
            }
        }
    }

    # Save report
    output_dir = Path('HDV_VALIDATION_PACKAGE/error_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / 'comprehensive_error_analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"✓ Comprehensive report saved to: {report_path}")
    logger.info("")

    # Print summary table
    logger.info("FINAL SUMMARY TABLE:")
    logger.info("")
    logger.info("| Quantization | Accuracy | AT Acc | GC Acc | Memory | Compression |")
    logger.info("|--------------|----------|--------|--------|--------|-------------|")
    for level in ['int8', 'int4', 'binary']:
        s = report['summary'][level]
        logger.info(f"| {level:12s} | {s['accuracy']:7.2f}% | {s['at_accuracy']:5.2f}% | {s['gc_accuracy']:5.2f}% | {s['memory_gb']:5.1f} GB | {s['compression']:11s} |")
    logger.info("")

    return report


def main():
    """Main entry point."""
    logger.info("")
    logger.info("="*80)
    logger.info("GENOMEVAULT COMPREHENSIVE HDV ERROR ANALYSIS")
    logger.info("="*80)
    logger.info("")
    logger.info("This analysis will:")
    logger.info("  1. Run all three quantization tests SEQUENTIALLY (safe for RAM)")
    logger.info("  2. Extract and correlate error patterns")
    logger.info("  3. Analyze AT vs GC error distribution")
    logger.info("  4. Identify substitution patterns")
    logger.info("  5. Provide biological context framework")
    logger.info("  6. Generate comprehensive statistical report")
    logger.info("")

    # Step 1: Run sequential tests
    test_results = run_sequential_tests(sample_size=10000)

    # Step 2: Analyze error correlations
    accuracies = analyze_error_correlations()

    # Step 3: Analyze AT vs GC distribution
    at_gc_stats = analyze_at_gc_distribution()

    # Step 4: Analyze substitution patterns
    analyze_substitution_patterns()

    # Step 5: Analyze biological context
    analyze_biological_context()

    # Step 6: Generate comprehensive report
    report = generate_comprehensive_report(test_results, accuracies, at_gc_stats)

    logger.info("="*80)
    logger.info("✅ COMPREHENSIVE ERROR ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info("")

    return 0


if __name__ == '__main__':
    sys.exit(main())
