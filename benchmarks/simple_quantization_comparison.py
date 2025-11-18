#!/usr/bin/env python3
"""
Simple Quantization Comparison - Uses existing proven implementations

Just wraps the individual tests we already ran and adds statistical analysis.
"""

import json
import gzip
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ground_truth(gdiff_path: Path):
    """Load ground truth from GDiff."""
    logger.info(f"Loading ground truth from {gdiff_path}")
    with gzip.open(gdiff_path, 'rt') as f:
        gdiff = json.load(f)

    variants = gdiff['differential_variants']
    logger.info(f"  ✓ Loaded {len(variants):,} variants")
    return variants


def select_test_positions(variants, n=5000, seed=42):
    """Select random test positions stratified by AT vs GC."""
    np.random.seed(seed)

    at_variants = [v for v in variants if v['alt'] in ['A', 'T']]
    gc_variants = [v for v in variants if v['alt'] in ['G', 'C']]

    n_at = len(at_variants)
    n_gc = len(gc_variants)
    total = n_at + n_gc

    # Proportional sampling
    n_at_sample = int(n * n_at / total)
    n_gc_sample = n - n_at_sample

    at_sample = np.random.choice(at_variants, size=n_at_sample, replace=False).tolist()
    gc_sample = np.random.choice(gc_variants, size=n_gc_sample, replace=False).tolist()

    test_positions = at_sample + gc_sample
    np.random.shuffle(test_positions)

    logger.info(f"  ✓ Selected {len(test_positions):,} test positions ({n_at_sample} AT, {n_gc_sample} GC)")
    return test_positions


def query_quantization_level(hdc_class, encoded_genome_path, test_positions):
    """Query a quantization level and return results."""
    logger.info(f"Loading {hdc_class.__name__}...")
    hdc = hdc_class(encoded_genome_path)
    hdc.load()

    logger.info(f"Querying {len(test_positions):,} positions...")

    results = {
        'correct': 0,
        'total': 0,
        'at_correct': 0,
        'at_total': 0,
        'gc_correct': 0,
        'gc_total': 0,
        'errors': []
    }

    for i, v in enumerate(test_positions):
        if (i + 1) % 1000 == 0:
            logger.info(f"  Progress: {i+1:,}/{len(test_positions):,}")

        try:
            pred, conf = hdc.query_nucleotide(v['chrom'], v['pos'])
            truth = v['alt']

            if pred == truth:
                results['correct'] += 1
                if truth in ['A', 'T']:
                    results['at_correct'] += 1
            else:
                results['errors'].append({
                    'chrom': v['chrom'],
                    'pos': v['pos'],
                    'truth': truth,
                    'pred': pred,
                    'conf': conf
                })

            results['total'] += 1

            if truth in ['A', 'T']:
                results['at_total'] += 1
            else:
                results['gc_total'] += 1
                if pred == truth:
                    results['gc_correct'] += 1

        except Exception as e:
            logger.warning(f"Query failed for {v['chrom']}:{v['pos']}: {e}")

    # Compute accuracies
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    results['at_accuracy'] = results['at_correct'] / results['at_total'] if results['at_total'] > 0 else 0
    results['gc_accuracy'] = results['gc_correct'] / results['gc_total'] if results['gc_total'] > 0 else 0

    logger.info(f"  ✓ {hdc_class.__name__}: {results['accuracy']:.2%} overall (AT: {results['at_accuracy']:.2%}, GC: {results['gc_accuracy']:.2%})")

    # Release memory
    del hdc
    import gc
    gc.collect()

    return results


def main():
    # Paths
    gdiff_path = Path('data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz')
    encoded_genome = Path('data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome.h5')
    output_dir = Path('HDV_VALIDATION_PACKAGE/error_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("SIMPLE QUANTIZATION COMPARISON")
    logger.info("=" * 80)
    logger.info("")

    # Load ground truth
    variants = load_ground_truth(gdiff_path)

    # Select test positions
    test_positions = select_test_positions(variants, n=5000)
    logger.info("")

    # Test each quantization level
    results = {}

    # Int8
    logger.info("=" * 80)
    logger.info("INT8 QUANTIZATION")
    logger.info("=" * 80)
    from int8_lightning_hdc import Int8LightningHDC
    results['int8'] = query_quantization_level(Int8LightningHDC, str(encoded_genome), test_positions)
    logger.info("")

    # Int4
    logger.info("=" * 80)
    logger.info("INT4 QUANTIZATION")
    logger.info("=" * 80)
    from int4_lightning_hdc import Int4LightningHDC
    results['int4'] = query_quantization_level(Int4LightningHDC, encoded_genome, test_positions)
    logger.info("")

    # Binary
    logger.info("=" * 80)
    logger.info("BINARY QUANTIZATION")
    logger.info("=" * 80)
    from binary_lightning_hdc import BinaryLightningHDC
    results['binary'] = query_quantization_level(BinaryLightningHDC, str(encoded_genome), test_positions)
    logger.info("")

    # Save results
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    # Summary
    summary = {
        level: {
            'accuracy': res['accuracy'],
            'at_accuracy': res['at_accuracy'],
            'gc_accuracy': res['gc_accuracy'],
            'errors': len(res['errors']),
            'total_queries': res['total']
        }
        for level, res in results.items()
    }

    summary_path = output_dir / 'quantization_comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  ✓ Summary saved to {summary_path}")

    # Error details
    errors_path = output_dir / 'quantization_comparison_errors.json'
    errors_data = {
        level: res['errors']
        for level, res in results.items()
    }
    with open(errors_path, 'w') as f:
        json.dump(errors_data, f, indent=2)
    logger.info(f"  ✓ Error details saved to {errors_path}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    for level in ['int8', 'int4', 'binary']:
        res = results[level]
        logger.info(f"{level:10s}: {res['accuracy']:6.2%} (AT: {res['at_accuracy']:6.2%}, GC: {res['gc_accuracy']:6.2%}, Errors: {len(res['errors']):,})")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
