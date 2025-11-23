#!/usr/bin/env python3
"""
Memory-Efficient Error Profiling - Loads ONE quantization level at a time

MEMORY STRATEGY:
- Float32: Disk streaming only (~20 MB RAM)
- Int8:    Load (30 GB) → Profile → Release → gc.collect()
- Int4:    Load (14 GB) → Profile → Release → gc.collect()
- Binary:  Load (3.5 GB) → Profile → Release → gc.collect()

Only ONE quantized genome in RAM at any time!
"""

import sys
import gc
import argparse
from pathlib import Path

# Import existing standalone implementations
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the implementations that already exist in root directory
from int8_lightning_hdc import Int8LightningHDC
from int4_lightning_hdc import Int4LightningHDC
from binary_lightning_hdc import BinaryLightningHDC

from genomevault.hypervector_transform.error_profiler import ComprehensiveErrorProfiler

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-size', type=int, default=5000)
    parser.add_argument('--output-dir', type=Path, default=Path('HDV_VALIDATION_PACKAGE/error_analysis'))
    parser.add_argument('--gdiff', type=Path, default=Path('data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz'))
    parser.add_argument('--encoded-genome', type=Path, default=Path('data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome.h5'))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MEMORY-EFFICIENT ERROR PROFILING")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Memory strategy: SEQUENTIAL PROCESSING")
    logger.info("  Only ONE quantized genome loaded into RAM at a time")
    logger.info("")

    # Initialize profiler
    profiler = ComprehensiveErrorProfiler(test_size=args.test_size)

    # Load ground truth
    logger.info("Loading ground truth...")
    variants = profiler.load_ground_truth(args.gdiff)
    logger.info(f"  ✓ {len(profiler.ground_truth_cache):,} variants")
    logger.info("")

    # Select test positions
    logger.info("Selecting test positions...")
    test_positions = profiler.select_test_positions(variants)
    logger.info(f"  ✓ {len(test_positions):,} positions selected")
    logger.info("")

    # =================================================================
    # INT8: Load → Profile → Release
    # =================================================================
    logger.info("="*80)
    logger.info("INT8 QUANTIZATION")
    logger.info("="*80)
    logger.info("")

    int8_hdc = Int8LightningHDC(str(args.encoded_genome))
    logger.info("Loading int8 into RAM...")
    int8_hdc.load()

    logger.info("Profiling int8...")
    profile_int8 = profiler.profile_quantization_level(
        level_name='int8',
        query_func=int8_hdc.query_nucleotide,
        test_positions=test_positions
    )
    profiler.profiles['int8'] = profile_int8

    # CRITICAL: Release memory before next level
    logger.info("Releasing int8 memory...")
    del int8_hdc
    gc.collect()
    logger.info("  ✓ Int8 memory released")
    logger.info("")

    # =================================================================
    # INT4: Load → Profile → Release
    # =================================================================
    logger.info("="*80)
    logger.info("INT4 QUANTIZATION")
    logger.info("="*80)
    logger.info("")

    int4_hdc = Int4LightningHDC(args.encoded_genome)
    logger.info("Loading int4 into RAM...")
    int4_hdc.load()

    logger.info("Profiling int4...")
    profile_int4 = profiler.profile_quantization_level(
        level_name='int4',
        query_func=int4_hdc.query_nucleotide,
        test_positions=test_positions
    )
    profiler.profiles['int4'] = profile_int4

    # CRITICAL: Release memory before next level
    logger.info("Releasing int4 memory...")
    del int4_hdc
    gc.collect()
    logger.info("  ✓ Int4 memory released")
    logger.info("")

    # =================================================================
    # BINARY: Load → Profile → Release
    # =================================================================
    logger.info("="*80)
    logger.info("BINARY QUANTIZATION")
    logger.info("="*80)
    logger.info("")

    binary_hdc = BinaryLightningHDC(str(args.encoded_genome))
    logger.info("Loading binary into RAM...")
    binary_hdc.load()

    logger.info("Profiling binary...")
    profile_binary = profiler.profile_quantization_level(
        level_name='binary',
        query_func=binary_hdc.query_nucleotide,
        test_positions=test_positions
    )
    profiler.profiles['binary'] = profile_binary

    # CRITICAL: Release memory
    logger.info("Releasing binary memory...")
    del binary_hdc
    gc.collect()
    logger.info("  ✓ Binary memory released")
    logger.info("")

    # =================================================================
    # GENERATE REPORTS
    # =================================================================
    logger.info("="*80)
    logger.info("GENERATING REPORTS")
    logger.info("="*80)
    logger.info("")

    profiler.generate_comprehensive_report(args.output_dir)

    logger.info("")
    logger.info(f"✓ Reports saved to: {args.output_dir}")
    logger.info("")

    # Print summary
    logger.info("SUMMARY:")
    for level in sorted(profiler.profiles.keys()):
        profile = profiler.profiles[level]
        logger.info(f"  {level:10s}: {profile.accuracy:6.2%} (AT: {profile.at_accuracy:6.2%}, GC: {profile.gc_accuracy:6.2%})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
