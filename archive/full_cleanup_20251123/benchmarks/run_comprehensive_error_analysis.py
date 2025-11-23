#!/usr/bin/env python3
"""
Comprehensive Error Analysis Across All Quantization Levels

This script runs exhaustive error profiling for Float32, Int8, Int4, and Binary
quantization levels, generating detailed reports on error patterns, clustering,
genomic context, and statistical comparisons.

Usage:
    python benchmarks/run_comprehensive_error_analysis.py --test-size 5000
"""

import argparse
import logging
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.hypervector_transform.error_profiler import (
    ComprehensiveErrorProfiler,
    ErrorProfile,
    ErrorPosition
)
from genomevault.hypervector_transform.int8_lightning_hdc import Int8LightningHDC
from genomevault.hypervector_transform.int4_lightning_hdc import Int4LightningHDC
from genomevault.hypervector_transform.binary_lightning_hdc import BinaryLightningHDC
from genomevault.differential_encoding.gdiff.decoder import GDiffDecoder
import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Float32DiskQuery:
    """
    Float32 baseline query system using disk I/O.

    This mimics the original validate_whole_genome_hdv.py approach.
    """

    def __init__(self, encoded_genome_path: Path):
        self.encoded_genome_path = encoded_genome_path
        self.D = None
        self.N = None
        self.num_chunks = None
        self.pos_codebook = None
        self.chunk_index = None

        logger.info("Initializing Float32 disk-based query system...")
        self._load_metadata()
        logger.info(f"  ✓ Loaded metadata: D={self.D}, N={self.N}, chunks={self.num_chunks}")

    def _load_metadata(self):
        """Load minimal metadata and position codebook."""
        with h5py.File(self.encoded_genome_path, 'r') as f:
            self.D = f.attrs['dimension']
            self.N = f.attrs['chunk_size']
            self.num_chunks = f['AT_vectors'].shape[0]

            # Load position codebook (small, ~20 MB for 10KD)
            self.pos_codebook = f['position_codebook'][:]

        # Load chunk index
        import pandas as pd
        index_path = self.encoded_genome_path.parent / 'chunk_index.parquet'
        self.chunk_index = pd.read_parquet(index_path)

    def query(self, chrom: str, pos: int) -> tuple[str, float]:
        """
        Query a single position using disk I/O.

        Returns:
            (predicted_nucleotide, confidence)
        """
        # Find chunk
        chunk_row = self.chunk_index[
            (self.chunk_index['chrom'] == chrom) &
            (self.chunk_index['start'] <= pos) &
            (self.chunk_index['end'] > pos)
        ]

        if chunk_row.empty:
            raise ValueError(f"No chunk found for {chrom}:{pos}")

        chunk_idx = chunk_row.iloc[0]['chunk_id']
        chunk_start = chunk_row.iloc[0]['start']
        offset = pos - chunk_start

        if offset < 0 or offset >= self.N:
            raise ValueError(f"Position {pos} out of chunk bounds")

        # Load ONLY this chunk's vectors from disk (on-demand)
        with h5py.File(self.encoded_genome_path, 'r') as f:
            at_vector = f['AT_vectors'][chunk_idx]  # Shape: (D,)
            gc_vector = f['GC_vectors'][chunk_idx]  # Shape: (D,)

        # Get position encoding
        pos_encoding = self.pos_codebook[offset]  # Shape: (D,)

        # Compute similarities
        sim_at = np.dot(at_vector, pos_encoding)
        sim_gc = np.dot(gc_vector, pos_encoding)

        # Two-stage retrieval
        if abs(sim_at) > abs(sim_gc):
            # AT pair
            if sim_at > 0:
                prediction = 'A'
                confidence = sim_at
            else:
                prediction = 'T'
                confidence = -sim_at
        else:
            # GC pair
            if sim_gc > 0:
                prediction = 'G'
                confidence = sim_gc
            else:
                prediction = 'C'
                confidence = -sim_gc

        return prediction, float(confidence)


def run_quantization_level_profiling(
    profiler: ComprehensiveErrorProfiler,
    quantization_level: str,
    encoded_genome_path: Path
) -> ErrorProfile:
    """
    Run error profiling for a single quantization level.

    Args:
        profiler: The error profiler instance
        quantization_level: One of 'float32', 'int8', 'int4', 'binary'
        encoded_genome_path: Path to encoded genome HDF5 file

    Returns:
        ErrorProfile for this quantization level
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"PROFILING: {quantization_level.upper()}")
    logger.info("=" * 80)
    logger.info("")

    start_time = time.time()

    # Initialize appropriate query system
    if quantization_level == 'float32':
        query_system = Float32DiskQuery(encoded_genome_path)
    elif quantization_level == 'int8':
        query_system = Int8LightningHDC(str(encoded_genome_path))
        logger.info("Loading int8 quantized genome into RAM...")
        query_system.load()
        logger.info("  ✓ Int8 genome loaded")
    elif quantization_level == 'int4':
        query_system = Int4LightningHDC(str(encoded_genome_path))
        logger.info("Loading int4 quantized genome into RAM...")
        query_system.load()
        logger.info("  ✓ Int4 genome loaded")
    elif quantization_level == 'binary':
        query_system = BinaryLightningHDC(str(encoded_genome_path))
        logger.info("Loading binary quantized genome into RAM...")
        query_system.load()
        logger.info("  ✓ Binary genome loaded")
    else:
        raise ValueError(f"Unknown quantization level: {quantization_level}")

    # Profile this level
    profile = profiler.profile_quantization_level(
        quantization_level=quantization_level,
        query_function=query_system.query
    )

    elapsed = time.time() - start_time
    logger.info("")
    logger.info(f"✓ {quantization_level.upper()} profiling complete in {elapsed:.1f}s")
    logger.info(f"  Accuracy: {profile.accuracy:.2%}")
    logger.info(f"  AT accuracy: {profile.at_accuracy:.2%}")
    logger.info(f"  GC accuracy: {profile.gc_accuracy:.2%}")
    logger.info(f"  Errors: {len(profile.errors)}/{profile.total_queries}")

    return profile


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive error analysis across all quantization levels"
    )
    parser.add_argument(
        '--encoded-genome',
        type=Path,
        default=Path('data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome.h5'),
        help='Path to encoded genome HDF5 file'
    )
    parser.add_argument(
        '--gdiff',
        type=Path,
        default=Path('data/experimental_strands/ERR3239334/experimental.gdiff.gz'),
        help='Path to GDiff ground truth file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('HDV_VALIDATION_PACKAGE/error_analysis'),
        help='Output directory for error analysis reports'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=5000,
        help='Number of random positions to test (default: 5000)'
    )
    parser.add_argument(
        '--levels',
        nargs='+',
        default=['float32', 'int8', 'int4', 'binary'],
        choices=['float32', 'int8', 'int4', 'binary'],
        help='Quantization levels to test (default: all)'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.encoded_genome.exists():
        logger.error(f"Encoded genome not found: {args.encoded_genome}")
        return 1

    if not args.gdiff.exists():
        logger.error(f"GDiff file not found: {args.gdiff}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE ERROR ANALYSIS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Encoded genome: {args.encoded_genome}")
    logger.info(f"GDiff ground truth: {args.gdiff}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Test size: {args.test_size:,} positions")
    logger.info(f"Quantization levels: {', '.join(args.levels)}")
    logger.info("")

    # Initialize profiler
    profiler = ComprehensiveErrorProfiler(test_size=args.test_size)

    # Load ground truth
    logger.info("Loading ground truth from GDiff...")
    profiler.load_ground_truth(args.gdiff)
    logger.info(f"  ✓ Loaded {len(profiler.ground_truth_cache):,} variants")
    logger.info("")

    # Select test positions
    logger.info("Selecting random test positions...")
    test_positions = profiler.select_test_positions()
    logger.info(f"  ✓ Selected {len(test_positions):,} positions")

    # Count AT vs GC
    at_count = sum(1 for _, nt in test_positions if nt in ['A', 'T'])
    gc_count = sum(1 for _, nt in test_positions if nt in ['G', 'C'])
    logger.info(f"  AT positions: {at_count:,} ({at_count/len(test_positions)*100:.1f}%)")
    logger.info(f"  GC positions: {gc_count:,} ({gc_count/len(test_positions)*100:.1f}%)")
    logger.info("")

    # Profile each quantization level
    total_start = time.time()

    for level in args.levels:
        try:
            profile = run_quantization_level_profiling(
                profiler=profiler,
                quantization_level=level,
                encoded_genome_path=args.encoded_genome
            )
            profiler.profiles[level] = profile
        except Exception as e:
            logger.error(f"Error profiling {level}: {e}", exc_info=True)
            continue

    total_elapsed = time.time() - total_start

    logger.info("")
    logger.info("=" * 80)
    logger.info("PROFILING COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Total profiling time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    logger.info(f"Levels profiled: {len(profiler.profiles)}")
    logger.info("")

    # Generate comprehensive report
    logger.info("Generating comprehensive error analysis report...")
    logger.info("")

    profiler.generate_comprehensive_report(args.output_dir)

    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Reports saved to: {args.output_dir}")
    logger.info("")
    logger.info("Generated files:")
    logger.info(f"  - error_analysis_report.md")
    logger.info(f"  - error_data_*.json (for each level)")
    logger.info(f"  - error_overlap_analysis.json")
    logger.info(f"  - error_transition_analysis.json")
    logger.info("")

    # Print quick summary
    logger.info("QUICK SUMMARY:")
    logger.info("")
    for level in sorted(profiler.profiles.keys()):
        profile = profiler.profiles[level]
        logger.info(f"{level.upper():10s}: {profile.accuracy:6.2%} accuracy "
                   f"(AT: {profile.at_accuracy:6.2%}, GC: {profile.gc_accuracy:6.2%}) "
                   f"- {len(profile.errors)} errors")

    return 0


if __name__ == '__main__':
    sys.exit(main())
