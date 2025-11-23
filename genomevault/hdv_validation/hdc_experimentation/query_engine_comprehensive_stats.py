#!/usr/bin/env python3
"""
Comprehensive Statistics Query Engine for 3-Ternary Bank HDC

Captures ALL relevant statistics:
- Query timing/latency (mean, median, p95, p99, min, max)
- Confidence distribution (overall, per-nucleotide, per-texture)
- Error confusion matrix (A→T, G→C, etc.)
- Texture-based accuracy
- Lens effectiveness (accuracy with/without each lens)
- Bank contribution analysis
- Coverage statistics
- SNR and sparsity metrics
- Genomic Monty Hall cross-validation statistics

Version: 3.0 (Comprehensive Statistics)
Date: November 2025
"""

import json
import gzip
import logging
import time
import h5py
import numpy as np
import pysam
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our corrected lens-aware decoder
from decoders.lens_aware_decoder_CORRECTED_3TERNARY import (
    LensLibrary, LensAwareDecoder, TextureClassifier
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryStatistics:
    """Comprehensive query statistics."""
    # Overall metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    correct_predictions: int = 0
    overall_accuracy: float = 0.0

    # Timing statistics
    query_times: List[float] = None
    mean_query_time: float = 0.0
    median_query_time: float = 0.0
    p95_query_time: float = 0.0
    p99_query_time: float = 0.0
    min_query_time: float = 0.0
    max_query_time: float = 0.0

    # Confidence statistics
    confidences: List[float] = None
    mean_confidence: float = 0.0
    median_confidence: float = 0.0
    std_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0

    # Per-nucleotide statistics
    per_nucleotide: Dict = None

    # Error confusion matrix
    confusion_matrix: Dict = None

    # Texture-based statistics
    texture_distribution: Dict = None
    texture_accuracy: Dict = None
    texture_confidence: Dict = None

    # Lens statistics
    lens_usage: Dict = None
    lens_accuracy: Dict = None
    lens_effectiveness: Dict = None

    # Bank statistics
    bank_sparsity: Dict = None
    bank_snr: Dict = None

    # Coverage statistics
    variant_positions: int = 0
    non_variant_positions: int = 0
    no_ground_truth: int = 0

    def __post_init__(self):
        if self.query_times is None:
            self.query_times = []
        if self.confidences is None:
            self.confidences = []
        if self.per_nucleotide is None:
            self.per_nucleotide = {
                nuc: {
                    'correct': 0, 'total': 0, 'accuracy': 0.0,
                    'confidences': [], 'mean_confidence': 0.0
                }
                for nuc in 'ATGC'
            }
        if self.confusion_matrix is None:
            self.confusion_matrix = {
                true: {pred: 0 for pred in 'ATGCN'}
                for true in 'ATGC'
            }
        if self.texture_distribution is None:
            self.texture_distribution = {}
        if self.texture_accuracy is None:
            self.texture_accuracy = {}
        if self.texture_confidence is None:
            self.texture_confidence = {}
        if self.lens_usage is None:
            self.lens_usage = {}
        if self.lens_accuracy is None:
            self.lens_accuracy = {}
        if self.lens_effectiveness is None:
            self.lens_effectiveness = {}
        if self.bank_sparsity is None:
            self.bank_sparsity = {'bank1': [], 'bank2': [], 'bank3': []}
        if self.bank_snr is None:
            self.bank_snr = {'bank1': [], 'bank2': [], 'bank3': []}

    def finalize(self):
        """Compute final statistics from accumulated data."""
        # Overall accuracy
        if self.successful_queries > 0:
            self.overall_accuracy = self.correct_predictions / self.successful_queries

        # Timing statistics
        if self.query_times:
            self.mean_query_time = np.mean(self.query_times)
            self.median_query_time = np.median(self.query_times)
            self.p95_query_time = np.percentile(self.query_times, 95)
            self.p99_query_time = np.percentile(self.query_times, 99)
            self.min_query_time = np.min(self.query_times)
            self.max_query_time = np.max(self.query_times)

        # Confidence statistics
        if self.confidences:
            self.mean_confidence = np.mean(self.confidences)
            self.median_confidence = np.median(self.confidences)
            self.std_confidence = np.std(self.confidences)
            self.min_confidence = np.min(self.confidences)
            self.max_confidence = np.max(self.confidences)

        # Per-nucleotide statistics
        for nuc in 'ATGC':
            nuc_stats = self.per_nucleotide[nuc]
            if nuc_stats['total'] > 0:
                nuc_stats['accuracy'] = nuc_stats['correct'] / nuc_stats['total']
            if nuc_stats['confidences']:
                nuc_stats['mean_confidence'] = np.mean(nuc_stats['confidences'])

        # Texture-based accuracy
        for texture, stats in self.texture_accuracy.items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']

        # Lens effectiveness
        for lens, stats in self.lens_accuracy.items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']

        # Bank statistics
        for bank in ['bank1', 'bank2', 'bank3']:
            if self.bank_sparsity[bank]:
                self.bank_sparsity[bank] = {
                    'mean': np.mean(self.bank_sparsity[bank]),
                    'median': np.median(self.bank_sparsity[bank]),
                    'std': np.std(self.bank_sparsity[bank])
                }
            if self.bank_snr[bank]:
                self.bank_snr[bank] = {
                    'mean': np.mean(self.bank_snr[bank]),
                    'median': np.median(self.bank_snr[bank]),
                    'std': np.std(self.bank_snr[bank])
                }


class ComprehensiveStatsQueryEngine:
    """
    Query engine with comprehensive statistics tracking.
    """

    def __init__(
        self,
        encoded_h5_path: Path,
        lens_library: Optional[LensLibrary] = None,
        use_magnitude_weighting: bool = True,
        lens_alpha: float = 0.3,
        D: int = 5120,
        N: int = 1024,
        seed: int = 42
    ):
        """Initialize query engine with statistics tracking."""
        self.D = D
        self.N = N
        self.seed = seed
        self.stats = QueryStatistics()

        # Generate position codebook (must match encoder)
        np.random.seed(seed)
        self.position_codebook = np.random.choice([-1, 1], size=(N, D)).astype(np.int8)

        # Initialize lens-aware decoder
        self.decoder = LensAwareDecoder(
            encoded_h5_path=str(encoded_h5_path),
            lens_library=lens_library,
            use_magnitude_weighting=use_magnitude_weighting,
            lens_alpha=lens_alpha
        )

        # Open H5 for direct bank access
        self.h5_file = h5py.File(encoded_h5_path, 'r')

        logger.info(f"Initialized Comprehensive Stats Query Engine")
        logger.info(f"  D={D}, N={N}, seed={seed}")
        logger.info(f"  Lens library: {len(lens_library.lenses) if lens_library else 0} lenses")
        logger.info(f"  Magnitude weighting: {use_magnitude_weighting}")
        logger.info(f"  Lens alpha: {lens_alpha}")

    def query_position_with_stats(
        self,
        chrom: str,
        pos: int,
        ground_truth: Optional[str] = None
    ) -> Tuple[str, float, Optional[str], Optional[str], Dict]:
        """
        Query a position and collect comprehensive statistics.

        Returns:
            (nucleotide, confidence, texture, lens_name, bank_stats)
        """
        start_time = time.time()

        try:
            # Query using decoder
            prediction, confidence, texture, lens_name = self.decoder.decode_position(
                chrom, pos, self.position_codebook
            )

            query_time = time.time() - start_time

            # Track timing
            self.stats.query_times.append(query_time)
            self.stats.successful_queries += 1

            # Track confidence
            self.stats.confidences.append(confidence)

            # Track ground truth comparison
            if ground_truth:
                is_correct = (prediction == ground_truth)
                if is_correct:
                    self.stats.correct_predictions += 1
                    if ground_truth in 'ATGC':
                        self.stats.per_nucleotide[ground_truth]['correct'] += 1

                if ground_truth in 'ATGC':
                    self.stats.per_nucleotide[ground_truth]['total'] += 1
                    self.stats.per_nucleotide[ground_truth]['confidences'].append(confidence)

                    # Confusion matrix
                    self.stats.confusion_matrix[ground_truth][prediction] += 1

            # Track texture statistics
            if texture:
                self.stats.texture_distribution[texture] = \
                    self.stats.texture_distribution.get(texture, 0) + 1

                if texture not in self.stats.texture_accuracy:
                    self.stats.texture_accuracy[texture] = {'correct': 0, 'total': 0, 'accuracy': 0.0}
                if texture not in self.stats.texture_confidence:
                    self.stats.texture_confidence[texture] = []

                self.stats.texture_accuracy[texture]['total'] += 1
                if ground_truth and prediction == ground_truth:
                    self.stats.texture_accuracy[texture]['correct'] += 1

                self.stats.texture_confidence[texture].append(confidence)

            # Track lens statistics
            if lens_name:
                self.stats.lens_usage[lens_name] = \
                    self.stats.lens_usage.get(lens_name, 0) + 1

                if lens_name not in self.stats.lens_accuracy:
                    self.stats.lens_accuracy[lens_name] = {'correct': 0, 'total': 0, 'accuracy': 0.0}

                self.stats.lens_accuracy[lens_name]['total'] += 1
                if ground_truth and prediction == ground_truth:
                    self.stats.lens_accuracy[lens_name]['correct'] += 1

            # Get bank statistics
            bank_stats = self._analyze_banks(chrom, pos)

            return prediction, confidence, texture, lens_name, bank_stats

        except Exception as e:
            self.stats.failed_queries += 1
            logger.error(f"Query failed for {chrom}:{pos}: {e}")
            raise

        finally:
            self.stats.total_queries += 1

    def _analyze_banks(self, chrom: str, pos: int) -> Dict:
        """Analyze bank-specific statistics for a position."""
        try:
            # Find chunk containing this position
            chunk_idx = self.decoder._find_chunk_index(chrom, pos)
            if chunk_idx is None:
                return {}

            # Load bank vectors
            chunk_vectors = self.decoder._load_chunk_vectors(chunk_idx)

            # Analyze each bank
            bank_stats = {}
            for bank_name in ['bank1', 'bank2', 'bank3']:
                vec = chunk_vectors[bank_name]

                # Sparsity: percentage of zero values
                sparsity = np.sum(vec == 0) / len(vec)

                # SNR estimate: mean absolute value / std
                nonzero_vals = vec[vec != 0]
                if len(nonzero_vals) > 0:
                    snr = np.mean(np.abs(nonzero_vals)) / (np.std(nonzero_vals) + 1e-10)
                else:
                    snr = 0.0

                bank_stats[bank_name] = {
                    'sparsity': sparsity,
                    'snr': snr,
                    'active_dims': int(np.sum(vec != 0))
                }

                # Accumulate for global statistics
                self.stats.bank_sparsity[bank_name].append(sparsity)
                self.stats.bank_snr[bank_name].append(snr)

            return bank_stats

        except Exception as e:
            return {}

    def close(self):
        """Close resources."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
        self.decoder.close()


def run_comprehensive_validation(
    encoded_h5_path: Path,
    gdiff_path: Path,
    lens_library_path: Optional[Path] = None,
    sample_size: int = 1000,
    use_magnitude_weighting: bool = True,
    lens_alpha: float = 0.3,
    seed: int = 42,
    output_dir: Optional[Path] = None
):
    """
    Run validation with comprehensive statistics collection.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE STATISTICS VALIDATION")
    logger.info("=" * 80)
    logger.info("")

    # Load lens library if provided
    lens_library = None
    if lens_library_path and lens_library_path.exists():
        logger.info(f"Loading lens library from {lens_library_path}...")
        lens_library = LensLibrary.load(lens_library_path)
        logger.info(f"  ✓ Loaded {len(lens_library.lenses)} lenses")
    else:
        logger.info("No lens library provided - using baseline decoding")

    # Initialize query engine
    logger.info(f"Initializing comprehensive stats query engine...")
    start_time = time.time()
    query_engine = ComprehensiveStatsQueryEngine(
        encoded_h5_path=encoded_h5_path,
        lens_library=lens_library,
        use_magnitude_weighting=use_magnitude_weighting,
        lens_alpha=lens_alpha,
        D=5120,
        N=1024,
        seed=seed
    )
    logger.info(f"  ✓ Initialized in {time.time() - start_time:.2f}s")
    logger.info("")

    # Load ground truth from GDiff
    logger.info("Loading ground truth from GDiff...")
    with gzip.open(gdiff_path, 'rt') as f:
        gdiff = json.load(f)

    variants = gdiff["differential_variants"]
    logger.info(f"  Total variants: {len(variants):,}")

    # Build variant index
    variant_index = {}
    for v in variants:
        key = f"{v['chrom']}:{v['pos']}"
        variant_index[key] = v

    # Open experimental BAM for non-variant ground truth
    exp_bam_path = Path("data/experimental_strands/ERR3239334/alignment/k11_bams/experimental_vs_ref1.sorted.bam")
    exp_bam = pysam.AlignmentFile(str(exp_bam_path), 'rb') if exp_bam_path.exists() else None
    logger.info("")

    # Sample random positions
    logger.info("Sampling random genomic positions...")
    np.random.seed(seed)

    # Open H5 to get chunk keys
    with h5py.File(encoded_h5_path, 'r') as f:
        chunk_keys_bytes = f['chunk_keys'][:]
        chunk_keys = [k.decode('utf-8') for k in chunk_keys_bytes]

    test_positions = []
    random_chunk_indices = np.random.randint(0, len(chunk_keys), size=sample_size)
    for chunk_idx in random_chunk_indices:
        random_chunk_key = chunk_keys[chunk_idx]
        chrom, chunk_start_str = random_chunk_key.split(':')
        chunk_start = int(chunk_start_str)
        pos = chunk_start + np.random.randint(0, query_engine.N)
        test_positions.append((chrom, pos))

    logger.info(f"  ✓ Sampled {len(test_positions)} positions")
    logger.info("")

    # Validation
    logger.info("=" * 80)
    logger.info("RUNNING QUERIES WITH STATISTICS COLLECTION")
    logger.info("=" * 80)
    logger.info("")

    for i, (chrom, pos) in enumerate(test_positions):
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(test_positions)} positions")

        # Get ground truth
        pos_key = f"{chrom}:{pos}"
        is_variant = pos_key in variant_index

        if is_variant:
            v = variant_index[pos_key]
            ground_truth = v["alt"]
            query_engine.stats.variant_positions += 1
        else:
            # Get from experimental BAM
            if exp_bam is None:
                query_engine.stats.no_ground_truth += 1
                continue

            try:
                pileup = exp_bam.pileup(chrom, pos, pos + 1, truncate=True, min_base_quality=20)
                bases = []
                for pileupcolumn in pileup:
                    if pileupcolumn.pos == pos:
                        for pileupread in pileupcolumn.pileups:
                            if not pileupread.is_del and not pileupread.is_refskip:
                                base = pileupread.alignment.query_sequence[pileupread.query_position]
                                bases.append(base.upper())

                if not bases:
                    query_engine.stats.no_ground_truth += 1
                    continue

                base_counts = Counter(bases)
                ground_truth = base_counts.most_common(1)[0][0]
                query_engine.stats.non_variant_positions += 1
            except Exception as e:
                query_engine.stats.no_ground_truth += 1
                continue

        if ground_truth not in ['A', 'T', 'G', 'C']:
            query_engine.stats.no_ground_truth += 1
            continue

        # Query position with statistics
        try:
            prediction, confidence, texture, lens_name, bank_stats = \
                query_engine.query_position_with_stats(chrom, pos, ground_truth)
        except Exception as e:
            continue

    # Finalize statistics
    logger.info("")
    logger.info("Finalizing statistics...")
    query_engine.stats.finalize()
    logger.info("")

    # Print comprehensive results
    print_comprehensive_results(query_engine.stats)

    # Save comprehensive results
    save_comprehensive_results(query_engine.stats, output_dir, lens_library_path,
                                use_magnitude_weighting, lens_alpha)

    # Cleanup
    if exp_bam:
        exp_bam.close()
    query_engine.close()

    return query_engine.stats


def print_comprehensive_results(stats: QueryStatistics):
    """Print comprehensive statistics to logger."""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE VALIDATION RESULTS")
    logger.info("=" * 80)
    logger.info("")

    # Overall metrics
    logger.info("Overall Metrics:")
    logger.info(f"  Total queries: {stats.total_queries:,}")
    logger.info(f"  Successful: {stats.successful_queries:,}")
    logger.info(f"  Failed: {stats.failed_queries:,}")
    logger.info(f"  Accuracy: {stats.overall_accuracy*100:.2f}% ({stats.correct_predictions}/{stats.successful_queries})")
    logger.info("")

    # Timing statistics
    logger.info("Query Timing (ms):")
    logger.info(f"  Mean: {stats.mean_query_time*1000:.3f} ms")
    logger.info(f"  Median: {stats.median_query_time*1000:.3f} ms")
    logger.info(f"  P95: {stats.p95_query_time*1000:.3f} ms")
    logger.info(f"  P99: {stats.p99_query_time*1000:.3f} ms")
    logger.info(f"  Min: {stats.min_query_time*1000:.3f} ms")
    logger.info(f"  Max: {stats.max_query_time*1000:.3f} ms")
    logger.info("")

    # Confidence statistics
    logger.info("Confidence Distribution:")
    logger.info(f"  Mean: {stats.mean_confidence:.3f}")
    logger.info(f"  Median: {stats.median_confidence:.3f}")
    logger.info(f"  Std Dev: {stats.std_confidence:.3f}")
    logger.info(f"  Min: {stats.min_confidence:.3f}")
    logger.info(f"  Max: {stats.max_confidence:.3f}")
    logger.info("")

    # Per-nucleotide statistics
    logger.info("Per-Nucleotide Statistics:")
    for nuc in 'ATGC':
        nuc_stats = stats.per_nucleotide[nuc]
        if nuc_stats['total'] > 0:
            logger.info(f"  {nuc}:")
            logger.info(f"    Accuracy: {nuc_stats['accuracy']*100:.1f}% ({nuc_stats['correct']}/{nuc_stats['total']})")
            logger.info(f"    Mean Confidence: {nuc_stats['mean_confidence']:.3f}")
    logger.info("")

    # Confusion matrix
    logger.info("Error Confusion Matrix (True → Predicted):")
    logger.info("      " + "  ".join(f"{pred:>6s}" for pred in 'ATGCN'))
    for true_nuc in 'ATGC':
        row = f"  {true_nuc}: "
        for pred_nuc in 'ATGCN':
            count = stats.confusion_matrix[true_nuc][pred_nuc]
            row += f"{count:>6d}  "
        logger.info(row)
    logger.info("")

    # Texture-based accuracy
    if stats.texture_accuracy:
        logger.info("Texture-Based Accuracy:")
        for texture, tex_stats in sorted(stats.texture_accuracy.items()):
            if tex_stats['total'] > 0:
                accuracy = tex_stats['accuracy'] * 100
                count = stats.texture_distribution.get(texture, 0)
                mean_conf = np.mean(stats.texture_confidence.get(texture, [0]))
                logger.info(f"  {texture:20s}: {accuracy:5.1f}% ({tex_stats['correct']}/{tex_stats['total']}) | "
                           f"Count: {count:4d} | Conf: {mean_conf:.3f}")
        logger.info("")

    # Lens effectiveness
    if stats.lens_accuracy:
        logger.info("Lens Effectiveness:")
        for lens_name, lens_stats in sorted(stats.lens_accuracy.items()):
            if lens_stats['total'] > 0:
                accuracy = lens_stats['accuracy'] * 100
                usage = stats.lens_usage.get(lens_name, 0)
                logger.info(f"  {lens_name:20s}: {accuracy:5.1f}% ({lens_stats['correct']}/{lens_stats['total']}) | "
                           f"Usage: {usage:4d}")
        logger.info("")

    # Bank statistics
    logger.info("Bank Sparsity (% zeros):")
    for bank in ['bank1', 'bank2', 'bank3']:
        sparsity = stats.bank_sparsity.get(bank, {})
        if isinstance(sparsity, dict) and 'mean' in sparsity:
            logger.info(f"  {bank}: Mean={sparsity['mean']*100:.1f}%, "
                       f"Median={sparsity['median']*100:.1f}%, "
                       f"Std={sparsity['std']*100:.1f}%")
    logger.info("")

    logger.info("Bank SNR:")
    for bank in ['bank1', 'bank2', 'bank3']:
        snr = stats.bank_snr.get(bank, {})
        if isinstance(snr, dict) and 'mean' in snr:
            logger.info(f"  {bank}: Mean={snr['mean']:.2f}, "
                       f"Median={snr['median']:.2f}, "
                       f"Std={snr['std']:.2f}")
    logger.info("")

    # Coverage statistics
    logger.info("Coverage Statistics:")
    logger.info(f"  Variant positions: {stats.variant_positions:,}")
    logger.info(f"  Non-variant positions: {stats.non_variant_positions:,}")
    logger.info(f"  No ground truth: {stats.no_ground_truth:,}")
    logger.info("")


def save_comprehensive_results(
    stats: QueryStatistics,
    output_dir: Optional[Path],
    lens_library_path: Optional[Path],
    use_magnitude_weighting: bool,
    lens_alpha: float
):
    """Save comprehensive results to JSON."""
    if output_dir is None:
        output_dir = Path("genomevault/hdv_validation/hdc_experimentation/results")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert stats to dict (handle non-serializable fields)
    results = {
        'architecture': '3-ternary banks',
        'lens_library': lens_library_path.name if lens_library_path else None,
        'magnitude_weighting': use_magnitude_weighting,
        'lens_alpha': lens_alpha,
        'overall': {
            'total_queries': stats.total_queries,
            'successful_queries': stats.successful_queries,
            'failed_queries': stats.failed_queries,
            'correct_predictions': stats.correct_predictions,
            'accuracy': stats.overall_accuracy
        },
        'timing': {
            'mean_ms': stats.mean_query_time * 1000,
            'median_ms': stats.median_query_time * 1000,
            'p95_ms': stats.p95_query_time * 1000,
            'p99_ms': stats.p99_query_time * 1000,
            'min_ms': stats.min_query_time * 1000,
            'max_ms': stats.max_query_time * 1000
        },
        'confidence': {
            'mean': stats.mean_confidence,
            'median': stats.median_confidence,
            'std': stats.std_confidence,
            'min': stats.min_confidence,
            'max': stats.max_confidence
        },
        'per_nucleotide': {
            nuc: {
                'accuracy': stats.per_nucleotide[nuc]['accuracy'],
                'correct': stats.per_nucleotide[nuc]['correct'],
                'total': stats.per_nucleotide[nuc]['total'],
                'mean_confidence': stats.per_nucleotide[nuc]['mean_confidence']
            }
            for nuc in 'ATGC'
        },
        'confusion_matrix': stats.confusion_matrix,
        'texture_distribution': stats.texture_distribution,
        'texture_accuracy': stats.texture_accuracy,
        'texture_confidence': {
            k: float(np.mean(v)) if v else 0.0
            for k, v in stats.texture_confidence.items()
        },
        'lens_usage': stats.lens_usage,
        'lens_accuracy': stats.lens_accuracy,
        'bank_sparsity': stats.bank_sparsity,
        'bank_snr': stats.bank_snr,
        'coverage': {
            'variant_positions': stats.variant_positions,
            'non_variant_positions': stats.non_variant_positions,
            'no_ground_truth': stats.no_ground_truth
        }
    }

    output_path = output_dir / "comprehensive_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Comprehensive results saved to: {output_path}")
    logger.info("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive Statistics 3-Ternary HDC Validation')
    parser.add_argument('--encoded-h5', type=str,
                        default='genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5',
                        help='Path to encoded_genome_3banks.h5')
    parser.add_argument('--gdiff', type=str,
                        default='data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz',
                        help='Path to experimental.gdiff.gz')
    parser.add_argument('--lens-library', type=str,
                        default='genomevault/hdv_validation/hdc_experimentation/output/lens_library.h5',
                        help='Path to lens_library.h5 (optional)')
    parser.add_argument('--sample-size', type=int, default=1000,
                        help='Number of positions to test')
    parser.add_argument('--no-magnitude', action='store_true',
                        help='Disable magnitude weighting')
    parser.add_argument('--lens-alpha', type=float, default=0.3,
                        help='Lens overlay strength (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    # Convert paths
    encoded_h5_path = Path(args.encoded_h5)
    gdiff_path = Path(args.gdiff)
    lens_library_path = Path(args.lens_library) if args.lens_library else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Validate paths
    if not encoded_h5_path.exists():
        logger.error(f"Encoded H5 not found: {encoded_h5_path}")
        sys.exit(1)

    if not gdiff_path.exists():
        logger.error(f"GDiff not found: {gdiff_path}")
        sys.exit(1)

    # Run comprehensive validation
    stats = run_comprehensive_validation(
        encoded_h5_path=encoded_h5_path,
        gdiff_path=gdiff_path,
        lens_library_path=lens_library_path,
        sample_size=args.sample_size,
        use_magnitude_weighting=not args.no_magnitude,
        lens_alpha=args.lens_alpha,
        seed=args.seed,
        output_dir=output_dir
    )
