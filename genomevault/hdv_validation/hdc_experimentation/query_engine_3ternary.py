#!/usr/bin/env python3
"""
Query Engine for 3-Ternary Bank HDC with Lens-Aware Decoder

Architecture:
- Bank 1: Hydrophobic (T=+1, A=-1, GC=0)
- Bank 2: Major Groove (G=+1, C=-1, AT=0)
- Bank 3: Hinge (YR=+1, RY=-1, neutral=0)

Features:
- Direct ternary quantization (no 6-binary reconstruction)
- ZCR-based texture classification
- Structural motif lens library
- LINEAR magnitude-based compositional weighting
- Genomic Monty Hall cross-validation

Version: 2.0 (3-Ternary Architecture)
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


class ThreeTernaryQueryEngine:
    """
    Query engine for 3-ternary bank HDC with lens awareness.

    This is the query interface for the 3-ternary architecture.
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
        """
        Initialize query engine.

        Args:
            encoded_h5_path: Path to encoded_genome_3banks.h5
            lens_library: Optional lens library for texture-aware decoding
            use_magnitude_weighting: Enable LINEAR magnitude weighting
            lens_alpha: Lens overlay strength (0.0-1.0)
            D: Dimension (must match encoder)
            N: Chunk size (must match encoder)
            seed: Random seed (must match encoder)
        """
        self.D = D
        self.N = N
        self.seed = seed

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

        logger.info(f"Initialized 3-Ternary Query Engine")
        logger.info(f"  D={D}, N={N}, seed={seed}")
        logger.info(f"  Lens library: {len(lens_library.lenses) if lens_library else 0} lenses")
        logger.info(f"  Magnitude weighting: {use_magnitude_weighting}")
        logger.info(f"  Lens alpha: {lens_alpha}")

    def query_position(self, chrom: str, pos: int) -> Tuple[str, float, Optional[str], Optional[str]]:
        """
        Query a genomic position.

        Returns:
            (nucleotide, confidence, texture_type, lens_name)
        """
        return self.decoder.decode_position(chrom, pos, self.position_codebook)

    def close(self):
        """Close decoder."""
        self.decoder.close()


def run_validation(
    encoded_h5_path: Path,
    gdiff_path: Path,
    lens_library_path: Optional[Path] = None,
    sample_size: int = 1000,
    use_magnitude_weighting: bool = True,
    lens_alpha: float = 0.3,
    seed: int = 42
):
    """
    Run comprehensive validation of 3-ternary HDC with lens awareness.

    Args:
        encoded_h5_path: Path to encoded_genome_3banks.h5
        gdiff_path: Path to experimental.gdiff.gz
        lens_library_path: Optional path to lens_library.h5
        sample_size: Number of positions to test
        use_magnitude_weighting: Enable LINEAR magnitude weighting
        lens_alpha: Lens overlay strength
        seed: Random seed
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("3-TERNARY BANK HDC VALIDATION")
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
    logger.info(f"Initializing query engine...")
    start_time = time.time()
    query_engine = ThreeTernaryQueryEngine(
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
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 80)
    logger.info("")

    correct = 0
    total = 0
    per_nuc_stats = {nuc: {'correct': 0, 'total': 0} for nuc in 'ATGC'}
    texture_stats = Counter()
    lens_stats = Counter()
    confidence_by_texture = defaultdict(list)

    for i, (chrom, pos) in enumerate(test_positions):
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(test_positions)} positions")

        # Get ground truth
        pos_key = f"{chrom}:{pos}"
        is_variant = pos_key in variant_index

        if is_variant:
            v = variant_index[pos_key]
            ground_truth = v["alt"]
        else:
            # Get from experimental BAM
            if exp_bam is None:
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
                    continue

                base_counts = Counter(bases)
                ground_truth = base_counts.most_common(1)[0][0]
            except Exception as e:
                continue

        if ground_truth not in ['A', 'T', 'G', 'C']:
            continue

        # Query position
        try:
            prediction, confidence, texture, lens_name = query_engine.query_position(chrom, pos)

            # Track statistics
            is_correct = (prediction == ground_truth)
            if is_correct:
                correct += 1
                per_nuc_stats[ground_truth]['correct'] += 1

            per_nuc_stats[ground_truth]['total'] += 1
            total += 1

            if texture:
                texture_stats[texture] += 1
                confidence_by_texture[texture].append(confidence)

            if lens_name:
                lens_stats[lens_name] += 1

        except Exception as e:
            logger.error(f"Error querying {chrom}:{pos}: {e}")
            continue

    # Results
    logger.info("")
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Overall Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    logger.info("")

    logger.info("Per-Nucleotide Accuracy:")
    for nuc in 'ATGC':
        nuc_total = per_nuc_stats[nuc]['total']
        nuc_correct = per_nuc_stats[nuc]['correct']
        nuc_acc = nuc_correct / nuc_total if nuc_total > 0 else 0
        logger.info(f"  {nuc}: {nuc_acc*100:.1f}% ({nuc_correct}/{nuc_total})")
    logger.info("")

    if texture_stats:
        logger.info("Texture Classification Distribution:")
        total_textures = sum(texture_stats.values())
        for texture, count in texture_stats.most_common():
            pct = (count / total_textures) * 100
            logger.info(f"  {texture:20s}: {count:4d} ({pct:5.1f}%)")
        logger.info("")

        logger.info("Confidence by Texture:")
        for texture in texture_stats.keys():
            confs = confidence_by_texture[texture]
            if confs:
                mean_conf = np.mean(confs)
                logger.info(f"  {texture:20s}: {mean_conf:.3f} mean confidence")
        logger.info("")

    if lens_stats:
        logger.info("Lens Usage Distribution:")
        total_lens_uses = sum(lens_stats.values())
        for lens_name, count in lens_stats.most_common():
            pct = (count / total_lens_uses) * 100
            logger.info(f"  {lens_name:20s}: {count:4d} ({pct:5.1f}%)")
        logger.info("")

    # Save results
    results = {
        'architecture': '3-ternary banks',
        'lens_library': lens_library_path.name if lens_library_path else None,
        'magnitude_weighting': use_magnitude_weighting,
        'lens_alpha': lens_alpha,
        'overall': {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        },
        'per_nucleotide': {
            nuc: {
                'accuracy': per_nuc_stats[nuc]['correct'] / per_nuc_stats[nuc]['total']
                    if per_nuc_stats[nuc]['total'] > 0 else 0,
                'correct': per_nuc_stats[nuc]['correct'],
                'total': per_nuc_stats[nuc]['total']
            }
            for nuc in 'ATGC'
        },
        'texture_distribution': dict(texture_stats),
        'lens_usage': dict(lens_stats)
    }

    output_path = Path("genomevault/hdv_validation/hdc_experimentation/results/3ternary_validation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Results saved to: {output_path}")
    logger.info("")

    # Cleanup
    if exp_bam:
        exp_bam.close()
    query_engine.close()

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Architecture: 3 ternary banks (Bank 1: Hydrophobic, Bank 2: Major Groove, Bank 3: Hinge)")
    logger.info(f"Lens-aware: {'Yes' if lens_library else 'No'}")
    logger.info(f"Magnitude weighting: {'LINEAR' if use_magnitude_weighting else 'Off'}")
    logger.info("")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3-Ternary Bank HDC Validation')
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

    args = parser.parse_args()

    # Convert paths
    encoded_h5_path = Path(args.encoded_h5)
    gdiff_path = Path(args.gdiff)
    lens_library_path = Path(args.lens_library) if args.lens_library else None

    # Validate paths
    if not encoded_h5_path.exists():
        logger.error(f"Encoded H5 not found: {encoded_h5_path}")
        sys.exit(1)

    if not gdiff_path.exists():
        logger.error(f"GDiff not found: {gdiff_path}")
        sys.exit(1)

    # Run validation
    results = run_validation(
        encoded_h5_path=encoded_h5_path,
        gdiff_path=gdiff_path,
        lens_library_path=lens_library_path,
        sample_size=args.sample_size,
        use_magnitude_weighting=not args.no_magnitude,
        lens_alpha=args.lens_alpha,
        seed=args.seed
    )
