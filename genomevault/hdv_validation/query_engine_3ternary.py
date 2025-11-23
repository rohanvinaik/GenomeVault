#!/usr/bin/env python3
"""
3-Ternary Bank HDC Query Engine

Integrates with existing validation infrastructure (compare_quantizations.py, validation_utils.py).

Architecture:
- Bank 1: Hydrophobic (T=+1, A=-1, GC=0)
- Bank 2: Major Groove (G=+1, C=-1, AT=0)
- Bank 3: Hinge (YR=+1, RY=-1, neutral=0)

Features:
- Direct ternary quantization {-1, 0, +1}
- ZCR-based texture classification
- Structural motif lens library (optional)
- LINEAR magnitude-based compositional weighting
- Genomic Monty Hall cross-validation

Compatible with existing validation_utils functions:
- load_gdiff()
- sample_test_positions()
- get_ground_truth()
- save_results()
- compute_confusion_matrix()
"""

import json
import gzip
import logging
import time
import h5py
import numpy as np
import pysam
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import defaultdict, Counter

# Import lens-aware decoder from hdc_experimentation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "hdv_validation" / "hdc_experimentation"))
from decoders.lens_aware_decoder_CORRECTED_3TERNARY import LensLibrary, LensAwareDecoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class PreEncoded3TernaryHDV:
    """
    Query engine for 3-ternary bank HDC with optional lens awareness.

    Compatible with existing validation infrastructure.
    """

    def __init__(
        self,
        hdf5_path: Path,
        lens_library: Optional[LensLibrary] = None,
        use_magnitude_weighting: bool = True,
        lens_alpha: float = 0.3,
        D: int = 5120,
        N: int = 1024,
        seed: int = 42,
        guide_fasta_dir: Optional[Path] = None
    ):
        """
        Initialize 3-ternary query engine.

        Args:
            hdf5_path: Path to encoded_genome_3banks.h5
            lens_library: Optional lens library for texture-aware decoding
            use_magnitude_weighting: Enable LINEAR magnitude weighting
            lens_alpha: Lens overlay strength (0.0-1.0)
            D: Dimension (must match encoder)
            N: Chunk size (must match encoder)
            seed: Random seed (must match encoder)
            guide_fasta_dir: Optional guide FASTA directory (for N checking)
        """
        self.D = D
        self.N = N
        self.seed = seed
        self.hdf5_path = hdf5_path
        self.lens_library = lens_library
        self.use_magnitude_weighting = use_magnitude_weighting
        self.lens_alpha = lens_alpha

        # Generate position codebook (must match encoder)
        np.random.seed(seed)
        self.position_codebook = np.random.choice([-1, 1], size=(N, D)).astype(np.int8)

        # Initialize lens-aware decoder
        self.decoder = LensAwareDecoder(
            encoded_h5_path=str(hdf5_path),
            lens_library=lens_library,
            use_magnitude_weighting=use_magnitude_weighting,
            lens_alpha=lens_alpha
        )

        # Load chunk index
        with h5py.File(hdf5_path, 'r') as f:
            chunk_keys_bytes = f['chunk_keys'][:]
            self.chunk_keys = [k.decode('utf-8') for k in chunk_keys_bytes]
            self.total_chunks = len(self.chunk_keys)

        self.chunk_index = {}
        for idx, key in enumerate(self.chunk_keys):
            self.chunk_index[key] = idx

        # Load guide FASTAs if provided (for N checking - compatibility with 5-lens system)
        self.guide_fastas = {}
        if guide_fasta_dir:
            logger.info("Opening guide FASTAs for N-position tracking...")
            for i in range(1, 12):
                guide_path = guide_fasta_dir / f"ref{i}.fa.gz"
                if guide_path.exists():
                    try:
                        self.guide_fastas[f'ref{i}'] = pysam.FastaFile(str(guide_path))
                        logger.info(f"  Guide {i}: Opened (indexed access)")
                    except:
                        logger.warning(f"  Guide {i}: Failed to open")
            logger.info(f"  Total guides opened: {len(self.guide_fastas)}")

        logger.info(f"Initialized 3-Ternary Query Engine")
        logger.info(f"  D={D}, N={N}, seed={seed}")
        logger.info(f"  Lens library: {len(lens_library.lenses) if lens_library else 0} lenses")
        logger.info(f"  Magnitude weighting: {use_magnitude_weighting}")
        logger.info(f"  Lens alpha: {lens_alpha}")

    def query_position(
        self,
        chrom: str,
        pos: int
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """
        Query a genomic position with 3-ternary architecture.

        Returns:
            (nucleotide, confidence, texture_type, lens_name)
        """
        return self.decoder.decode_position(chrom, pos, self.position_codebook)

    def query_position_detailed(
        self,
        chrom: str,
        pos: int
    ) -> Dict:
        """
        Query with detailed bank-level information.

        Returns dict compatible with existing validation infrastructure.
        """
        nucleotide, confidence, texture, lens_name = self.query_position(chrom, pos)

        # Return in format compatible with existing validation tools
        return {
            'predicted': nucleotide,
            'confidence': confidence,
            'texture_type': texture,
            'lens_name': lens_name,
            'architecture': '3-ternary'
        }

    def close(self):
        """Close decoder and guide FASTAs."""
        self.decoder.close()
        for guide_name, guide_fasta in self.guide_fastas.items():
            guide_fasta.close()


def predict_3ternary_voting(
    query_result: Dict,
    use_magnitude_weighting: bool = True
) -> Tuple[str, float]:
    """
    Extract prediction and confidence from 3-ternary query result.

    Compatible with existing validation infrastructure.

    Args:
        query_result: Result from query_position_detailed()
        use_magnitude_weighting: Whether magnitude weighting was used

    Returns:
        (predicted_nucleotide, confidence)
    """
    return query_result['predicted'], query_result['confidence']


def run_3ternary_validation(
    encoded_h5_path: Path,
    gdiff_path: Path,
    lens_library_path: Optional[Path] = None,
    sample_size: int = 1000,
    use_magnitude_weighting: bool = True,
    lens_alpha: float = 0.3,
    seed: int = 42,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run validation of 3-ternary HDC using existing validation infrastructure.

    Compatible with compare_quantizations.py workflow.

    Args:
        encoded_h5_path: Path to encoded_genome_3banks.h5
        gdiff_path: Path to experimental.gdiff.gz
        lens_library_path: Optional path to lens_library.h5
        sample_size: Number of positions to test
        use_magnitude_weighting: Enable LINEAR magnitude weighting
        lens_alpha: Lens overlay strength
        seed: Random seed
        output_dir: Output directory for results

    Returns:
        Results dict compatible with existing reporting tools
    """
    from genomevault.hdv_validation.validation_utils import (
        load_gdiff,
        sample_test_positions,
        get_ground_truth,
        save_results,
        compute_confusion_matrix
    )

    if output_dir is None:
        output_dir = Path("genomevault/hdv_validation/results/3ternary_results")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

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

    guide_fasta_dir = Path("/Volumes/1TBStorage/guide_strands")

    query_engine = PreEncoded3TernaryHDV(
        hdf5_path=encoded_h5_path,
        lens_library=lens_library,
        use_magnitude_weighting=use_magnitude_weighting,
        lens_alpha=lens_alpha,
        D=5120,
        N=1024,
        seed=seed,
        guide_fasta_dir=guide_fasta_dir if guide_fasta_dir.exists() else None
    )
    logger.info(f"  ✓ Initialized in {time.time() - start_time:.2f}s")
    logger.info("")

    # Load ground truth from GDiff
    logger.info("Loading ground truth from GDiff...")
    gdiff, variant_index = load_gdiff(gdiff_path)
    logger.info(f"  Total variants: {len(variant_index):,}")

    # Sample test positions
    logger.info("")
    logger.info("Sampling test positions...")

    test_positions, high_n_set = sample_test_positions(
        query_engine.chunk_keys,
        [],  # No validated N positions for now
        sample_size,
        n_sample_ratio=0.0,
        seed=seed
    )

    logger.info(f"  ✓ Sampled {len(test_positions)} positions")
    logger.info("")

    # Open experimental BAM for non-variant ground truth
    exp_bam_path = Path("data/experimental_strands/ERR3239334/alignment/k11_bams/experimental_vs_ref1.sorted.bam")
    exp_bam = pysam.AlignmentFile(str(exp_bam_path), 'rb') if exp_bam_path.exists() else None
    region_map = gdiff.get("region_guide_map", {})

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
    predictions = []

    for i, (chrom, pos) in enumerate(test_positions):
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(test_positions)} positions")

        # Get ground truth using existing infrastructure
        ground_truth, guide_idx, has_n = get_ground_truth(
            chrom, pos, variant_index, exp_bam, region_map
        )

        if not ground_truth or ground_truth not in ['A', 'T', 'G', 'C']:
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

            # Store prediction for analysis
            predictions.append({
                'position': f"{chrom}:{pos}",
                'ground_truth': ground_truth,
                'predicted': prediction,
                'confidence': confidence,
                'correct': is_correct,
                'texture': texture,
                'lens_name': lens_name
            })

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

    # Compute confusion matrix using existing utility
    pred_list = [p['predicted'] for p in predictions]
    truth_list = [p['ground_truth'] for p in predictions]
    confusion = compute_confusion_matrix(pred_list, truth_list)

    # Save results using existing infrastructure
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
        'lens_usage': dict(lens_stats),
        'confusion_matrix': confusion,
        'predictions': predictions
    }

    output_path = output_dir / "3ternary_validation_results.json"
    save_results(results, output_path)

    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Architecture: 3 ternary banks (Bank 1: Hydrophobic, Bank 2: Major Groove, Bank 3: Hinge)")
    logger.info(f"Lens-aware: {'Yes' if lens_library else 'No'}")
    logger.info(f"Magnitude weighting: {'LINEAR' if use_magnitude_weighting else 'Off'}")
    logger.info("")

    # Cleanup
    if exp_bam:
        exp_bam.close()
    query_engine.close()

    return results


if __name__ == '__main__':
    import argparse

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
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')

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
    results = run_3ternary_validation(
        encoded_h5_path=encoded_h5_path,
        gdiff_path=gdiff_path,
        lens_library_path=lens_library_path,
        sample_size=args.sample_size,
        use_magnitude_weighting=not args.no_magnitude,
        lens_alpha=args.lens_alpha,
        seed=args.seed,
        output_dir=args.output_dir
    )
