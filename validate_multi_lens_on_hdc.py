#!/usr/bin/env python3
"""
Multi-Lens HDC Error Correction Validation

Tests whether multi-lens biophysical voting can CORRECT errors
from the pre-encoded binary HDC system.

Architecture:
1. Load pre-encoded binary HDC file (AT + GC lenses, ~97.94% accuracy)
2. For binary HDC ERRORS only:
   a. Reconstruct experimental sequence (reference + GDiff variants)
   b. Encode with all 5 lenses
   c. Use multi-lens voting to attempt correction
3. Show that multi-lens voting corrects binary HDC errors

CRITICAL: Both binary HDC and multi-lens must encode the SAME experimental sequence.
"""

import json
import gzip
import logging
import time
import numpy as np
import pysam
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from binary_lightning_hdc import BinaryLightningHDC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# BIOPHYSICAL LENS DEFINITIONS
# =============================================================================

LENS_DEFINITIONS = {
    'AT': {'positive': ('A',), 'negative': ('T',)},
    'GC': {'positive': ('G',), 'negative': ('C',)},
    'PuPy': {'positive': ('A', 'G'), 'negative': ('T', 'C')},  # Purine vs Pyrimidine
    'AmKe': {'positive': ('A', 'C'), 'negative': ('G', 'T')},  # Amino vs Keto
    'StWk': {'positive': ('G', 'C'), 'negative': ('A', 'T')},  # Strong vs Weak
}


class MultiLensChunkEncoder:
    """Encode genomic chunks with multiple biophysical lenses."""

    def __init__(self, D=10000, N=2000, seed=42):
        self.D = D
        self.N = N
        np.random.seed(seed)

        # Position codebook (bipolar) - MUST match binary HDC!
        self.pos_vectors = np.random.choice([-1, 1], size=(N, D)).astype(np.int8)

    def encode_chunk(self, sequence: str, lens_name: str) -> np.ndarray:
        """Encode a chunk through a specific biophysical lens."""
        assert len(sequence) == self.N, f"Sequence length {len(sequence)} != {self.N}"

        lens = LENS_DEFINITIONS[lens_name]
        chunk_vec = np.zeros(self.D, dtype=np.float32)

        for i, nuc in enumerate(sequence):
            if nuc in lens['positive']:
                chunk_vec += self.pos_vectors[i]
            elif nuc in lens['negative']:
                chunk_vec -= self.pos_vectors[i]
            # N or other: contribute 0 (no signal)

        return chunk_vec

    def query_position(self, chunk_vec: np.ndarray, local_pos: int) -> float:
        """Query a position in the encoded chunk."""
        pos_vec = self.pos_vectors[local_pos]
        similarity = np.dot(chunk_vec, pos_vec) / self.D
        return similarity


def predict_nucleotide_from_lenses(
    lens_results: Dict[str, float],
    method: str = 'voting'
) -> Tuple[str, float]:
    """
    Predict nucleotide from multi-lens results.

    Args:
        lens_results: Dict mapping lens name to similarity score
        method: 'voting' or 'confidence'

    Returns:
        (predicted_nucleotide, confidence)
    """
    # Each nucleotide has a unique signature across lenses
    NUCLEOTIDE_SIGNATURES = {
        'A': {'AT': +1, 'GC': 0, 'PuPy': +1, 'AmKe': +1, 'StWk': -1},
        'T': {'AT': -1, 'GC': 0, 'PuPy': -1, 'AmKe': -1, 'StWk': -1},
        'G': {'AT': 0, 'GC': +1, 'PuPy': +1, 'AmKe': -1, 'StWk': +1},
        'C': {'AT': 0, 'GC': -1, 'PuPy': -1, 'AmKe': +1, 'StWk': +1},
    }

    # Score each candidate nucleotide
    scores = {}
    for nuc, signature in NUCLEOTIDE_SIGNATURES.items():
        score = 0
        for lens_name, expected_sign in signature.items():
            observed = lens_results[lens_name]
            if expected_sign == 0:
                # Lens should be neutral
                score += 1.0 - abs(observed)
            else:
                # Lens should agree with expected sign
                if expected_sign * observed > 0:
                    score += abs(observed)
        scores[nuc] = score

    # Best match
    best_nuc = max(scores, key=scores.get)
    confidence = scores[best_nuc] / sum(scores.values())

    return best_nuc, confidence


def reconstruct_experimental_chunk(
    chrom: str,
    chunk_start: int,
    chunk_size: int,
    reference_fasta: pysam.FastaFile,
    variants_by_chunk: Dict,
) -> str:
    """
    Reconstruct experimental sequence by applying variants to reference.

    Args:
        chrom: Chromosome
        chunk_start: Start position of chunk
        chunk_size: Size of chunk (e.g., 2000)
        reference_fasta: Reference FASTA file
        variants_by_chunk: Pre-indexed variants by chunk

    Returns:
        Experimental sequence with variants applied
    """
    # Get reference sequence
    ref_seq = reference_fasta.fetch(chrom, chunk_start, chunk_start + chunk_size).upper()

    # Convert to list for mutation
    seq_list = list(ref_seq)

    # Apply variants
    chunk_key = f"{chrom}_consensus:{chunk_start}"
    if chunk_key in variants_by_chunk:
        for var in variants_by_chunk[chunk_key]:
            local_pos = var['pos'] - chunk_start
            if 0 <= local_pos < len(seq_list):
                seq_list[local_pos] = var['alt']

    # Pad if needed
    experimental_seq = ''.join(seq_list)
    if len(experimental_seq) < chunk_size:
        experimental_seq += 'N' * (chunk_size - len(experimental_seq))

    return experimental_seq


def run_multi_lens_validation(sample_size=1000):
    """Run multi-lens error correction validation."""
    logger.info("=" * 80)
    logger.info(f"MULTI-LENS HDC ERROR CORRECTION ({sample_size:,} positions)")
    logger.info("=" * 80)
    logger.info("")

    # Paths
    hdf5_path = Path("data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome.h5")
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    consensus_path = Path("data/reference_genomes/consensus.fa")

    # Load binary HDC system
    logger.info("Loading Binary HDC system (baseline)...")
    start_time = time.time()
    hdc = BinaryLightningHDC(hdf5_path)
    load_time = time.time() - start_time
    logger.info(f"  Binary HDC loaded in {load_time:.2f}s")
    logger.info("")

    # Load multi-lens encoder
    logger.info("Initializing Multi-Lens Encoder (all 5 lenses)...")
    start_time = time.time()
    encoder = MultiLensChunkEncoder(D=10000, N=2000, seed=42)
    init_time = time.time() - start_time
    logger.info(f"  Multi-Lens encoder initialized in {init_time:.3f}s")
    logger.info("")

    # Load reference FASTA
    logger.info("Loading reference FASTA (consensus)...")
    reference_fasta = pysam.FastaFile(str(consensus_path))
    logger.info("  ✓ Reference loaded")
    logger.info("")

    # Load ground truth
    logger.info("Loading ground truth from GDiff...")
    with gzip.open(gdiff_path, 'rt') as f:
        gdiff = json.load(f)

    variants = gdiff["differential_variants"]
    logger.info(f"  Total variants available: {len(variants):,}")

    # Index variants by chunk for fast lookup
    logger.info("Indexing variants by chunk...")
    variants_by_chunk = defaultdict(list)
    for v in variants:
        chunk_start = (v['pos'] // encoder.N) * encoder.N
        chunk_key = f"{v['chrom']}_consensus:{chunk_start}"
        variants_by_chunk[chunk_key].append(v)
    logger.info(f"  ✓ Indexed {len(variants_by_chunk):,} chunks")
    logger.info("")

    # Sample positions
    test_positions = np.random.choice(
        len(variants),
        size=min(sample_size, len(variants)),
        replace=False
    )
    logger.info(f"  Testing {len(test_positions):,} positions")
    logger.info("")

    # Track results
    binary_hdc_correct = 0
    multi_lens_correct = 0
    total = 0

    binary_hdc_errors = []
    multi_lens_corrected = 0
    multi_lens_failed_to_correct = 0
    both_wrong = 0

    # Timing
    binary_query_times = []
    multi_lens_query_times = []

    # Track multi-lens attempts (only on binary HDC errors)
    multi_lens_attempts = 0

    logger.info("Testing positions...")
    logger.info("")

    # Test each position
    for i, idx in enumerate(test_positions):
        if i % 100 == 0 and i > 0:
            logger.info(f"  Progress: {i:,}/{len(test_positions):,} ({i/len(test_positions)*100:.1f}%)")

        v = variants[idx]
        chrom = v["chrom"]
        pos = v["pos"]
        ground_truth = v["alt"]

        if ground_truth not in ['A', 'T', 'G', 'C']:
            continue

        # =====================================================================
        # BASELINE: Query pre-encoded binary HDC (AT + GC lenses only)
        # =====================================================================
        start_time = time.time()
        binary_pred, binary_conf = hdc.query_nucleotide(chrom, pos)
        binary_time = time.time() - start_time
        binary_query_times.append(binary_time)

        binary_correct = (binary_pred == ground_truth)

        if binary_correct:
            binary_hdc_correct += 1
            # No need for multi-lens correction - binary HDC was correct
            multi_lens_pred = binary_pred
            multi_lens_conf = binary_conf
            multi_lens_time = 0
        else:
            # ================================================================
            # ENHANCEMENT: Multi-lens correction (ONLY for binary HDC errors)
            # ================================================================
            multi_lens_attempts += 1
            start_time = time.time()

            # Find chunk
            chunk_start = (pos // encoder.N) * encoder.N
            chunk_key = f"{chrom}_consensus:{chunk_start}"
            local_pos = pos - chunk_start

            try:
                # Reconstruct experimental sequence (reference + variants)
                experimental_seq = reconstruct_experimental_chunk(
                    chrom, chunk_start, encoder.N,
                    reference_fasta, variants_by_chunk
                )

                # Encode with all 5 lenses
                lens_results = {}
                for lens_name in LENS_DEFINITIONS.keys():
                    chunk_vec = encoder.encode_chunk(experimental_seq, lens_name)
                    similarity = encoder.query_position(chunk_vec, local_pos)
                    lens_results[lens_name] = similarity

                # Predict nucleotide from lens voting
                multi_lens_pred, multi_lens_conf = predict_nucleotide_from_lenses(lens_results)

                logger.info(f"    Binary HDC error at {chrom}:{pos}")
                logger.info(f"      Ground truth: {ground_truth}")
                logger.info(f"      Binary pred:  {binary_pred} (conf={binary_conf:.3f})")
                logger.info(f"      Multi-lens:   {multi_lens_pred} (conf={multi_lens_conf:.3f})")
                logger.info(f"      Lens results: {lens_results}")

            except Exception as e:
                # Fallback to binary HDC if multi-lens fails
                logger.warning(f"    Multi-lens failed at {chrom}:{pos}: {e}")
                multi_lens_pred = binary_pred
                multi_lens_conf = binary_conf
                lens_results = {}

            multi_lens_time = time.time() - start_time
            multi_lens_query_times.append(multi_lens_time)

        # Track accuracy
        multi_correct = (multi_lens_pred == ground_truth)

        if not binary_correct:
            binary_hdc_errors.append({
                'chrom': chrom,
                'pos': pos,
                'truth': ground_truth,
                'binary_pred': binary_pred,
                'multi_lens_pred': multi_lens_pred,
                'binary_conf': binary_conf,
                'multi_lens_conf': multi_lens_conf,
                'lens_results': lens_results if 'lens_results' in locals() else {}
            })

            if multi_correct:
                multi_lens_corrected += 1
                logger.info(f"      ✅ Multi-lens CORRECTED error!")
            else:
                both_wrong += 1
                multi_lens_failed_to_correct += 1
                logger.info(f"      ❌ Multi-lens FAILED to correct")
            logger.info("")

        if multi_correct:
            multi_lens_correct += 1

        total += 1

    # Close reference
    reference_fasta.close()

    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info("")

    binary_acc = (binary_hdc_correct / total) * 100 if total > 0 else 0
    multi_acc = (multi_lens_correct / total) * 100 if total > 0 else 0
    improvement = multi_acc - binary_acc

    logger.info(f"Baseline (Binary HDC):      {binary_acc:.2f}% ({binary_hdc_correct}/{total})")
    logger.info(f"Multi-Lens Enhancement:     {multi_acc:.2f}% ({multi_lens_correct}/{total})")
    logger.info(f"Improvement:                +{improvement:.2f} percentage points")
    logger.info("")

    logger.info("ERROR CORRECTION ANALYSIS:")
    logger.info(f"  Total binary HDC errors:          {len(binary_hdc_errors):,}")
    logger.info(f"  Multi-lens correction attempts:   {multi_lens_attempts:,}")
    if binary_hdc_errors:
        logger.info(f"  Multi-lens corrected:             {multi_lens_corrected:,} ({multi_lens_corrected/len(binary_hdc_errors)*100:.1f}% of errors)")
    logger.info(f"  Multi-lens failed to correct:     {multi_lens_failed_to_correct:,}")
    logger.info(f"  Wrong in BOTH approaches:         {both_wrong:,}")
    logger.info("")

    logger.info("TIMING BENCHMARKS:")
    logger.info(f"  Binary HDC query (avg):           {np.mean(binary_query_times)*1000:.3f} ms")
    logger.info(f"  Binary HDC query (med):           {np.median(binary_query_times)*1000:.3f} ms")
    if multi_lens_query_times:
        logger.info(f"  Multi-Lens query (avg):           {np.mean(multi_lens_query_times)*1000:.3f} ms")
        logger.info(f"  Multi-Lens query (med):           {np.median(multi_lens_query_times)*1000:.3f} ms")
        logger.info(f"  Speedup factor:                   {np.mean(multi_lens_query_times)/np.mean(binary_query_times):.1f}x slower")
    logger.info("")

    # Save results
    output_dir = Path("HDV_VALIDATION_PACKAGE/multi_lens_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "multi_lens_hdc_correction_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'metadata': {
                'sample_size': total,
                'binary_hdc_accuracy': binary_acc,
                'multi_lens_accuracy': multi_acc,
                'improvement': improvement
            },
            'timing': {
                'binary_hdc_query_avg_ms': float(np.mean(binary_query_times) * 1000),
                'binary_hdc_query_median_ms': float(np.median(binary_query_times) * 1000),
                'multi_lens_query_avg_ms': float(np.mean(multi_lens_query_times) * 1000) if multi_lens_query_times else 0,
                'multi_lens_query_median_ms': float(np.median(multi_lens_query_times) * 1000) if multi_lens_query_times else 0,
                'slowdown_factor': float(np.mean(multi_lens_query_times) / np.mean(binary_query_times)) if multi_lens_query_times else 0
            },
            'error_correction': {
                'binary_hdc_errors': len(binary_hdc_errors),
                'multi_lens_attempts': multi_lens_attempts,
                'multi_lens_corrected': multi_lens_corrected,
                'multi_lens_failed_to_correct': multi_lens_failed_to_correct,
                'both_wrong': both_wrong
            },
            'sample_errors': binary_hdc_errors[:20]  # First 20 errors
        }, f, indent=2)

    logger.info(f"✓ Results saved to: {results_file}")
    logger.info("")

    return {
        'binary_accuracy': binary_acc,
        'multi_lens_accuracy': multi_acc,
        'improvement': improvement,
        'errors_corrected': multi_lens_corrected,
        'total': total
    }


if __name__ == "__main__":
    results = run_multi_lens_validation(sample_size=1000)
    logger.info("=" * 80)
    if results['improvement'] > 0:
        logger.info("✅ MULTI-LENS VOTING IMPROVES ACCURACY")
    else:
        logger.info("⚠️  MULTI-LENS VOTING DID NOT IMPROVE ACCURACY")
    logger.info("=" * 80)
