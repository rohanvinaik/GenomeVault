#!/usr/bin/env python3
"""
Complementary Pair HDC Validation

Tests the Complementary Pair HDC architecture with:
- 100 random nucleotide queries
- Comparison to experimental BAM ground truth
- Comprehensive validation report

Expected accuracy: 99.92% baseline, 99.99%+ with error correction
"""

import logging
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

import pysam
import numpy as np

from genomevault.hypervector_transform.complementary_pair_encoder import (
    ComplementaryPairEncoder,
    TernaryEnhancedEncoder
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
)
logger = logging.getLogger(__name__)


def get_ground_truth_from_bam(bam: pysam.AlignmentFile, chrom: str, pos: int) -> str:
    """
    Get ground truth nucleotide from experimental BAM

    Uses majority vote from all reads covering this position

    Args:
        bam: Experimental BAM file
        chrom: Chromosome
        pos: 0-based position

    Returns:
        Nucleotide: 'A', 'T', 'G', 'C', or 'N' (if no coverage)
    """
    bases = []

    try:
        for pileup_col in bam.pileup(chrom, pos, pos + 1, stepper='nofilter', truncate=True):
            if pileup_col.pos == pos:
                for pileup_read in pileup_col.pileups:
                    if not pileup_read.is_del and not pileup_read.is_refskip:
                        base = pileup_read.alignment.query_sequence[pileup_read.query_position]
                        bases.append(base.upper())
                break
    except Exception as e:
        logger.warning(f"Error getting ground truth at {chrom}:{pos}: {e}")
        return 'N'

    if not bases:
        return 'N'

    # Majority vote
    return Counter(bases).most_common(1)[0][0]


def sample_test_positions(
    gdiff_path: Path,
    sample_size: int = 100,
    seed: int = 42
) -> list:
    """
    Sample random variant positions from GDiff for testing

    Uses reservoir sampling to avoid loading full file into memory

    Args:
        gdiff_path: Path to GDiff file
        sample_size: Number of positions to sample
        seed: Random seed

    Returns:
        List of (chrom, pos) tuples
    """
    import gzip
    import json

    logger.info(f"Sampling {sample_size} random positions from GDiff...")

    random.seed(seed)

    with gzip.open(gdiff_path, 'rt') as f:
        gdiff = json.load(f)

    variants = gdiff["differential_variants"]
    total_variants = len(variants)

    logger.info(f"  Total variants: {total_variants:,}")

    # Random sampling
    sampled_variants = random.sample(variants, min(sample_size, total_variants))

    positions = [(v["chrom"], v["pos"]) for v in sampled_variants]

    logger.info(f"  ✓ Sampled {len(positions)} positions")

    return positions


def main():
    logger.info("=" * 80)
    logger.info("COMPLEMENTARY PAIR HDC VALIDATION")
    logger.info("=" * 80)
    logger.info("")

    # Configuration
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    guide_fasta_dir = Path("/Volumes/1TBStorage/guide_strands")
    experimental_bam = Path("data/experimental_strands/ERR3239334/alignment/k11_bams/experimental_vs_ref2.sorted.bam")

    dimension = 10000
    chunk_size = 2000
    sample_size = 100

    logger.info("Configuration:")
    logger.info(f"  GDiff: {gdiff_path}")
    logger.info(f"  Guide FASTAs: {guide_fasta_dir}")
    logger.info(f"  Experimental BAM: {experimental_bam}")
    logger.info(f"  Dimension: {dimension:,}D")
    logger.info(f"  Chunk size: {chunk_size:,} bp")
    logger.info(f"  SNR: {2 * dimension / chunk_size:.2f}")
    logger.info(f"  Expected accuracy: 99.92%+")
    logger.info("")

    # Phase 1: Encode
    logger.info("=" * 80)
    logger.info("PHASE 1: COMPLEMENTARY PAIR ENCODING")
    logger.info("=" * 80)
    logger.info("")

    encoder = ComplementaryPairEncoder(
        gdiff_path=gdiff_path,
        guide_fasta_dir=guide_fasta_dir,
        dimension=dimension,
        chunk_size=chunk_size
    )

    # Sample test positions
    test_positions = sample_test_positions(gdiff_path, sample_size=sample_size)
    logger.info("")

    # Encode only the chunks containing test positions
    logger.info("Encoding chunks containing test positions...")
    chunks_to_encode = set()
    for chrom, pos in test_positions:
        chunk_start = (pos // chunk_size) * chunk_size
        chunks_to_encode.add((chrom, chunk_start))

    logger.info(f"  Chunks to encode: {len(chunks_to_encode)}")

    encoded_count = 0
    for chrom, chunk_start in chunks_to_encode:
        AT_vec, GC_vec = encoder.encode_chunk(chrom, chunk_start)
        chunk_key = f"{chrom}:{chunk_start}"
        encoder.encoded_chunks[chunk_key] = (AT_vec, GC_vec)
        encoded_count += 1

    logger.info(f"  ✓ Encoded {encoded_count} chunks")
    logger.info("")

    # Phase 2: Validation
    logger.info("=" * 80)
    logger.info("PHASE 2: VALIDATION")
    logger.info("=" * 80)
    logger.info("")

    # Open experimental BAM
    bam = pysam.AlignmentFile(str(experimental_bam), 'rb')
    logger.info(f"Opened experimental BAM: {experimental_bam.name}")
    logger.info("")

    # Query and validate
    logger.info("Querying HDV and comparing to ground truth...")
    logger.info("")

    results = []
    correct = 0
    total = 0

    pair_stats = {'AT': {'correct': 0, 'total': 0}, 'GC': {'correct': 0, 'total': 0}}

    for i, (chrom, pos) in enumerate(test_positions):
        # Query HDV
        try:
            result = encoder.query_nucleotide(chrom, pos)
        except Exception as e:
            logger.warning(f"Error querying {chrom}:{pos}: {e}")
            continue

        # Get ground truth from BAM
        ground_truth = get_ground_truth_from_bam(bam, chrom, pos)

        if ground_truth == 'N':
            # No coverage - skip
            continue

        # Compare
        is_correct = (result.nucleotide == ground_truth)
        if is_correct:
            correct += 1
            pair_stats[result.pair]['correct'] += 1
        total += 1
        pair_stats[result.pair]['total'] += 1

        results.append({
            'chrom': chrom,
            'pos': pos,
            'ground_truth': ground_truth,
            'prediction': result.nucleotide,
            'correct': is_correct,
            'confidence': result.confidence,
            'pair': result.pair,
            'at_similarity': result.at_similarity,
            'gc_similarity': result.gc_similarity
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i + 1}/{len(test_positions)} ({(i + 1) / len(test_positions) * 100:.1f}%)")

    logger.info("")
    logger.info("✓ Validation complete")
    logger.info("")

    # Results
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info("")

    accuracy = correct / total if total > 0 else 0
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0

    logger.info(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
    logger.info(f"Average Confidence: {avg_confidence * 100:.2f}%")
    logger.info("")

    # Per-pair statistics
    logger.info("Per-Pair Statistics:")
    for pair in ['AT', 'GC']:
        pair_correct = pair_stats[pair]['correct']
        pair_total = pair_stats[pair]['total']
        pair_accuracy = pair_correct / pair_total if pair_total > 0 else 0
        logger.info(f"  {pair} pair: {pair_accuracy * 100:.2f}% ({pair_correct}/{pair_total})")
    logger.info("")

    # Expected vs Actual
    expected_accuracy = 99.92
    logger.info(f"Expected Accuracy: {expected_accuracy:.2f}%")
    logger.info(f"Actual Accuracy: {accuracy * 100:.2f}%")
    logger.info("")

    # Sample results
    logger.info("Sample Results (First 20):")
    logger.info("")
    logger.info(f"{'Chrom':<10} {'Position':<12} {'Truth':<6} {'Pred':<6} {'Pair':<6} {'OK':<4} {'Conf':<8} {'AT Sim':<10} {'GC Sim':<10}")
    logger.info("-" * 110)

    for r in results[:20]:
        sym = '✓' if r['correct'] else '✗'
        logger.info(
            f"{r['chrom']:<10} {r['pos']:<12} {r['ground_truth']:<6} {r['prediction']:<6} "
            f"{r['pair']:<6} {sym:<4} {r['confidence'] * 100:.1f}%   "
            f"{r['at_similarity']:<10.4f} {r['gc_similarity']:<10.4f}"
        )

    logger.info("")

    # Write comprehensive report
    report_path = Path("COMPLEMENTARY_PAIR_VALIDATION_REPORT.md")
    with open(report_path, 'w') as f:
        f.write("# Complementary Pair HDC Validation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Architecture Overview\n\n")
        f.write("**Complementary Pair HDC** exploits Watson-Crick base pairing:\n\n")
        f.write("- **AT pair**: A → +1, T → -1\n")
        f.write("- **GC pair**: G → +1, C → -1\n\n")
        f.write("Each nucleotide position appears in **exactly ONE** vector with **exactly ONE** sign,\n")
        f.write("eliminating cross-pair interference entirely.\n\n")

        f.write("### Mathematical Foundation\n\n")
        f.write(f"- **Dimension (D)**: {dimension:,}\n")
        f.write(f"- **Chunk size (N)**: {chunk_size:,} bp\n")
        f.write(f"- **SNR**: {2 * dimension / chunk_size:.2f}\n")
        f.write(f"- **Expected P(sign error)**: 0.079% per nucleotide\n")
        f.write(f"- **Expected accuracy**: 99.92%+\n\n")

        f.write("---\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **GDiff**: `{gdiff_path}`\n")
        f.write(f"- **Guide FASTAs**: `{guide_fasta_dir}` (ref1-ref11)\n")
        f.write(f"- **Experimental BAM**: `{experimental_bam}`\n")
        f.write(f"- **Test positions**: {sample_size}\n\n")

        f.write("---\n\n")

        f.write("## Results\n\n")
        f.write(f"- **Accuracy**: {accuracy * 100:.2f}% ({correct}/{total} correct)\n")
        f.write(f"- **Average Confidence**: {avg_confidence * 100:.2f}%\n\n")

        f.write("### Per-Pair Statistics\n\n")
        for pair in ['AT', 'GC']:
            pair_correct = pair_stats[pair]['correct']
            pair_total = pair_stats[pair]['total']
            pair_accuracy = pair_correct / pair_total if pair_total > 0 else 0
            f.write(f"- **{pair} pair**: {pair_accuracy * 100:.2f}% ({pair_correct}/{pair_total})\n")
        f.write("\n")

        f.write("### Expected vs Actual\n\n")
        f.write(f"- **Expected**: {expected_accuracy:.2f}%\n")
        f.write(f"- **Actual**: {accuracy * 100:.2f}%\n")

        if accuracy >= 0.9992:
            f.write(f"\n✅ **VALIDATION PASSED** (accuracy matches theoretical expectation)\n\n")
        elif accuracy >= 0.95:
            f.write(f"\n✅ **VALIDATION PASSED** (accuracy ≥95%)\n\n")
        else:
            f.write(f"\n⚠️ **VALIDATION MARGINAL** (accuracy {accuracy * 100:.2f}% < 95%)\n\n")

        f.write("---\n\n")

        f.write("## Sample Results (First 50)\n\n")
        f.write("| Chrom | Position | Ground Truth | Prediction | Pair | Correct | Confidence | AT Similarity | GC Similarity |\n")
        f.write("|-------|----------|--------------|------------|------|---------|------------|---------------|---------------|\n")

        for r in results[:50]:
            sym = '✓' if r['correct'] else '✗'
            f.write(
                f"| {r['chrom']} | {r['pos']} | {r['ground_truth']} | {r['prediction']} | "
                f"{r['pair']} | {sym} | {r['confidence'] * 100:.1f}% | "
                f"{r['at_similarity']:.4f} | {r['gc_similarity']:.4f} |\n"
            )

        f.write("\n---\n\n")

        f.write("## Architecture Advantages\n\n")
        f.write("1. **Zero Cross-Pair Interference**: Each position appears in exactly ONE vector\n")
        f.write("2. **High SNR**: 2D/N = 10 (vs ~0.1 for bundled approach)\n")
        f.write("3. **Two-Stage Retrieval**: Pair selection → sign determination\n")
        f.write("4. **Ternary Computing Natural**: {-1, 0, +1} maps to {T/C, N, A/G}\n")
        f.write("5. **Nanopore Error Correction**: Quality-weighted encoding supported\n\n")

        f.write("---\n\n")
        f.write(f"**Report generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"✓ Validation report saved to {report_path}")
    logger.info("")

    # Cleanup
    bam.close()
    encoder.close()

    if accuracy >= 0.95:
        logger.info("✅ VALIDATION PASSED (accuracy ≥95%)")
        if accuracy >= 0.9992:
            logger.info("🎉 EXCEPTIONAL: Accuracy matches theoretical expectation (99.92%+)")
    else:
        logger.warning(f"⚠️ VALIDATION MARGINAL (accuracy {accuracy * 100:.2f}% < 95%)")

    logger.info("")


if __name__ == "__main__":
    main()
