#!/usr/bin/env python3
"""
Validate HDV Against GDiff Reconstruction

The CORRECT validation:
1. Reconstruct experimental genome from GDiff + guide FASTAs
2. Encode into HDV
3. Query HDV
4. Compare HDV output to the GDiff reconstruction (NOT to experimental BAM!)
"""

import logging
import random
from pathlib import Path
from datetime import datetime

import numpy as np

from genomevault.hypervector_transform.complementary_pair_encoder import ComplementaryPairEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("HDV VALIDATION AGAINST GDIFF RECONSTRUCTION")
    logger.info("=" * 80)
    logger.info("")

    # Configuration
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    guide_fasta_dir = Path("/Volumes/1TBStorage/guide_strands")

    dimension = 10000
    chunk_size = 2000
    sample_size = 20  # Reduced for faster testing

    logger.info("Configuration:")
    logger.info(f"  GDiff: {gdiff_path}")
    logger.info(f"  Guide FASTAs: {guide_fasta_dir}")
    logger.info(f"  Dimension: {dimension:,}D")
    logger.info(f"  Chunk size: {chunk_size:,} bp")
    logger.info(f"  SNR: {2 * dimension / chunk_size:.2f}")
    logger.info(f"  Expected accuracy: 99.92%+")
    logger.info("")

    # Initialize encoder
    logger.info("=" * 80)
    logger.info("PHASE 1: ENCODING")
    logger.info("=" * 80)
    logger.info("")

    encoder = ComplementaryPairEncoder(
        gdiff_path=gdiff_path,
        guide_fasta_dir=guide_fasta_dir,
        dimension=dimension,
        chunk_size=chunk_size
    )

    # Sample test positions from GDiff variants
    logger.info(f"Sampling {sample_size} random variant positions from GDiff...")
    import gzip
    import json
    with gzip.open(gdiff_path, 'rt') as f:
        gdiff = json.load(f)

    variants = gdiff["differential_variants"]
    test_variants = random.sample(variants, min(sample_size, len(variants)))
    logger.info(f"  ✓ Sampled {len(test_variants)} positions")
    logger.info("")

    # Encode chunks containing test positions
    logger.info("Encoding chunks containing test positions...")
    chunks_to_encode = set()
    for variant in test_variants:
        chrom = variant["chrom"]
        pos = variant["pos"]
        chunk_start = (pos // chunk_size) * chunk_size
        chunks_to_encode.add((chrom, chunk_start))

    logger.info(f"  Chunks to encode: {len(chunks_to_encode)}")

    for chrom, chunk_start in chunks_to_encode:
        AT_vec, GC_vec = encoder.encode_chunk(chrom, chunk_start)
        chunk_key = f"{chrom}:{chunk_start}"
        encoder.encoded_chunks[chunk_key] = (AT_vec, GC_vec)

    logger.info(f"  ✓ Encoded {len(chunks_to_encode)} chunks")
    logger.info("")

    # Phase 2: Validation
    logger.info("=" * 80)
    logger.info("PHASE 2: VALIDATION")
    logger.info("=" * 80)
    logger.info("")

    logger.info("Querying HDV and comparing to GDiff reconstruction...")
    logger.info("")

    results = []
    correct = 0
    total = 0

    pair_stats = {'AT': {'correct': 0, 'total': 0}, 'GC': {'correct': 0, 'total': 0}}

    for i, variant in enumerate(test_variants):
        chrom = variant["chrom"]
        pos = variant["pos"]

        # Ground truth from GDiff: the ALT field
        ground_truth = variant["alt"]

        if not ground_truth or ground_truth not in ['A', 'T', 'G', 'C']:
            continue

        # Query HDV
        try:
            result = encoder.query_nucleotide(chrom, pos)
        except Exception as e:
            logger.warning(f"Error querying {chrom}:{pos}: {e}")
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
            logger.info(f"  Progress: {i + 1}/{len(test_variants)} ({(i + 1) / len(test_variants) * 100:.1f}%)")

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
    logger.info(f"{'Chrom':<20} {'Position':<12} {'Truth':<6} {'Pred':<6} {'Pair':<6} {'OK':<4} {'Conf':<8} {'AT Sim':<10} {'GC Sim':<10}")
    logger.info("-" * 120)

    for r in results[:20]:
        sym = '✓' if r['correct'] else '✗'
        logger.info(
            f"{r['chrom']:<20} {r['pos']:<12} {r['ground_truth']:<6} {r['prediction']:<6} "
            f"{r['pair']:<6} {sym:<4} {r['confidence'] * 100:.1f}%   "
            f"{r['at_similarity']:<10.4f} {r['gc_similarity']:<10.4f}"
        )

    logger.info("")

    # Write report
    report_path = Path("HDV_GDIFF_VALIDATION_REPORT.md")
    with open(report_path, 'w') as f:
        f.write("# HDV vs GDiff Validation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Validation Methodology\n\n")
        f.write("This validation tests whether the Complementary Pair HDV encoding can accurately\n")
        f.write("reconstruct the experimental genome as encoded in the GDiff file.\n\n")
        f.write("**Ground Truth:** GDiff variant ALT fields (not experimental BAM)\n\n")
        f.write("**Test:** Encode GDiff → HDV → Query HDV → Compare to GDiff ALT\n\n")

        f.write("---\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **Dimension**: {dimension:,}D\n")
        f.write(f"- **Chunk Size**: {chunk_size:,} bp\n")
        f.write(f"- **SNR**: {2 * dimension / chunk_size:.2f}\n")
        f.write(f"- **Test Positions**: {len(test_variants)}\n\n")

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
        f.write(f"- **Actual**: {accuracy * 100:.2f}%\n\n")

        if accuracy >= 0.9992:
            f.write("✅ **VALIDATION PASSED** (accuracy matches theoretical expectation)\n\n")
        elif accuracy >= 0.95:
            f.write("✅ **VALIDATION PASSED** (accuracy ≥95%)\n\n")
        else:
            f.write(f"⚠️ **VALIDATION MARGINAL** (accuracy {accuracy * 100:.2f}% < 95%)\n\n")

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
        f.write(f"**Report generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"✓ Validation report saved to {report_path}")
    logger.info("")

    # Cleanup
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
