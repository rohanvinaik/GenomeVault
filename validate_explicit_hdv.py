#!/usr/bin/env python3
"""
Validate Explicit Nucleotide HDV Encoding

Test with actual full nucleotide resolution:
- 50,000D vectors (high dimension for low interference)
- 10 KB regions (smaller for less bundling)
- Direct position→nucleotide mapping (explicit storage)
"""

import logging
import random
from pathlib import Path
from datetime import datetime

import pysam
import numpy as np

from genomevault.hypervector_transform.nucleotide_hdv_explicit import ExplicitNucleotideHDV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("EXPLICIT NUCLEOTIDE HDV VALIDATION")
    logger.info("=" * 80)
    logger.info("")

    # Configuration
    gdiff_path = Path("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")
    experimental_bam_dir = Path("data/experimental_strands/ERR3239334/alignment/k11_bams")

    logger.info("Configuration:")
    logger.info(f"  GDiff: {gdiff_path}")
    logger.info(f"  Experimental BAMs: {experimental_bam_dir}")
    logger.info("")

    # Phase 1: Encode
    logger.info("=" * 80)
    logger.info("PHASE 1: EXPLICIT ENCODING")
    logger.info("=" * 80)
    logger.info("")

    encoder = ExplicitNucleotideHDV(
        gdiff_path=gdiff_path,
        dimension=50000,  # HIGH dimension
        region_size=10_000,  # SMALL regions
    )

    encoder.encode()

    # Save
    output_path = Path("genome_explicit_hdv.npz")
    encoder.save(output_path)
    logger.info("")

    # Phase 2: Validation
    logger.info("=" * 80)
    logger.info("PHASE 2: VALIDATION")
    logger.info("=" * 80)
    logger.info("")

    # Select 100 random variant positions
    logger.info("Selecting 100 random variant positions...")
    variants = encoder.gdiff["differential_variants"]
    test_variants = random.sample(variants, min(100, len(variants)))
    logger.info(f"  ✓ Selected {len(test_variants)} test positions")
    logger.info("")

    # Find experimental BAM
    experimental_bam = experimental_bam_dir / "experimental_vs_ref2.sorted.bam"
    bam = pysam.AlignmentFile(str(experimental_bam), 'rb')

    logger.info(f"Using experimental BAM: {experimental_bam.name}")
    logger.info("")

    # Query and validate
    results = []
    correct = 0
    total = 0

    for i, variant in enumerate(test_variants):
        chrom = variant["chrom"]
        pos = variant["pos"]

        # Query HDV
        result = encoder.query_with_voting(chrom, pos, num_votes=3)

        # Get ground truth from BAM
        try:
            pileup_col = next(bam.pileup(chrom, pos, pos + 1, stepper='nofilter',  truncate=True))
            bases = []
            for pileup_read in pileup_col.pileups:
                if not pileup_read.is_del and not pileup_read.is_refskip:
                    base = pileup_read.alignment.query_sequence[pileup_read.query_position]
                    bases.append(base.upper())

            if bases:
                # Majority vote from BAM
                from collections import Counter
                ground_truth = Counter(bases).most_common(1)[0][0]

                # Compare
                is_correct = (result.nucleotide == ground_truth)
                if is_correct:
                    correct += 1
                total += 1

                results.append({
                    'chrom': chrom,
                    'pos': pos,
                    'ground_truth': ground_truth,
                    'prediction': result.nucleotide,
                    'correct': is_correct,
                    'confidence': result.confidence,
                    'votes': result.votes
                })
        except StopIteration:
            # No coverage at this position
            continue

        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i + 1}/{len(test_variants)} ({(i + 1) / len(test_variants) * 100:.1f}%)")

    logger.info("")
    logger.info("✓ Validation complete")
    logger.info("")

    # Generate report
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info("")

    accuracy = correct / total if total > 0 else 0
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0

    logger.info(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
    logger.info(f"Average Confidence: {avg_confidence * 100:.2f}%")
    logger.info("")

    # Sample results
    logger.info("Sample Results (First 20):")
    logger.info("")
    logger.info(f"{'Chrom':<20} {'Position':<12} {'Truth':<6} {'Pred':<6} {'Correct':<8} {'Conf':<8} {'Votes'}")
    logger.info("-" * 100)

    for r in results[:20]:
        chrom = r['chrom']
        pos = r['pos']
        truth = r['ground_truth']
        pred = r['prediction']
        correct_sym = '✓' if r['correct'] else '✗'
        conf = f"{r['confidence'] * 100:.1f}%"
        votes = str(r['votes'])

        logger.info(f"{chrom:<20} {pos:<12} {truth:<6} {pred:<6} {correct_sym:<8} {conf:<8} {votes}")

    logger.info("")

    # Write report
    report_path = Path("EXPLICIT_HDV_VALIDATION_REPORT.md")
    with open(report_path, 'w') as f:
        f.write(f"# Explicit Nucleotide HDV Validation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"---\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- **Dimension:** 50,000D\n")
        f.write(f"- **Region Size:** 10 KB\n")
        f.write(f"- **Encoding:** Explicit position→nucleotide mapping\n")
        f.write(f"- **Voting Rounds:** 3\n\n")
        f.write(f"## Results\n\n")
        f.write(f"- **Accuracy:** {accuracy * 100:.2f}% ({correct}/{total} correct)\n")
        f.write(f"- **Average Confidence:** {avg_confidence * 100:.2f}%\n\n")
        f.write(f"## Sample Results (First 20)\n\n")
        f.write(f"| Chrom | Position | Ground Truth | HDV Prediction | Correct | Confidence | Votes |\n")
        f.write(f"|-------|----------|--------------|----------------|---------|------------|-------|\n")

        for r in results[:20]:
            correct_sym = '✓' if r['correct'] else '✗'
            f.write(f"| {r['chrom']} | {r['pos']} | {r['ground_truth']} | {r['prediction']} | {correct_sym} | {r['confidence'] * 100:.1f}% | {r['votes']} |\n")

        f.write(f"\n---\n\n")
        f.write(f"**Report generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"✓ Validation report saved to {report_path}")
    logger.info("")

    # Cleanup
    bam.close()
    encoder.close()

    if accuracy >= 0.95:
        logger.info("✅ VALIDATION PASSED (accuracy ≥95%)")
    else:
        logger.warning(f"⚠️ VALIDATION MARGINAL (accuracy {accuracy * 100:.2f}% < 95%)")


if __name__ == "__main__":
    main()
