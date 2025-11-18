# Complementary Pair HDC Validation Report

**Date:** 2025-11-14 19:41:32

---

## Architecture Overview

**Complementary Pair HDC** exploits Watson-Crick base pairing:

- **AT pair**: A → +1, T → -1
- **GC pair**: G → +1, C → -1

Each nucleotide position appears in **exactly ONE** vector with **exactly ONE** sign,
eliminating cross-pair interference entirely.

### Mathematical Foundation

- **Dimension (D)**: 10,000
- **Chunk size (N)**: 2,000 bp
- **SNR**: 10.00
- **Expected P(sign error)**: 0.079% per nucleotide
- **Expected accuracy**: 99.92%+

---

## Configuration

- **GDiff**: `data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz`
- **Guide FASTAs**: `/Volumes/1TBStorage/guide_strands` (ref1-ref11)
- **Experimental BAM**: `data/experimental_strands/ERR3239334/alignment/k11_bams/experimental_vs_ref2.sorted.bam`
- **Test positions**: 100

---

## Results

- **Accuracy**: 26.88% (25/93 correct)
- **Average Confidence**: 80.41%

### Per-Pair Statistics

- **AT pair**: 27.91% (12/43)
- **GC pair**: 26.00% (13/50)

### Expected vs Actual

- **Expected**: 99.92%
- **Actual**: 26.88%

⚠️ **VALIDATION MARGINAL** (accuracy 26.88% < 95%)

---

## Sample Results (First 50)

| Chrom | Position | Ground Truth | Prediction | Pair | Correct | Confidence | AT Similarity | GC Similarity |
|-------|----------|--------------|------------|------|---------|------------|---------------|---------------|
| chr14_consensus | 86859883 | C | C | GC | ✓ | 81.9% | -0.5825 | -2.6401 |
| chr1_consensus | 122821917 | C | A | AT | ✗ | 56.1% | 1.3460 | -1.0516 |
| chr17_consensus | 64657882 | T | G | GC | ✗ | 84.6% | -0.7044 | 3.8629 |
| chr6_consensus | 24761177 | G | G | GC | ✓ | 88.9% | -0.4472 | 3.5676 |
| chr5_consensus | 70041986 | C | T | AT | ✗ | 66.8% | -1.8104 | 0.9014 |
| chr4_consensus | 183474937 | G | G | GC | ✓ | 75.3% | -0.8083 | 2.4680 |
| chr2_consensus | 177989055 | T | A | AT | ✗ | 68.0% | 2.9589 | -1.3942 |
| chr18_consensus | 18925911 | T | A | AT | ✗ | 90.9% | 2.9298 | -0.2921 |
| chr2_consensus | 92295541 | A | G | GC | ✗ | 58.6% | -1.3350 | 1.8883 |
| chr16_consensus | 30196128 | G | C | GC | ✗ | 64.2% | -1.0142 | -1.8213 |
| chr18_consensus | 19981276 | T | A | AT | ✗ | 69.3% | 3.4618 | 1.5331 |
| chr12_consensus | 31113556 | T | G | GC | ✗ | 95.5% | 0.1292 | 2.7394 |
| chr1_consensus | 236689016 | T | G | GC | ✗ | 87.1% | -0.6011 | 4.0466 |
| chr13_consensus | 1551990 | A | A | AT | ✓ | 90.6% | 4.8261 | -0.5028 |
| chr9_consensus | 65232758 | T | C | GC | ✗ | 82.5% | 0.8624 | -4.0759 |
| chr1_consensus | 123496886 | A | A | AT | ✓ | 88.1% | 3.6840 | -0.4976 |
| chr1_consensus | 123296809 | A | T | AT | ✗ | 60.6% | -1.8210 | -1.1859 |
| chr2_consensus | 46664775 | C | G | GC | ✗ | 86.3% | -0.6359 | 4.0111 |
| chr4_consensus | 129408122 | A | C | GC | ✗ | 75.8% | 1.2155 | -3.8081 |
| chr5_consensus | 5392050 | T | T | AT | ✓ | 83.3% | -2.5405 | 0.5084 |
| chr11_consensus | 52836613 | G | G | GC | ✓ | 85.0% | 0.8661 | 4.8991 |
| chr13_consensus | 48493019 | T | C | GC | ✗ | 81.6% | -0.5970 | -2.6444 |
| chr1_consensus | 122969437 | T | T | AT | ✓ | 97.2% | -2.9569 | 0.0849 |
| chr12_consensus | 36755559 | C | G | GC | ✗ | 74.4% | -1.2552 | 3.6399 |
| chr4_consensus | 57854972 | T | A | AT | ✗ | 97.7% | 3.4999 | -0.0833 |
| chr17_consensus | 46966685 | A | G | GC | ✗ | 70.6% | 1.3106 | 3.1442 |
| chr15_consensus | 36917040 | G | A | AT | ✗ | 75.4% | 4.2953 | -1.4015 |
| chr17_consensus | 26045234 | T | T | AT | ✓ | 89.6% | -4.4348 | -0.5156 |
| chr12_consensus | 28990831 | A | C | GC | ✗ | 90.4% | 0.1148 | -1.0778 |
| chr9_consensus | 64618849 | C | C | GC | ✓ | 80.6% | -0.9513 | -3.9523 |
| chr4_consensus | 163931351 | A | G | GC | ✗ | 77.2% | -0.9192 | 3.1123 |
| chr9_consensus | 108241511 | C | G | GC | ✗ | 81.6% | 0.8274 | 3.6583 |
| chr13_consensus | 1359259 | T | A | AT | ✗ | 75.2% | 3.2994 | -1.0904 |
| chr21_consensus | 8032601 | C | T | AT | ✗ | 89.1% | -2.9177 | 0.3560 |
| chrX_consensus | 92698010 | A | T | AT | ✗ | 86.0% | -2.5241 | -0.4114 |
| chr1_consensus | 46890739 | A | G | GC | ✗ | 80.6% | -1.0060 | 4.1830 |
| chr18_consensus | 70210646 | A | C | GC | ✗ | 76.8% | 1.1560 | -3.8263 |
| chr21_consensus | 3165660 | A | A | AT | ✓ | 91.2% | 3.2821 | 0.3155 |
| chr3_consensus | 91248019 | G | G | GC | ✓ | 84.7% | 0.3645 | 2.0109 |
| chr17_consensus | 22373667 | A | T | AT | ✗ | 87.5% | -4.0152 | -0.5736 |
| chr9_consensus | 65486687 | C | T | AT | ✗ | 90.9% | -3.3934 | -0.3385 |
| chr7_consensus | 112465564 | T | A | AT | ✗ | 97.7% | 2.6999 | -0.0639 |
| chr6_consensus | 32572639 | A | C | GC | ✗ | 89.8% | 0.5404 | -4.7579 |
| chr3_consensus | 56503786 | A | C | GC | ✗ | 94.3% | 0.3494 | -5.7745 |
| chr4_consensus | 116620847 | G | A | AT | ✗ | 75.1% | 1.0961 | 0.3632 |
| chr19_consensus | 6977425 | A | G | GC | ✗ | 68.2% | 2.4238 | 5.2057 |
| chr7_consensus | 77015012 | A | T | AT | ✗ | 67.2% | -1.9807 | -0.9677 |
| chr2_consensus | 92274655 | C | A | AT | ✗ | 87.7% | 1.3939 | -0.1946 |
| chr2_consensus | 41745019 | T | C | GC | ✗ | 96.3% | 0.0985 | -2.5372 |
| chr8_consensus | 70156274 | G | C | GC | ✗ | 79.0% | 1.0053 | -3.7836 |

---

## Architecture Advantages

1. **Zero Cross-Pair Interference**: Each position appears in exactly ONE vector
2. **High SNR**: 2D/N = 10 (vs ~0.1 for bundled approach)
3. **Two-Stage Retrieval**: Pair selection → sign determination
4. **Ternary Computing Natural**: {-1, 0, +1} maps to {T/C, N, A/G}
5. **Nanopore Error Correction**: Quality-weighted encoding supported

---

**Report generated:** 2025-11-14 19:41:32
