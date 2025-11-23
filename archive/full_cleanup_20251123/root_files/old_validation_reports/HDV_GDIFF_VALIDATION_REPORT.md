# HDV vs GDiff Validation Report

**Date:** 2025-11-14 19:56:29

---

## Validation Methodology

This validation tests whether the Complementary Pair HDV encoding can accurately
reconstruct the experimental genome as encoded in the GDiff file.

**Ground Truth:** GDiff variant ALT fields (not experimental BAM)

**Test:** Encode GDiff → HDV → Query HDV → Compare to GDiff ALT

---

## Configuration

- **Dimension**: 10,000D
- **Chunk Size**: 2,000 bp
- **SNR**: 10.00
- **Test Positions**: 20

---

## Results

- **Accuracy**: 95.00% (19/20 correct)
- **Average Confidence**: 79.10%

### Per-Pair Statistics

- **AT pair**: 100.00% (6/6)
- **GC pair**: 92.86% (13/14)

### Expected vs Actual

- **Expected**: 99.92%
- **Actual**: 95.00%

✅ **VALIDATION PASSED** (accuracy ≥95%)

---

## Sample Results (First 50)

| Chrom | Position | Ground Truth | Prediction | Pair | Correct | Confidence | AT Similarity | GC Similarity |
|-------|----------|--------------|------------|------|---------|------------|---------------|---------------|
| chr4_consensus | 42635249 | C | C | GC | ✓ | 79.7% | -0.5942 | -2.3378 |
| chr18_consensus | 16527030 | G | G | GC | ✓ | 93.3% | 0.3473 | 4.8428 |
| chr1_consensus | 24202565 | G | G | GC | ✓ | 81.0% | -0.8502 | 3.6341 |
| chr6_consensus | 18701755 | G | G | GC | ✓ | 96.2% | 0.1397 | 3.5426 |
| chr1_consensus | 124910100 | A | A | AT | ✓ | 67.5% | 3.4887 | -1.6834 |
| chr18_consensus | 16058852 | T | T | AT | ✓ | 76.7% | -2.1258 | 0.6455 |
| chr13_consensus | 57158323 | G | G | GC | ✓ | 97.7% | 0.0757 | 3.2495 |
| chr17_consensus | 46471798 | A | A | AT | ✓ | 87.0% | 4.0516 | -0.6065 |
| chr3_consensus | 111308789 | C | C | GC | ✓ | 90.1% | 0.6493 | -5.8913 |
| chr1_consensus | 124127350 | C | C | GC | ✓ | 50.6% | -1.8252 | -1.8659 |
| chr1_consensus | 237038348 | T | T | AT | ✓ | 67.7% | -1.1646 | -0.5545 |
| chr16_consensus | 32742646 | T | T | AT | ✓ | 92.9% | -4.3104 | -0.3277 |
| chr9_consensus | 44417336 | A | C | GC | ✗ | 52.2% | 2.0084 | -2.1964 |
| chr21_consensus | 2045683 | G | G | GC | ✓ | 81.1% | -0.8396 | 3.5960 |
| chr22_consensus | 13439761 | C | C | GC | ✓ | 82.3% | 0.7397 | -3.4409 |
| chr12_consensus | 36199488 | G | G | GC | ✓ | 94.0% | -0.2177 | 3.3883 |
| chr6_consensus | 63316 | G | G | GC | ✓ | 78.5% | -0.9700 | 3.5466 |
| chr5_consensus | 69546578 | C | C | GC | ✓ | 63.7% | -1.7480 | -3.0723 |
| chr5_consensus | 171619918 | G | G | GC | ✓ | 67.9% | -0.9955 | 2.1079 |
| chr9_consensus | 67227495 | T | T | AT | ✓ | 81.8% | -2.4207 | 0.5382 |

---

**Report generated:** 2025-11-14 19:56:29
