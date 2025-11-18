# Streaming HDV Validation Report

**Date:** 2025-11-14 18:18:54

**Memory-Safe:** ✓ (streaming, no RAM bloat)

---

## Configuration

- **Dimension:** 50,000D
- **Voting Rounds:** 3
- **Test Positions:** 100

## Results

- **Accuracy:** 26.60% (25/94 correct)
- **Average Confidence:** 57.80%

⚠️ **VALIDATION MARGINAL** (accuracy 26.60% < 95%)

## Sample Results (First 20)

| Chrom | Position | Ground Truth | Prediction | Correct | Confidence | Votes |
|-------|----------|--------------|------------|---------|------------|-------|
| chr9_consensus | 40083869 | G | C | ✗ | 100.0% | {'A': 0, 'T': 0, 'G': 0, 'C': 3} |
| chr5_consensus | 100049017 | C | T | ✗ | 66.7% | {'A': 0, 'T': 2, 'G': 1, 'C': 0} |
| chr4_consensus | 40088936 | C | G | ✗ | 66.7% | {'A': 0, 'T': 1, 'G': 2, 'C': 0} |
| chr1_consensus | 123146922 | T | A | ✗ | 66.7% | {'A': 2, 'T': 0, 'G': 0, 'C': 1} |
| chr4_consensus | 49777835 | A | G | ✗ | 66.7% | {'A': 0, 'T': 0, 'G': 2, 'C': 1} |
| chr13_consensus | 95705684 | G | A | ✗ | 33.3% | {'A': 1, 'T': 1, 'G': 0, 'C': 1} |
| chr1_consensus | 183866465 | A | T | ✗ | 33.3% | {'A': 0, 'T': 1, 'G': 1, 'C': 1} |
| chr9_consensus | 44930532 | C | A | ✗ | 66.7% | {'A': 2, 'T': 0, 'G': 1, 'C': 0} |
| chr3_consensus | 195438502 | T | A | ✗ | 33.3% | {'A': 1, 'T': 1, 'G': 1, 'C': 0} |
| chr15_consensus | 5386816 | T | A | ✗ | 33.3% | {'A': 1, 'T': 1, 'G': 0, 'C': 1} |
| chr8_consensus | 7542748 | A | A | ✓ | 33.3% | {'A': 1, 'T': 0, 'G': 1, 'C': 1} |
| chr2_consensus | 190652814 | T | A | ✗ | 33.3% | {'A': 1, 'T': 0, 'G': 1, 'C': 1} |
| chrX_consensus | 94275574 | T | T | ✓ | 66.7% | {'A': 1, 'T': 2, 'G': 0, 'C': 0} |
| chr1_consensus | 148844723 | G | C | ✗ | 66.7% | {'A': 0, 'T': 1, 'G': 0, 'C': 2} |
| chr16_consensus | 36924946 | T | T | ✓ | 33.3% | {'A': 0, 'T': 1, 'G': 1, 'C': 1} |
| chr9_consensus | 43526007 | T | A | ✗ | 33.3% | {'A': 1, 'T': 1, 'G': 1, 'C': 0} |
| chr12_consensus | 36235803 | T | G | ✗ | 66.7% | {'A': 1, 'T': 0, 'G': 2, 'C': 0} |
| chr16_consensus | 29593536 | T | A | ✗ | 33.3% | {'A': 1, 'T': 1, 'G': 1, 'C': 0} |
| chr11_consensus | 13085522 | T | T | ✓ | 66.7% | {'A': 0, 'T': 2, 'G': 1, 'C': 0} |
| chr6_consensus | 59132238 | T | T | ✓ | 66.7% | {'A': 1, 'T': 2, 'G': 0, 'C': 0} |

---

**Memory footprint:** ~200 KB max (streaming validation)
