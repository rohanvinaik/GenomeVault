# HDC Fingerprint Validation Report

Generated: 2025-08-24 22:05:27
Protocol: subject_disjoint
Folds: 5
Seed: 42

## Cohort Information
- Subjects: 200
- Families: 50
- Batches: 10
- Samples per subject: 5

## Aggregate Performance
- **AUC**: 1.000 [1.000, 1.000]
- **EER**: 0.000 [0.000, 0.000]
- **d-prime**: 27.94 [27.08, 38.50]
- **Score Margin**: 0.074 [0.066, 0.079]

## Validation Checks ✓
- **Label Shuffle AUC**: 0.482 (should be ~0.5)
- **Duplicate Rate**: 0.000 (should be ~0)

## Per-Fold Results
| Fold | AUC | CI | EER | d' | Margin | Genuine μ±σ | Impostor μ±σ |
|------|-----|-------|-----|-----|---------|-------------|--------------|
| 0 | 1.000 | [1.000, 1.000] | 0.000 | 55.9 | 0.266 | 0.984±0.003 | 0.674±0.007 |
| 1 | 1.000 | [1.000, 1.000] | 0.000 | 27.1 | 0.059 | 0.984±0.003 | 0.674±0.016 |
| 2 | 1.000 | [1.000, 1.000] | 0.000 | 25.6 | 0.079 | 0.984±0.003 | 0.673±0.017 |
| 3 | 1.000 | [1.000, 1.000] | 0.000 | 38.5 | 0.074 | 0.984±0.003 | 0.673±0.011 |
| 4 | 1.000 | [1.000, 1.000] | 0.000 | 27.9 | 0.066 | 0.984±0.003 | 0.674±0.015 |

## Sample Sizes
- Total genuine pairs: 2,000
- Total impostor pairs: 3,900
- Balance ratio: 0.513