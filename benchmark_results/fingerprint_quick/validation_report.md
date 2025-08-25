# HDC Fingerprint Validation Report

Generated: 2025-08-24 22:14:44
Protocol: subject_disjoint
Folds: 2
Seed: 42

## Cohort Information
- Subjects: 50
- Families: 12
- Batches: 5
- Samples per subject: 3

## Aggregate Performance
- **AUC**: 1.000 [1.000, 1.000]
- **EER**: 0.000 [0.000, 0.000]
- **d-prime**: 9.22 [9.21, 9.23]
- **Score Margin**: 0.079 [0.072, 0.086]

## Validation Checks ✓
- **Label Shuffle AUC**: 0.530 (should be ~0.5)
- **Duplicate Rate**: 0.000 (should be ~0)

## Per-Fold Results
| Fold | AUC | CI | EER | d' | Margin | Genuine μ±σ | Impostor μ±σ |
|------|-----|-------|-----|-----|---------|-------------|--------------|
| 0 | 1.000 | [1.000, 1.000] | 0.000 | 9.2 | 0.065 | 0.974±0.004 | 0.544±0.066 |
| 1 | 1.000 | [1.000, 1.000] | 0.000 | 9.2 | 0.093 | 0.974±0.005 | 0.543±0.066 |

## Sample Sizes
- Total genuine pairs: 144
- Total impostor pairs: 552
- Balance ratio: 0.261