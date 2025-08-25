# HDC Fingerprint Validation Report

Generated: 2025-08-24 22:17:44
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
- **d-prime**: 27.88 [27.00, 37.66]
- **Score Margin**: 0.105 [0.100, 0.116]

## Validation Checks ✓
- **Label Shuffle AUC**: 0.516 (should be ~0.5)
- **Duplicate Rate**: 0.000 (should be ~0)

## Per-Fold Results
| Fold | AUC | CI | EER | d' | Margin | Genuine μ±σ | Impostor μ±σ |
|------|-----|-------|-----|-----|---------|-------------|--------------|
| 0 | 1.000 | [1.000, 1.000] | 0.000 | 54.2 | 0.361 | 0.974±0.005 | 0.526±0.011 |
| 1 | 1.000 | [1.000, 1.000] | 0.000 | 27.0 | 0.089 | 0.974±0.005 | 0.527±0.023 |
| 2 | 1.000 | [1.000, 1.000] | 0.000 | 25.3 | 0.116 | 0.974±0.004 | 0.529±0.024 |
| 3 | 1.000 | [1.000, 1.000] | 0.000 | 37.7 | 0.105 | 0.974±0.006 | 0.526±0.016 |
| 4 | 1.000 | [1.000, 1.000] | 0.000 | 27.9 | 0.100 | 0.975±0.004 | 0.528±0.022 |

## Sample Sizes
- Total genuine pairs: 2,000
- Total impostor pairs: 3,900
- Balance ratio: 0.513