# HDC Fingerprint Validation Report

Generated: 2025-08-24 22:19:18
Protocol: LBxO
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
- **d-prime**: 16.74 [16.49, 16.82]
- **Score Margin**: 0.103 [0.092, 0.129]

## Validation Checks ✓
- **Label Shuffle AUC**: 0.490 (should be ~0.5)
- **Duplicate Rate**: 0.000 (should be ~0)

## Per-Fold Results
| Fold | AUC | CI | EER | d' | Margin | Genuine μ±σ | Impostor μ±σ |
|------|-----|-------|-----|-----|---------|-------------|--------------|
| 0 | 1.000 | [1.000, 1.000] | 0.000 | 15.1 | 0.092 | 0.974±0.005 | 0.533±0.041 |
| 1 | 1.000 | [1.000, 1.000] | 0.000 | 16.7 | 0.129 | 0.975±0.005 | 0.532±0.037 |
| 2 | 1.000 | [1.000, 1.000] | 0.000 | 16.5 | 0.076 | 0.974±0.006 | 0.528±0.038 |
| 3 | 1.000 | [1.000, 1.000] | 0.000 | 16.8 | 0.103 | 0.974±0.005 | 0.530±0.037 |
| 4 | 1.000 | [1.000, 1.000] | 0.000 | 19.3 | 0.133 | 0.975±0.004 | 0.530±0.032 |

## Sample Sizes
- Total genuine pairs: 970
- Total impostor pairs: 903
- Balance ratio: 1.074