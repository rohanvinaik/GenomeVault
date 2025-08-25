# HDC Fingerprint Validation Report

Generated: 2025-08-24 22:08:57
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
- **d-prime**: 9.14 [9.13, 9.14]
- **Score Margin**: 0.049 [0.044, 0.054]

## Validation Checks ✓
- **Label Shuffle AUC**: 0.444 (should be ~0.5)
- **Duplicate Rate**: 0.000 (should be ~0)

## Per-Fold Results
| Fold | AUC | CI | EER | d' | Margin | Genuine μ±σ | Impostor μ±σ |
|------|-----|-------|-----|-----|---------|-------------|--------------|
| 0 | 1.000 | [1.000, 1.000] | 0.000 | 9.1 | 0.039 | 0.985±0.003 | 0.685±0.046 |
| 1 | 1.000 | [1.000, 1.000] | 0.000 | 9.1 | 0.059 | 0.984±0.003 | 0.684±0.046 |

## Sample Sizes
- Total genuine pairs: 144
- Total impostor pairs: 552
- Balance ratio: 0.261