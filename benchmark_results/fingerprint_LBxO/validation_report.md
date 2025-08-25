# HDC Fingerprint Validation Report

Generated: 2025-08-24 22:05:52
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
- **d-prime**: 16.68 [16.64, 17.09]
- **Score Margin**: 0.074 [0.057, 0.086]

## Validation Checks ✓
- **Label Shuffle AUC**: 0.509 (should be ~0.5)
- **Duplicate Rate**: 0.000 (should be ~0)

## Per-Fold Results
| Fold | AUC | CI | EER | d' | Margin | Genuine μ±σ | Impostor μ±σ |
|------|-----|-------|-----|-----|---------|-------------|--------------|
| 0 | 1.000 | [1.000, 1.000] | 0.000 | 14.9 | 0.057 | 0.984±0.003 | 0.678±0.029 |
| 1 | 1.000 | [1.000, 1.000] | 0.000 | 16.7 | 0.086 | 0.984±0.003 | 0.678±0.026 |
| 2 | 1.000 | [1.000, 1.000] | 0.000 | 16.6 | 0.054 | 0.984±0.004 | 0.676±0.026 |
| 3 | 1.000 | [1.000, 1.000] | 0.000 | 17.1 | 0.074 | 0.984±0.003 | 0.676±0.025 |
| 4 | 1.000 | [1.000, 1.000] | 0.000 | 19.3 | 0.093 | 0.984±0.003 | 0.676±0.022 |

## Sample Sizes
- Total genuine pairs: 970
- Total impostor pairs: 903
- Balance ratio: 1.074