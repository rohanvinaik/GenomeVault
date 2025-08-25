# HDC Fingerprint Validation Report

Generated: 2025-08-24 22:18:31
Protocol: LFamO
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
- **d-prime**: 54.47 [37.29, 65.63]
- **Score Margin**: 0.118 [0.117, 0.135]

## Validation Checks ✓
- **Label Shuffle AUC**: 0.497 (should be ~0.5)
- **Duplicate Rate**: 0.000 (should be ~0)

## Per-Fold Results
| Fold | AUC | CI | EER | d' | Margin | Genuine μ±σ | Impostor μ±σ |
|------|-----|-------|-----|-----|---------|-------------|--------------|
| 0 | 1.000 | [1.000, 1.000] | 0.000 | 54.5 | 0.117 | 0.977±0.003 | 0.851±0.001 |
| 1 | 1.000 | [1.000, 1.000] | 0.000 | 35.2 | 0.115 | 0.977±0.003 | 0.849±0.004 |
| 2 | 1.000 | [1.000, 1.000] | 0.000 | 65.6 | 0.118 | 0.978±0.002 | 0.854±0.001 |
| 3 | 1.000 | [1.000, 1.000] | 0.000 | 73.2 | 0.157 | 0.972±0.002 | 0.806±0.002 |
| 4 | 1.000 | [1.000, 1.000] | 0.000 | 37.3 | 0.135 | 0.973±0.005 | 0.819±0.004 |

## Sample Sizes
- Total genuine pairs: 200
- Total impostor pairs: 30
- Balance ratio: 6.667