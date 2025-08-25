# HDC Fingerprint Validation Report

Generated: 2025-08-24 22:05:39
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
- **d-prime**: 50.55 [42.22, 53.53]
- **Score Margin**: 0.080 [0.079, 0.096]

## Validation Checks ✓
- **Label Shuffle AUC**: 0.570 (should be ~0.5)
- **Duplicate Rate**: 0.000 (should be ~0)

## Per-Fold Results
| Fold | AUC | CI | EER | d' | Margin | Genuine μ±σ | Impostor μ±σ |
|------|-----|-------|-----|-----|---------|-------------|--------------|
| 0 | 1.000 | [1.000, 1.000] | 0.000 | 50.5 | 0.079 | 0.986±0.002 | 0.900±0.001 |
| 1 | 1.000 | [1.000, 1.000] | 0.000 | 42.2 | 0.078 | 0.985±0.002 | 0.900±0.002 |
| 2 | 1.000 | [1.000, 1.000] | 0.000 | 53.5 | 0.080 | 0.986±0.002 | 0.900±0.001 |
| 3 | 1.000 | [1.000, 1.000] | 0.000 | 85.2 | 0.103 | 0.983±0.002 | 0.875±0.001 |
| 4 | 1.000 | [1.000, 1.000] | 0.000 | 36.7 | 0.096 | 0.984±0.003 | 0.875±0.003 |

## Sample Sizes
- Total genuine pairs: 200
- Total impostor pairs: 30
- Balance ratio: 6.667