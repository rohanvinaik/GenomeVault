# HDC Fingerprint Quality Evaluation Report

Generated: 2025-08-24 20:55:29


## Executive Summary

- **Best Accuracy**: 4096D @ 70% sparsity
  - EER: 0.467, Storage: 4.8KB

- **Best Storage**: 4096D @ 70% sparsity
  - EER: 0.467, Storage: 4.8KB

- **Best Balanced**: 4096D @ 70% sparsity
  - EER: 0.467, Storage: 4.8KB


## Detailed Results

| Dimension | Sparsity | Storage (KB) | EER | FAR | FRR | AUC | 95% CI |
|-----------|----------|-------------|-----|-----|-----|-----|--------|
| 4096 | 40% | 8.0 | 0.488 | 0.489 | 0.487 | 0.534 | [0.481, 0.587] |
| 4096 | 50% | 8.0 | 0.495 | 0.496 | 0.493 | 0.500 | [0.457, 0.555] |
| 4096 | 60% | 6.4 | 0.495 | 0.496 | 0.493 | 0.495 | [0.447, 0.537] |
| 4096 | 70% | 4.8 | 0.467 | 0.468 | 0.467 | 0.522 | [0.467, 0.568] |
| 8192 | 40% | 15.8 | 0.526 | 0.526 | 0.527 | 0.457 | [0.418, 0.519] |
| 8192 | 50% | 16.0 | 0.473 | 0.473 | 0.473 | 0.565 | [0.524, 0.605] |
| 8192 | 60% | 12.8 | 0.530 | 0.533 | 0.527 | 0.460 | [0.419, 0.518] |
| 8192 | 70% | 9.6 | 0.487 | 0.487 | 0.487 | 0.539 | [0.504, 0.575] |
| 16384 | 40% | 32.1 | 0.489 | 0.485 | 0.493 | 0.497 | [0.447, 0.546] |
| 16384 | 50% | 31.9 | 0.496 | 0.492 | 0.500 | 0.496 | [0.457, 0.536] |
| 16384 | 60% | 25.6 | 0.516 | 0.518 | 0.513 | 0.488 | [0.450, 0.538] |
| 16384 | 70% | 19.2 | 0.471 | 0.462 | 0.480 | 0.517 | [0.470, 0.567] |

## Recommendations

- **Clinical High-Accuracy**: Use 16384D @ 50% sparsity (best accuracy vs storage)
- **Balanced Research**: Use 8192D @ 60% sparsity (good accuracy, moderate storage)
- **Resource-Constrained**: Use 4096D @ 70% sparsity (minimal storage, acceptable accuracy)

## Statistical Analysis

- Cohort Size: 50 subjects × 3 samples
- Genuine Comparisons: 150 pairs
- Impostor Comparisons: 1000 pairs
- Bootstrap Iterations: 100 (for CI calculation)