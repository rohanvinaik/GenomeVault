# HDC Fingerprint Validation Report
**Protocol**: Subject Disjoint
**Generated**: 2025-08-24T23:17:56.861058

## Protocol Configuration
- **Split Strategy**: subject_disjoint
- **Cross-validation Folds**: 5
- **Random Seed**: 42
- **Cluster Bootstrap**: subject
- **Normalization**: train-only

## Cohort Information
- **Subjects**: 282
- **Families**: 56
- **Batches**: 20
- **Samples per Subject**: 5-5-5 (min-median-max)

## Test Pairs
- **Genuine Pairs**: 25,000
- **Impostor Pairs**: 200,000
- **Total Test Pairs**: 225,000
- **Balance Ratio**: 0.125
- **Subsampling**: None

## Aggregate Performance
- **AUC**: 1.000 [1.000, 1.000]
- **EER**: 0.000 (95% upper bound: 0.000)
- **d-prime**: 38.01
- **Score Margin**: 0.107
- **Genuine**: μ=0.976, σ=0.005
- **Impostor**: μ=0.516, σ=0.024

## Operating Points
| Operating Point | FAR | FRR |
|-----------------|-----|-----|
| 0.1% FRR | 0.0000 | 0.001 |
| 1% FRR | 0.0000 | 0.01 |
| 5% FRR | 0.0000 | 0.05 |
| 0.1% FAR | 0.001 | 1.0000 |
| 1% FAR | 0.01 | 1.0000 |
| 5% FAR | 0.05 | 1.0000 |

## Validation Checks ✓
- **Label Shuffle AUC**: 0.521 (should be ≈ 0.5)
- **Label Shuffle EER**: 0.479 (should be ≈ 0.5)
- **Duplicate Rate**: 0.000 (should be ≈ 0)
- **Validation Status**: ✅

## Per-Fold Results
| Fold | AUC | CI | EER | d' | Margin | μ_gen±σ | μ_imp±σ | N_pairs |
|------|-----|----|----|----|---------|---------|---------|---------| 
| 0 | 1.000 | [1.000, 1.000] | 0.000 | 38.0 | 0.106 | 0.977±0.006 | 0.510±0.024 | 5000+40000 |
| 1 | 1.000 | [1.000, 1.000] | 0.000 | 43.0 | 0.107 | 0.975±0.004 | 0.514±0.026 | 5000+40000 |
| 2 | 1.000 | [1.000, 1.000] | 0.000 | 28.8 | 0.075 | 0.975±0.005 | 0.519±0.026 | 5000+40000 |
| 3 | 1.000 | [1.000, 1.000] | 0.000 | 33.4 | 0.133 | 0.977±0.005 | 0.518±0.024 | 5000+40000 |
| 4 | 1.000 | [1.000, 1.000] | 0.000 | 43.8 | 0.148 | 0.975±0.005 | 0.520±0.022 | 5000+40000 |

## Artifacts
- **ROC Curves**: roc_curves.png
- **DET Curves**: det_curves.png
- **Score Distributions**: score_distributions.png

## Provenance
- **Timestamp**: 2025-08-24T23:17:56.861058
- **Dataset SHA256**: `1269cbf6a5ce66b82579d9a340f73efdc459145ad93b0a7b24aecc408c4a6da2`
- **Code Git SHA**: `894f3047b5150bfd00fe06cda4f0728b06857c5e`
- **Python Version**: 3.11.8 | packaged by conda-forge | (main, Feb 16 2024, 20:49:36) [Clang 16.0.6 ]

## PIR Performance Context
- **Scheme**: IT-PIR (Information-Theoretic)
- **Database Sizes**: 100,000, 1,000,000 rows
- **Response Size**: 1024 bytes

| Topology | Servers | P50 Latency (ms) | Client CPU (%) | Server CPU (%) | Overhead (KB) |
|----------|---------|------------------|----------------|----------------|---------------|
| Single Server | 1 | 592.8 | 62.3 | 53.3 | 1.1 |
| Multi Server 3 | 3 | 6352.1 | 260.0 | 294.0 | 538.1 |

## ZK Proof Backend Performance
- **Hardware**: Apple M1 Max (10 cores, 64GB RAM)
- **Constraints**: 15,234

| Backend | Proof Size (bytes) | Prove P50 (ms) | Prove P99 (ms) | Verify P50 (ms) | Verify P99 (ms) |
|---------|-------------------|----------------|----------------|-----------------|-----------------|
| Groth16 | 192 | 1148.3 | 1729.3 | 4.00 | 5.81 |
| PLONK | 1024 | 817.5 | 898.2 | 14.50 | 16.02 |
| Halo2 | 5120 | 602.6 | 710.8 | 20.36 | 23.17 |

## Signature Verification
```bash
# Verify bundle integrity
openssl dgst -sha256 -verify docs/keys/benchmark_pubkey.pem -signature bundle_subject_disjoint.tar.gz.sig bundle_subject_disjoint.tar.gz
```
