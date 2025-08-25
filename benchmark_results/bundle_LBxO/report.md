# HDC Fingerprint Validation Report
**Protocol**: Lbxo
**Generated**: 2025-08-24T22:31:14.020401

## Protocol Configuration
- **Split Strategy**: LBxO
- **Cross-validation Folds**: 5
- **Random Seed**: 42
- **Cluster Bootstrap**: family
- **Normalization**: train-only

## Cohort Information
- **Subjects**: 200
- **Families**: 50
- **Batches**: 10
- **Samples per Subject**: 5-5-5 (min-median-max)

## Test Pairs
- **Genuine Pairs**: 970
- **Impostor Pairs**: 903
- **Total Test Pairs**: 1,873
- **Balance Ratio**: 1.074
- **Subsampling**: None

## Aggregate Performance
- **AUC**: 1.000 [1.000, 1.000]
- **EER**: 0.000 (95% upper bound: 0.010)
- **d-prime**: 16.74
- **Score Margin**: 0.103
- **Genuine**: μ=0.974, σ=0.005
- **Impostor**: μ=0.531, σ=0.037

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
- **Label Shuffle AUC**: 0.490 (should be ≈ 0.5)
- **Label Shuffle EER**: 0.490 (should be ≈ 0.5)
- **Duplicate Rate**: 0.000 (should be ≈ 0)
- **Validation Status**: ✅

## Per-Fold Results
| Fold | AUC | CI | EER | d' | Margin | μ_gen±σ | μ_imp±σ | N_pairs |
|------|-----|----|----|----|---------|---------|---------|---------| 
| 0 | 1.000 | [1.000, 1.000] | 0.000 | 15.1 | 0.092 | 0.974±0.005 | 0.533±0.041 | 230+253 |
| 1 | 1.000 | [1.000, 1.000] | 0.000 | 16.7 | 0.129 | 0.975±0.005 | 0.532±0.037 | 170+136 |
| 2 | 1.000 | [1.000, 1.000] | 0.000 | 16.5 | 0.076 | 0.974±0.006 | 0.528±0.038 | 190+171 |
| 3 | 1.000 | [1.000, 1.000] | 0.000 | 16.8 | 0.103 | 0.974±0.005 | 0.530±0.037 | 180+153 |
| 4 | 1.000 | [1.000, 1.000] | 0.000 | 19.3 | 0.133 | 0.975±0.004 | 0.530±0.032 | 200+190 |

## Artifacts
- **ROC Curves**: roc_curves.png
- **DET Curves**: det_curves.png
- **Score Distributions**: score_distributions.png

## Provenance
- **Timestamp**: 2025-08-24T22:31:14.020401
- **Dataset SHA256**: `1269cbf6a5ce66b82579d9a340f73efdc459145ad93b0a7b24aecc408c4a6da2`
- **Code Git SHA**: `cebf7d8a3a3ae971e0c9a320cae3cf1f237af45f`
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
openssl dgst -sha256 -verify docs/keys/benchmark_pubkey.pem -signature bundle_LBxO.tar.gz.sig bundle_LBxO.tar.gz
```
