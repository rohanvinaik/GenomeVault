# 🔬 ZK Circuit Benchmark Report

**Date**: 2025-08-24 10:13:28
**Backend**: Mock (Development)

## 📊 Summary

Performance metrics for GenomeVault's zero-knowledge proof circuits.

## 📈 Detailed Results

| Circuit | Constraints | Input Size | Witness (ms) | Proof (ms) | Verify (ms) | Proof Size | CPU (%) | RAM (MB) |
|---------|------------|------------|--------------|------------|-------------|------------|---------|----------|
| variant_presence | 5,000 | 1 | 0.7 | 0.7 | 1.0 | 256 B | 0.1 | 0.0 |
| variant_presence | 5,000 | 10 | 3.0 | 3.0 | 1.0 | 257 B | 0.0 | 0.1 |
| variant_presence | 5,000 | 100 | 0.5 | 0.5 | 1.0 | 255 B | 0.0 | 0.0 |
| polygenic_risk_score | 20,000 | 10 | 0.7 | 0.7 | N/A | N/A | 0.0 | 0.0 |
| polygenic_risk_score | 20,000 | 50 | 2.1 | 2.1 | N/A | N/A | 0.0 | 0.0 |
| polygenic_risk_score | 20,000 | 200 | 1.0 | 1.0 | N/A | N/A | 0.0 | 0.0 |
| pharmacogenomic | 10,000 | 5 | 0.4 | 0.4 | 1.0 | 377 B | 0.0 | 0.0 |
| pharmacogenomic | 10,000 | 20 | 0.6 | 0.6 | 1.0 | 377 B | 0.1 | 0.0 |
| pharmacogenomic | 10,000 | 50 | 0.6 | 0.6 | 1.0 | 377 B | 0.0 | 0.0 |
| diabetes_risk_alert | 15,000 | 1 | 3.0 | 3.0 | 1.0 | 485 B | 0.0 | 0.0 |
| diabetes_risk_alert | 15,000 | 5 | 0.7 | 0.7 | 1.0 | 483 B | 0.0 | 0.0 |
| diabetes_risk_alert | 15,000 | 10 | 3.1 | 3.1 | 1.0 | 484 B | 0.0 | 0.1 |
| ancestry_composition | 15,000 | 10 | 1.3 | 1.3 | 1.0 | 427 B | 0.0 | 0.0 |
| ancestry_composition | 15,000 | 50 | 0.5 | 0.5 | 1.0 | 427 B | 0.0 | 0.0 |
| ancestry_composition | 15,000 | 100 | 0.6 | 0.6 | 1.0 | 427 B | 0.0 | 0.0 |

## 🎯 Performance Insights

- **Average Witness Generation**: 1.3ms
- **Maximum Witness Generation**: 3.1ms
- **Average Memory Usage**: 0.0MB

## 📐 Scaling Analysis

- **variant_presence**: sub-linear ✅ (100x size → 0.7x time)
- **polygenic_risk_score**: sub-linear ✅ (20x size → 1.4x time)
- **pharmacogenomic**: sub-linear ✅ (10x size → 1.4x time)
- **diabetes_risk_alert**: sub-linear ✅ (10x size → 1.0x time)
- **ancestry_composition**: sub-linear ✅ (10x size → 0.4x time)

## 💻 Hardware Information

- **CPU**: 10 cores (10 threads)
- **RAM**: 64.0 GB
- **Platform**: darwin

## 💡 Recommendations

⚠️ **Using mock backend** - Install Circom for real measurements:
```bash
npm install -g circom snarkjs
```

### Optimization Opportunities

Slowest operations:
- diabetes_risk_alert (size 10): 3.1ms
- variant_presence (size 10): 3.0ms
- diabetes_risk_alert (size 1): 3.0ms