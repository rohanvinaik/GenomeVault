# GenomeVault Benchmark Report

**Date**: 2025-08-24T22:14:34.705725
**Platform**: macOS-26.0-arm64-arm-64bit
**Python**: 3.11.8
**Seed**: 42
**Git Commit**: 2b0d83c9

## Results

| Benchmark | Category | Duration (ms) | Input | Output | Ratio | Checksum |
|-----------|----------|---------------|-------|--------|-------|----------|
| benchmark_hdc_compression | error | 0.00 | 0 | 0 | 0.0× |  |
| zk_variant_presence | zk_proof | 0.05 | 439 | 32 | 13.7× | e047d360 |
| pir_query_100 | pir | 0.00 | 4190 | 42 | 99.8× | bc0130c9 |

## Summary

- Total benchmarks: 3
- Successful: 2
- Failed: 1
- Total duration: 0.05ms
- Run time: 3.18s

## Verification

To verify these results:
```bash
PYTHONHASHSEED=42 python benchmarks/run.py
```

The checksums should match exactly if run on the same platform.