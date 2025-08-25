# GenomeVault Benchmark Report

**Date**: 2025-08-24T22:55:34.087443
**Platform**: macOS-26.0-arm64-arm-64bit
**Python**: 3.11.8
**Seed**: 42
**Git Commit**: 2b0d83c9

## Results

| Benchmark | Category | Duration (ms) | Input | Output | Ratio | Checksum |
|-----------|----------|---------------|-------|--------|-------|----------|
| hdc_compression_1k | compression | 0.08 | 53487 | 32 | 1671.5× | 37014ee7 |
| zk_variant_presence | zk_proof | 0.01 | 439 | 32 | 13.7× | e047d360 |
| pir_query_100 | pir | 0.64 | 4190 | 32 | 130.9× | 5a5e9201 |

## Summary

- Total benchmarks: 3
- Successful: 3
- Failed: 0
- Total duration: 0.73ms
- Run time: 3.26s

## Verification

To verify these results:
```bash
PYTHONHASHSEED=42 python benchmarks/run.py
```

The checksums should match exactly if run on the same platform.