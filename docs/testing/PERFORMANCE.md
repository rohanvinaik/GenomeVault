# GenomeVault Performance Documentation

## Compression Performance

### Methodology
We use a multi-tier compression system optimized for different use cases:

1. **Tier 1 (Mini)**: Top 100 most significant variants
   - Input: ~400 KB (VCF format)
   - Output: ~25 KB (HDC encoded)
   - Ratio: ~16×
   - Use: Quick clinical summaries

2. **Tier 2 (Clinical)**: Top 1,000 clinically relevant variants
   - Input: ~4 MB (VCF format)
   - Output: ~300 KB (HDC encoded)
   - Ratio: ~13×
   - Use: Full clinical reports

3. **Tier 3 (Research)**: Up to 10,000 variants
   - Input: ~40 MB (VCF format)
   - Output: 100-200 KB (HDC encoded)
   - Ratio: 200-400×
   - Use: Research datasets

### Why Not "Genome in a Tweet"?

The "genome in a tweet" claim (~280 bytes) would require:
- **Extreme lossy compression**: Loss of clinically relevant information
- **Reference-based encoding**: Requires external reference genome
- **Variant filtering**: Only storing differences, not full genome

While technically possible, this level of compression is not suitable for:
- Clinical diagnosis
- Regulatory compliance
- Research reproducibility

### Realistic Targets

For clinical use, we target:
- **Lossless** compression of significant variants
- **Cryptographic** verifiability
- **Privacy-preserving** encoding
- **Reasonable** size (100-500 KB for clinical use)

## Zero-Knowledge Proof Performance

### Current Performance
| Circuit | Constraints | Witness (ms) | Proof (s) | Verify (ms) |
|---------|------------|--------------|-----------|-------------|
| variant_presence | 1,000 | 1-3 | 0.5-1 | 10-20 |
| prs_calculation | 5,000 | 5-10 | 2-3 | 20-30 |
| ancestry | 10,000 | 10-20 | 5-7 | 30-50 |

### Optimization Roadmap
1. **Phase 1**: Witness caching (90% reduction for repeated queries)
2. **Phase 2**: GPU acceleration (10× speedup for large circuits)
3. **Phase 3**: Specialized circuits (2-5× improvement)

## PIR Query Performance

### Current Performance
- Database size: 1 GB reference genome
- Query latency: 100-200 ms
- Throughput: 10-50 queries/second
- Privacy: Information-theoretic security

### Scaling Targets
- Support for 10 TB databases
- Sub-100ms query latency
- 1000+ queries/second
- Multi-server redundancy

## Hardware Requirements

### Minimum (Development)
```yaml
cpu: 4 cores
ram: 8 GB
disk: 20 GB SSD
gpu: Not required
```

### Recommended (Production)
```yaml
cpu: 8+ cores (Intel/AMD x86_64)
ram: 16-32 GB
disk: 100 GB NVMe SSD
gpu: NVIDIA GPU with 8+ GB VRAM (optional)
network: 1 Gbps
```

### Cloud Deployment
- **AWS**: t3.xlarge or better
- **GCP**: n2-standard-4 or better
- **Azure**: Standard_D4s_v3 or better

## Benchmark Reproduction

To reproduce our benchmarks:

```bash
# Install dependencies
pip install -r requirements.txt

# Run deterministic benchmark
python benchmark_harness.py

# Run full benchmark suite
python -m genomevault.benchmarks.full_suite
```

Results are deterministic with fixed seeds (SEED=42).
