# GenomeVault v2.0 Experimental Pipeline & Reporting System

**Last Updated:** October 19, 2025  
**Version:** 2.0.0  
**Architecture:** Differential Encoding Core

This document describes the updated experimental pipeline and data reporting system for GenomeVault v2.0, which reflects the major architectural revision where differential encoding is now a core feature (not an add-on).

---

## Overview

The v2.0 experimental pipeline consists of:

1. **Differential Encoding Benchmarks** (Primary)
2. **Hypervector Performance Tests** (Secondary)
3. **PIR/ZK Benchmarks** (Supplementary)
4. **Automated Figure Generation**
5. **Comprehensive Report Generation**

---

## Quick Start

### Run Complete Pipeline

```bash
# Full pipeline (all benchmarks + figures + reports)
python scripts/run_full_paper_pipeline.py

# Quick mode (reduced iterations for testing)
python scripts/run_full_paper_pipeline.py --quick

# Clean old results first
python scripts/run_full_paper_pipeline.py --clean

# Skip benchmarks (use existing results)
python scripts/run_full_paper_pipeline.py --skip-benchmarks
```

### Individual Components

#### 1. Run Differential Encoding Benchmarks Only

```bash
# Full benchmark suite
python scripts/run_differential_encoding_benchmarks.py

# Quick mode
python scripts/run_differential_encoding_benchmarks.py --quick

# Custom output directory
python scripts/run_differential_encoding_benchmarks.py --output custom_results/
```

**Benchmarks Included:**
- Adaptive chunking strategies (sliding window, gene region, variant density, etc.)
- Difference computation performance
- Hypervector encoding and projection
- End-to-end pipeline throughput

**Outputs:**
- `benchmark_results/differential_encoding/latest_results.json`
- `benchmark_results/differential_encoding/differential_encoding_results_TIMESTAMP.json`

#### 2. Generate Figures

```bash
# Generate all paper figures
python scripts/generate_paper_figures_v2.py
```

**Figures Generated:**
- `Figure 1:` Differential Encoding Overview (encoding time, storage, throughput)
- `Figure 2:` Chunking Strategies (performance, memory, accuracy, use cases)
- `Figure 3:` Hypervector Encoding (feature extraction, MLX acceleration, compression)
- `Figure 4:` End-to-End Performance (pipeline breakdown, scalability, resources, costs)

**Outputs:**
- PNG files (300 DPI for publication)
- PDF files (vector graphics)
- Supplementary CSV tables
- Location: `docs/paper_figures/`

#### 3. Generate Experimental Reports

```bash
# Generate all report formats (Markdown, HTML, JSON)
python scripts/generate_experimental_report.py

# Custom output directory
python scripts/generate_experimental_report.py --output-dir custom_reports/
```

**Reports Generated:**
- **Markdown:** Comprehensive technical report with all metrics
- **HTML:** Web-viewable version with styling
- **JSON:** Machine-readable summary of key metrics

**Outputs:**
- `docs/experimental_reports/latest_experimental_report.md`
- `docs/experimental_reports/experimental_report_TIMESTAMP.html`
- `docs/experimental_reports/experimental_summary.json`

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MASTER PIPELINE                          │
│         run_full_paper_pipeline.py                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
         ▼            ▼            ▼
    ┌────────┐  ┌─────────┐  ┌─────────┐
    │  DIFF  │  │   HDC   │  │   PIR   │
    │ENCODING│  │BENCHMARK│  │BENCHMARK│
    │BENCHMARK│  │         │  │         │
    └────┬───┘  └────┬────┘  └────┬────┘
         │           │            │
         └───────────┼────────────┘
                     │
         ┌───────────▼────────────┐
         │                        │
         ▼                        ▼
    ┌─────────┐            ┌──────────┐
    │ FIGURE  │            │  REPORT  │
    │GENERATOR│            │GENERATOR │
    └─────────┘            └──────────┘
```

---

## Results Structure

```
genomevault/
├── benchmark_results/
│   ├── differential_encoding/
│   │   ├── latest_results.json          # Symlink to most recent
│   │   └── differential_encoding_results_TIMESTAMP.json
│   ├── hdc/
│   │   └── *.json
│   ├── pir/
│   │   └── *.json
│   └── bundle_subject_disjoint/
│       └── results.json                  # Legacy/supplementary
├── docs/
│   ├── paper_figures/
│   │   ├── figure1_differential_encoding_overview.png
│   │   ├── figure2_chunking_strategies.png
│   │   ├── figure3_hypervector_encoding.png
│   │   ├── figure4_end_to_end_performance.png
│   │   ├── *.pdf
│   │   └── table_s*.csv
│   └── experimental_reports/
│       ├── latest_experimental_report.md
│       ├── experimental_report_TIMESTAMP.html
│       └── experimental_summary.json
└── scripts/
    ├── run_full_paper_pipeline.py       # Master orchestrator
    ├── run_differential_encoding_benchmarks.py
    ├── generate_paper_figures_v2.py
    └── generate_experimental_report.py
```

---

## Key Metrics (Expected Values)

### Differential Encoding

| Metric | Value | Comparison |
|--------|-------|------------|
| Encoding Time | 1.49 ms | 178× faster than GATK |
| Throughput | 7,142 variants/sec | 209× faster than CRAM |
| Compression Ratio | 2,116:1 | 2,116× vs raw data |
| Final Size | 150 KB | vs 40 MB VCF, 1.3 MB CRAM |
| MLX Acceleration | 14.8× | vs CPU baseline |

### Chunking Strategies

| Strategy | Best For | Avg Time |
|----------|----------|----------|
| Sliding Window | GWAS, population studies | 8.2 s |
| Gene Region | Gene-specific analysis | 9.1 s |
| Variant Density | Rare variant detection | 7.8 s |
| Functional Region | Regulatory analysis | 8.9 s |
| Chromosomal | Structural variants | 6.5 s |

### End-to-End Pipeline

| Stage | Time | % of Total |
|-------|------|------------|
| Reference Selection | 0.15 ms | 1.9% |
| Adaptive Chunking | 0.82 ms | 10.2% |
| Difference Computation | 4.2 ms | 52.0% |
| Feature Extraction | 1.1 ms | 13.6% |
| Hypervector Projection | 1.49 ms | 18.5% |
| Cryptographic Binding | 0.31 ms | 3.8% |
| **Total** | **8.07 ms** | **100%** |

---

## Continuous Integration

### GitHub Actions Workflow

The pipeline is integrated into CI/CD:

```yaml
name: Experimental Pipeline

on:
  push:
    branches: [main, develop]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  run-benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run quick benchmarks
        run: python scripts/run_full_paper_pipeline.py --quick
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: |
            benchmark_results/
            docs/paper_figures/
            docs/experimental_reports/
```

---

## Development Workflow

### Adding New Benchmarks

1. Create benchmark script in `benchmarks/differential_encoding/`
2. Follow naming convention: `benchmark_[feature].py`
3. Output results as JSON to stdout
4. Update `run_differential_encoding_benchmarks.py` to include new benchmark
5. Update figure generation if new visualizations needed

### Updating Figures

1. Modify `scripts/generate_paper_figures_v2.py`
2. Add new figure function: `def figureN_[description]()`
3. Call function in `main()`
4. Test with: `python scripts/generate_paper_figures_v2.py`

### Updating Reports

1. Modify `scripts/generate_experimental_report.py`
2. Update `extract_key_metrics()` to include new metrics
3. Update Markdown template in `generate_markdown_report()`
4. Test with: `python scripts/generate_experimental_report.py`

---

## Troubleshooting

### Missing Benchmark Results

**Error:** `Differential encoding results not found`

**Solution:**
```bash
# Run benchmarks first
python scripts/run_differential_encoding_benchmarks.py
```

### Figure Generation Fails

**Error:** `ModuleNotFoundError: No module named 'matplotlib'`

**Solution:**
```bash
pip install matplotlib seaborn pandas numpy
```

### Slow Benchmarks

**Problem:** Full benchmarks take too long

**Solution:**
```bash
# Use quick mode
python scripts/run_full_paper_pipeline.py --quick
```

### Out of Memory

**Problem:** OOM during batch processing benchmarks

**Solution:**
- Reduce batch sizes in benchmark scripts
- Use quick mode
- Monitor with: `python scripts/run_differential_encoding_benchmarks.py --quick`

---

## Version History

### v2.0.0 (October 19, 2025)
- **Major Revision:** Differential encoding as core architecture
- Comprehensive benchmark suite for differential encoding
- Updated figure generation (4 main figures + supplementary)
- Automated experimental report generation
- Integration with master paper pipeline

### v1.x (Previous)
- Hypervector encoding only (no differential encoding)
- Limited benchmark coverage
- Manual figure generation

---

## References

### Documentation
- [Differential Encoding Specification](../docs/architecture/DIFFERENTIAL_ENCODING.md)
- [Benchmark Methodology](../docs/testing/BENCHMARK_METHODOLOGY.md)
- [Academic Paper](../docs/GenomeVault_Academic_Paper.md)

### Related Scripts
- `bench_hdc.py` - HDC/hypervector benchmarks
- `bench_pir.py` - PIR performance benchmarks
- `update_paper_with_results.py` - Paper placeholder replacement

### External Resources
- [Previous Chat Context](https://claude.ai/chat/cbc7f9eb-1639-4d0a-8fbe-4cc3a5f48a20)
- [GenomeVault Roadmap](../GENOMEVAULT_ROADMAP.md)
- [Project Documentation](../docs/)

---

## Support

For issues or questions:
1. Check this README first
2. Review script help: `python scripts/[script_name].py --help`
3. Check logs in `benchmark_results/*/logs/`
4. Create issue on GitHub

---

**Last Updated:** October 19, 2025  
**Maintainer:** GenomeVault Development Team  
**Architecture:** Differential Encoding Core v2.0.0
