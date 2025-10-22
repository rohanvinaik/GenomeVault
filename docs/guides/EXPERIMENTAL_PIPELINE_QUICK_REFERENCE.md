# GenomeVault v2.0 Experimental Pipeline - Quick Reference

**Version:** 2.0.0  
**Architecture:** Differential Encoding Core  
**Last Updated:** October 19, 2025

---

## 🚀 Common Commands

### Validate Setup
```bash
# Check if everything is configured correctly
python scripts/validate_experimental_pipeline.py

# Verbose mode (show detailed errors)
python scripts/validate_experimental_pipeline.py --verbose

# Quick check (skip execution tests)
python scripts/validate_experimental_pipeline.py --skip-tests
```

### Run Complete Pipeline
```bash
# Full pipeline (benchmarks + figures + reports + PDF)
python scripts/run_full_paper_pipeline.py

# Quick mode (fast benchmarks for testing)
python scripts/run_full_paper_pipeline.py --quick

# Clean old results first
python scripts/run_full_paper_pipeline.py --clean

# Skip benchmarks, just regenerate outputs
python scripts/run_full_paper_pipeline.py --skip-benchmarks
```

### Run Individual Components

#### Benchmarks Only
```bash
# Differential encoding benchmarks (primary)
python scripts/run_differential_encoding_benchmarks.py

# Quick benchmarks
python scripts/run_differential_encoding_benchmarks.py --quick

# HDC benchmarks (secondary)
python scripts/bench_hdc.py

# PIR benchmarks (supplementary)
python scripts/bench_pir.py
```

#### Figures Only
```bash
# Generate all paper figures
python scripts/generate_paper_figures_v2.py
```

#### Reports Only
```bash
# Generate comprehensive experimental report
python scripts/generate_experimental_report.py
```

---

## 📁 Output Locations

### Benchmark Results
```
benchmark_results/
├── differential_encoding/
│   └── latest_results.json              ← Primary results
├── hdc/
│   └── *.json                           ← HDC benchmarks
└── pir/
    └── *.json                           ← PIR benchmarks
```

### Figures
```
docs/paper_figures/
├── figure1_differential_encoding_overview.png
├── figure2_chunking_strategies.png
├── figure3_hypervector_encoding.png
├── figure4_end_to_end_performance.png
├── *.pdf                                ← Vector versions
└── table_s*.csv                         ← Supplementary tables
```

### Reports
```
docs/experimental_reports/
├── latest_experimental_report.md        ← Latest report
├── experimental_report_TIMESTAMP.html   ← Timestamped versions
└── experimental_summary.json            ← Machine-readable summary
```

---

## 🔄 Typical Workflows

### First Time Setup
```bash
# 1. Validate setup
python scripts/validate_experimental_pipeline.py

# 2. Run quick test
python scripts/run_full_paper_pipeline.py --quick --clean

# 3. Check outputs
ls -lh benchmark_results/differential_encoding/
ls -lh docs/paper_figures/
ls -lh docs/experimental_reports/
```

### Daily Development
```bash
# Quick benchmarks during development
python scripts/run_differential_encoding_benchmarks.py --quick

# Regenerate figures with latest data
python scripts/generate_paper_figures_v2.py

# Update report
python scripts/generate_experimental_report.py
```

### Pre-Submission
```bash
# Full benchmarks (takes longer but comprehensive)
python scripts/run_full_paper_pipeline.py --clean

# Review outputs
cat docs/experimental_reports/latest_experimental_report.md
open docs/paper_figures/  # View figures

# Generate PDF
# (Handled by run_full_paper_pipeline.py if pandoc installed)
```

### Continuous Integration
```bash
# Quick validation (suitable for CI)
python scripts/run_full_paper_pipeline.py --quick
```

---

## 📊 Expected Results

### Differential Encoding Benchmarks

#### Key Metrics
```json
{
  "encoding_time_ms": 1.49,
  "throughput_variants_per_sec": 7142,
  "compression_ratio": 2116,
  "final_size_kb": 150
}
```

#### Comparison
- **178× faster** than GATK
- **209× faster** than CRAM
- **2,116× compression** vs raw data

### Chunking Strategies

| Strategy | Time (s) | Memory (MB) | Best For |
|----------|----------|-------------|----------|
| Sliding Window | 8.2 | 180 | GWAS |
| Gene Region | 9.1 | 220 | Gene analysis |
| Variant Density | 7.8 | 165 | Rare variants |
| Functional Region | 8.9 | 210 | Regulatory |
| Chromosomal | 6.5 | 140 | Structural variants |

### End-to-End Pipeline

| Stage | Time (ms) | % |
|-------|-----------|---|
| Reference Selection | 0.15 | 1.9% |
| Chunking | 0.82 | 10.2% |
| Difference Computation | 4.2 | 52.0% |
| Feature Extraction | 1.1 | 13.6% |
| Hypervector Projection | 1.49 | 18.5% |
| Cryptographic Binding | 0.31 | 3.8% |
| **Total** | **8.07** | **100%** |

---

## 🐛 Troubleshooting

### Problem: Scripts not found
```bash
# Ensure you're in the genomevault root directory
cd /Users/rohanvinaik/genomevault
pwd  # Should show .../genomevault
```

### Problem: Missing dependencies
```bash
# Install required packages
pip install matplotlib seaborn pandas numpy

# Or use requirements file
pip install -r requirements.txt
```

### Problem: Benchmarks fail
```bash
# Check if benchmark scripts exist
ls -la benchmarks/differential_encoding/

# Try quick mode first
python scripts/run_differential_encoding_benchmarks.py --quick

# Check logs
cat benchmark_results/differential_encoding/latest_results.json
```

### Problem: Out of memory
```bash
# Use quick mode (reduced iterations)
python scripts/run_differential_encoding_benchmarks.py --quick

# Or adjust batch sizes in benchmark scripts
```

### Problem: Figures don't generate
```bash
# Check if results exist first
ls -la benchmark_results/differential_encoding/

# Try running benchmarks first
python scripts/run_differential_encoding_benchmarks.py --quick

# Then generate figures
python scripts/generate_paper_figures_v2.py
```

---

## 📝 File Checklist

Before running the pipeline, ensure these files exist:

### Required Scripts
- [x] `scripts/run_full_paper_pipeline.py`
- [x] `scripts/run_differential_encoding_benchmarks.py`
- [x] `scripts/generate_paper_figures_v2.py`
- [x] `scripts/generate_experimental_report.py`
- [x] `scripts/validate_experimental_pipeline.py`

### Required Benchmark Scripts
- [ ] `benchmarks/differential_encoding/benchmark_chunking.py`
- [ ] `benchmarks/differential_encoding/benchmark_difference_computation.py`
- [ ] `benchmarks/differential_encoding/benchmark_hypervector_encoding.py`
- [ ] `benchmarks/differential_encoding/benchmark_end_to_end.py`

### Optional Scripts
- [ ] `scripts/bench_hdc.py`
- [ ] `scripts/bench_pir.py`
- [ ] `scripts/generate_paper_pdf.py`

---

## 🔗 Related Documentation

- **Full Documentation:** [EXPERIMENTAL_PIPELINE_README.md](./EXPERIMENTAL_PIPELINE_README.md)
- **Architecture:** [DIFFERENTIAL_ENCODING.md](../docs/architecture/DIFFERENTIAL_ENCODING.md)
- **Benchmark Methodology:** [BENCHMARK_METHODOLOGY.md](../docs/testing/BENCHMARK_METHODOLOGY.md)
- **Previous Context:** [Chat Link](https://claude.ai/chat/cbc7f9eb-1639-4d0a-8fbe-4cc3a5f48a20)

---

## ⚡ Quick Start (TL;DR)

```bash
# 1. Validate
python scripts/validate_experimental_pipeline.py

# 2. Run quick test
python scripts/run_full_paper_pipeline.py --quick --clean

# 3. Check results
cat docs/experimental_reports/latest_experimental_report.md
open docs/paper_figures/

# 4. For full benchmarks (production)
python scripts/run_full_paper_pipeline.py --clean
```

---

## 📞 Getting Help

```bash
# Get help for any script
python scripts/[script_name].py --help

# Examples:
python scripts/run_full_paper_pipeline.py --help
python scripts/run_differential_encoding_benchmarks.py --help
python scripts/generate_paper_figures_v2.py --help
```

---

**Quick Reference Card v2.0.0**  
**Last Updated:** October 19, 2025  
**Architecture:** Differential Encoding Core
