# CLAUDE.md

Quick reference for Claude Code when working with the GenomeVault codebase.

**Note**: The project now includes a comprehensive **Probabilistic Alignment & Privacy Stack** that extends beyond the original Byzantine Consensus approach. See `docs/guides/PROBABILISTIC_ALIGNMENT_PRIVACY_STACK.md` for details.

## Project Overview

GenomeVault: Privacy-preserving genomic computing platform using hyperdimensional computing (HDC), zero-knowledge proofs, and private information retrieval. Achieves ~1,500× compression from raw FASTQ data (100-150 GB → 78 MB) or 38.4× from VCF variants (3 GB → 78 MB), with 264× architectural efficiency and mathematical privacy guarantees.

## 🚀 Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run main pipeline (RECOMMENDED) ⚡
python benchmarks/run_alignment_optimized_pipeline.py --preset production

# Run COMPLETE 4-layer privacy pipeline (with real FASTQ alignment) 🔒
python benchmarks/run_complete_privacy_pipeline.py \
    --reference-pool-fastq ref1_R1.fq ref1_R2.fq ref2_R1.fq ref2_R2.fq ref3_R1.fq ref3_R2.fq \
    --query-fastq query_R1.fq query_R2.fq \
    --output results/ \
    --skip-consensus  # Use existing consensus

# Quick test
python benchmarks/run_alignment_optimized_pipeline.py --preset production --quick

# Start REST API server
uvicorn genomevault.api.app:app --reload --port 8000
# Access API docs: http://localhost:8000/api/docs
# See GETTING_STARTED_API.md for API usage guide

# Clinical SNP Database (Query clinically-relevant variants)
# 1. Build clinical database from ClinVar
python -m genomevault.clinical_db.data_acquisition \
    --genome-build GRCh38 \
    --output-dir data \
    --pathogenic-only \
    --min-stars 1

# 2. Query clinical variants
python -m genomevault.cli.clinical_query_cli query-position --chr chr11 --pos 5227002
python -m genomevault.cli.clinical_query_cli query-gene BRCA1
python -m genomevault.cli.clinical_query_cli stats
```

## 📂 Project Structure

```
genomevault/
├── genomevault/
│   ├── differential_encoding/     # 11× compression, k-anonymity
│   │   └── align_to_reference_pool.py  # 🔒 Privacy-preserving query alignment
│   ├── reference/                 # 🆕 Probabilistic alignment system
│   │   ├── byzantine_consensus_builder.py       # Layer 1: Consensus reference
│   │   ├── probabilistic_alignment_system.py    # Hierarchical SNP classification
│   │   ├── advanced_indel_detection.py          # Smith-Waterman realignment
│   │   └── comprehensive_alignment_engine.py    # 7 challenge categories
│   ├── clinical_db/               # 🩺 Clinical SNP database (ClinVar)
│   │   ├── database.py            # Query clinical variants
│   │   └── data_acquisition.py    # Download and build database
│   ├── hypervector_transform/     # 24× HDC projection
│   ├── zk_proofs/                 # Zero-knowledge circuits (Halo2/Groth16)
│   ├── pir/                       # Private information retrieval
│   ├── blockchain/                # Attestation registry (opt-in)
│   ├── compute/                   # Hardware abstraction (CPU/Metal/CUDA)
│   ├── cli/                       # Command-line tools
│   │   └── clinical_query_cli.py  # Clinical variant queries
│   └── api/                       # REST API endpoints
│       └── routers/clinical_query.py  # Clinical variant API
├── benchmarks/
│   ├── run_complete_privacy_pipeline.py  # 🔒 COMPLETE 4-LAYER PIPELINE
│   ├── run_alignment_optimized_pipeline.py  # ⚡ MAIN BENCHMARK (with QC)
│   ├── run_probabilistic_alignment_pipeline.py  # Probabilistic analysis
│   └── run_full_pipeline_with_reference_pool.py
├── tests/                         # Comprehensive test suite
└── docs/
    └── guides/
        ├── PROBABILISTIC_ALIGNMENT_COMPLETE_GUIDE.md  # 🔒 COMPLETE GUIDE
        ├── PROBABILISTIC_ALIGNMENT_SECURITY_MODEL.md  # Security analysis
        └── PROBABILISTIC_ALIGNMENT_PIPELINE_GUIDE.md  # Usage guide
```

## 🎯 Running the Main Pipeline

### **Alignment-Optimized Pipeline** (RECOMMENDED)

The production-ready pipeline with 5.92× speedup through alignment optimizations.

**Quick Run (2-3 seconds):**
```bash
python benchmarks/run_alignment_optimized_pipeline.py --preset production
```

**With Comparison to Baseline:**
```bash
python benchmarks/run_alignment_optimized_pipeline.py --preset production --compare
```

**What it does:**
- Differential encoding with reference pool (k=3 anonymity)
- HDC integration (10,000D hypervector)
- ZK proof generation (Groth16, 743 bytes)
- PIR query (IT-PIR, 0.25% breach probability)
- Complete privacy-preserving pipeline in ~2 seconds

**Output Location:**
```
benchmark_results/full_pipeline_results/pipeline_run_YYYYMMDD_HHMMSS/
├── pipeline_results.json       # Main metrics
├── encoding_result.json        # Differential encoding
├── zk_proof.json              # Zero-knowledge proof
└── pir_query_result.json      # PIR query result
```

### **Complete 4-Layer Privacy Pipeline** 🔒 (NEW - with Real FASTQ Alignment)

**CRITICAL SECURITY**: Query NEVER aligns directly to consensus - uses privacy-preserving handoff through reference pool.

**Full Pipeline (~1-1.5 hours for chr22):**
```bash
python benchmarks/run_complete_privacy_pipeline.py \
    --reference-pool-fastq \
        data/downloaded/fastq/ERR3239276_1.fastq.gz data/downloaded/fastq/ERR3239276_2.fastq.gz \
        data/downloaded/fastq/ERR3239454_1.fastq.gz data/downloaded/fastq/ERR3239454_2.fastq.gz \
        data/downloaded/fastq/ERR3239475_1.fastq.gz data/downloaded/fastq/ERR3239475_2.fastq.gz \
    --query-fastq \
        data/downloaded/fastq/ERR3239334_1.fastq.gz \
        data/downloaded/fastq/ERR3239334_2.fastq.gz \
    --output benchmark_results/complete_privacy_pipeline \
    --chromosome chr22 \
    --skip-consensus  # Use existing consensus
```

**What it does (4 Layers):**
1. **Layer 1**: Byzantine Consensus Reference (hg38 + hg19 + chm13 → consensus with positional uncertainty)
2. **Layer 2**: Reference Pool Assembly (3 FASTQ → align to consensus → ref1.vcf, ref2.vcf, ref3.vcf)
3. **Layer 3**: Privacy-Preserving Query Alignment (Query FASTQ → align to **REFERENCE POOL** → query.vcf)
   - **CRITICAL**: Query → Pool → Consensus (NO DIRECT CONSENSUS LINK!)
4. **Layer 4**: GenomeVault Core (Differential + HDC + ZK + PIR)

**Privacy Guarantees:**
- ✅ No direct consensus alignment (untraceable to public references)
- ✅ k=3 anonymity (query hidden among pool)
- ✅ Positional uncertainty (~128-bit entropy)
- ✅ User-specific randomization (SHA-256 security)

**Output Location:**
```
benchmark_results/complete_privacy_pipeline/
├── consensus/consensus.fa              # Layer 1: Byzantine consensus
├── reference_pool/
│   ├── ref1.vcf.gz                     # Layer 2: Pool member 1
│   ├── ref2.vcf.gz                     # Layer 2: Pool member 2
│   └── ref3.vcf.gz                     # Layer 2: Pool member 3
├── query/query.vcf.gz                  # Layer 3: Privacy-preserving query
├── genomevault_core/                   # Layer 4: GenomeVault results
└── pipeline_summary.json               # Complete summary
```

**Documentation:**
- Complete Guide: `docs/guides/PROBABILISTIC_ALIGNMENT_COMPLETE_GUIDE.md`
- Security Model: `docs/guides/PROBABILISTIC_ALIGNMENT_SECURITY_MODEL.md`
- Usage Guide: `docs/guides/PROBABILISTIC_ALIGNMENT_PIPELINE_GUIDE.md`

### Standard Pipeline (Baseline)

**Quick Test (10-15 seconds):**
```bash
python benchmarks/run_full_pipeline_with_reference_pool.py --quick
```

**Full Run:**
```bash
python benchmarks/run_full_pipeline_with_reference_pool.py
```

**Format-Specific Runs:**
```bash
# FASTQ input (requires minimap2, samtools, bcftools)
python benchmarks/run_full_pipeline_with_reference_pool.py --format fastq

# VCF input
python benchmarks/run_full_pipeline_with_reference_pool.py --format vcf

# All formats
python benchmarks/run_full_pipeline_with_reference_pool.py --format all
```

### Input Data Requirements

The pipelines expect data in these locations:

**Reference Pool:**
```
benchmark_results/differential_encoding_samples/
├── reference_pool_1/    # k=3 reference genome 1
├── reference_pool_2/    # k=3 reference genome 2
└── reference_pool_3/    # k=3 reference genome 3
```

**Reference Genome:**
```
benchmark_results/full_pipeline_synthetic/reference/chr22.fa
```

**Generate Synthetic Data (if needed):**
```bash
./benchmarks/full_pipeline_synthetic_data.sh
# Takes 30-40 min for chr22 with 30x coverage
```

## 📊 Expected Performance

| Stage | Duration | Details |
|-------|----------|---------|
| **Differential Encoding** | 1.36s | 120 variants, k=3 anonymity, 292 differences |
| **HDC Integration** | 0.5ms | 264× architectural compression |
| **ZK Proof (Groth16)** | 0.74s | 743 bytes, 117,143 constraints |
| **PIR Query (IT-PIR)** | 4.33ms | 0.25% breach probability |
| **⚡ TOTAL** | **2.11s** | **5.92× speedup vs baseline** |

**Key Improvements:**
- Baseline: 12.47s → Optimized: 2.11s
- 83.1% reduction in total time
- 100% security preservation

## 🔧 Essential Commands

```bash
# Development
pytest                                    # Run all tests
pytest tests/test_compute_backend.py      # Test hardware backends
python benchmarks/compression_summary.py  # Verify compression

# Main pipeline (pick one)
python benchmarks/run_alignment_optimized_pipeline.py --preset production  # ⚡ RECOMMENDED
python benchmarks/run_full_pipeline_with_reference_pool.py --quick         # Quick test

# Differential encoding benchmark
python benchmarks/differential_encoding/benchmark_end_to_end.py --quick

# ZK proofs
./benchmarks/setup_groth16_enhanced.sh   # One-time setup
python benchmarks/zk_groth16_benchmark.py

# Blockchain integration tests
pytest tests/test_blockchain_integration.py -v  # 40 tests, <2ms overhead
```

## 🗺️ Navigation Guide

### Finding Components

| What | Where | Key Files |
|------|-------|-----------|
| **Benchmarks** | `/benchmarks/` | `run_alignment_optimized_pipeline.py` ⚡ |
| **Latest Results** | `/benchmark_results/` | `pipeline_results.json`, `latest_results.json` |
| **ZK Circuits** | `/genomevault/zk_proofs/circuits/` | `variant_presence_enhanced.circom` |
| **HDC Encoding** | `/genomevault/hypervector_transform/` | `unified_encoder.py`, `backend_adapter.py` |
| **Differential Encoding** | `/genomevault/differential_encoding/` | `enhanced_pipeline.py` |
| **Alignment System** | `/genomevault/differential_encoding/` | `optimized_sequence_alignment.py` (920 lines) |
| **Probabilistic Alignment** | `/genomevault/reference/` | `probabilistic_alignment_system.py` (new!) |
| **Byzantine Consensus** | `/genomevault/reference/` | `byzantine_consensus_builder.py` (updated!) |
| **Hardware Backends** | `/genomevault/compute/` | `backend.py` (CPU/Metal/CUDA) |
| **Blockchain** | `/genomevault/blockchain/` | `attestation_registry.py` |
| **Tests** | `/tests/` | Organized by component |
| **Config** | `/genomevault/config/` | `compute.yaml`, `blockchain.yaml` |

### Search Patterns

```bash
# Find all benchmarks
find benchmarks/ -name "*benchmark*.py"

# Find results
find benchmark_results/ -name "*results*.json"

# Find ZK circuits
find genomevault/zk_proofs/circuits/ -name "*.circom"

# Find test files
find tests/ -name "test_*.py"
```

## ⚙️ Configuration

### Hardware Backend

Edit `genomevault/config/compute.yaml` or use environment variables:

```bash
# Auto-detect (Metal > CUDA > CPU)
export GENOMEVAULT_BACKEND=auto

# Force specific backend
export GENOMEVAULT_BACKEND=cpu     # CPU-only (default)
export GENOMEVAULT_BACKEND=metal   # Apple Silicon
export GENOMEVAULT_BACKEND=cuda    # NVIDIA GPU
```

### Blockchain Integration

Edit `genomevault/config/blockchain.yaml`:

```yaml
blockchain:
  enabled: false              # Disabled by default
  network: "polygon-mumbai"   # or "ethereum-mainnet", "polygon"
  attestation:
    batch_mode: true          # Gas optimization
    batch_size: 10
```

**See:** `docs/reports/BLOCKCHAIN_INTEGRATION_COMPLETE.md` for full guide

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: chr22.fa` | Run `./benchmarks/full_pipeline_synthetic_data.sh` |
| Import errors | Check `genomevault/core/exceptions.py` |
| Slow HDC batch | Enable GPU: `GENOMEVAULT_BACKEND=auto` |
| GPU not detected | Run `python tests/test_compute_backend.py` |
| ZK setup fails | Run `./benchmarks/setup_groth16_enhanced.sh` |
| PIR authentication error | Fixed in latest code (client key shared with server) |
| API "Reference manager has no reference genomes" | Run `python scripts/genomevault_setup_references.py --use-case development` and ensure VCF files are in `vcf_pool/` directory |

### Validating Results

```bash
# View pipeline results
cat benchmark_results/full_pipeline_results/pipeline_run_*/pipeline_results.json

# Check success rate (should be 100.0)
jq '.summary.success_rate' pipeline_results.json

# Verify ZK proof (should be "valid")
jq '.stages[] | select(.stage=="ZK Proof Generation") | .metrics.verification_status' pipeline_results.json

# Check PIR security (should be true)
jq '.stages[] | select(.stage=="PIR Query") | .metrics.information_theoretic_security' pipeline_results.json
```

## 📈 Current Status

**Branch:** `main`  
**Status:** 🟢 **PRODUCTION READY**

**Latest Benchmarks (October 2025):**
- ✅ Complete Pipeline: 2.11s (5.92× speedup)
- ✅ Differential Encoding: 1.36s (120 variants, k=3)
- ✅ Architectural Compression: 264× (11× diff × 24× HDC)
- ✅ Empirical Space Savings: 38.4× (1,500 KB → 39.06 KB)
- ✅ ZK Proofs: 0.74s (Groth16, 743 bytes)
- ✅ PIR Queries: 4.33ms (IT-PIR)
- ✅ Blockchain: 40/40 tests passing, <2ms overhead
- ✅ REST API: 24/24 system checks passed, 2.84s average processing

**Key Features:**
- Alignment System: Minimizers, Bloom filters, parallel scoring, LRU caching
- Hardware Acceleration: Metal/CUDA backends for batch operations
- ZK Circuit: 117,143 constraints (Groth16, production-ready)
- Blockchain: Phase 1 + Phase 2 complete (attestation + institutional onboarding)
- Security: 100% preserved (SHA-256 for crypto, xxhash for performance only)
- REST API: Production-ready with comprehensive validation

## 📚 Documentation

### Core Docs
- **Probabilistic Alignment & Privacy Stack:** `docs/guides/PROBABILISTIC_ALIGNMENT_PRIVACY_STACK.md` (comprehensive guide) **⭐ NEW**
- **Complete Results:** `docs/reports/COMPLETE_BENCHMARK_RESULTS.md`
- **Alignment Optimization:** `docs/reports/ALIGNMENT_OPTIMIZATION_RESULTS_SUMMARY.md`
- **Blockchain Integration:** `docs/reports/BLOCKCHAIN_INTEGRATION_COMPLETE.md`
- **REST API Guide:** `docs/API_USAGE_GUIDE.md` (550+ lines)
- **API Getting Started:** `docs/api-docs/GETTING_STARTED_API.md` (step-by-step for end users)
- **API Implementation:** `docs/api-docs/ANALYSIS_API_IMPLEMENTATION_SUMMARY.md`
- **System Test Report:** `docs/reports/SYSTEM_TEST_REPORT.md` (comprehensive 7-phase validation)
- **Security Analysis:** `docs/guides/HYPERVECTOR_SECURITY.md`
- **ZK Production:** `docs/guides/ZK_PRODUCTION_GUIDE.md`

### Detailed Guides
- **Alignment System:** `docs/guides/alignment_system_improvements.md`
- **Backend Migration:** `docs/backend_migration_guide.md`
- **Differential Encoding:** `docs/differential_encoding_guide.md`
- **Implementation:** `docs/IMPLEMENTATION_GUIDE_COMPLETE.md`

### Examples
- **Alignment Example:** `examples/alignment_example.py`
- **Differential Encoding:** `examples/differential_encoding_basic.py`
- **Complete Pipeline:** `examples/complete_pipeline_demo.py`

### Academic
- **Paper (31 pages):** `docs/GenomeVault_Paper_Current/GenomeVault_Academic_Paper.pdf`
- **Paper Source:** `docs/GenomeVault_Paper_Current/GenomeVault_Academic_Paper.tex`

## 🔗 Key Features Summary

### Multi-Format Input Support
- **FASTQ** (raw sequencing): Auto-alignment with minimap2/BWA
- **VCF** (variants): Direct differential encoding
- **BAM/SAM** (aligned): Automatic variant calling

**Dependencies:**
```bash
conda install -c bioconda minimap2 samtools bcftools
```

### Blockchain Integration (Optional)
- **Phase 1:** Attestation registry with <1ms overhead
- **Phase 2:** HIPAA compliance with NPI verification
- **Default:** Disabled (opt-in via config)

### Hardware Acceleration
- **CPU:** NumPy + FAISS (always available)
- **Metal:** MLX for Apple Silicon (14.8× speedup)
- **CUDA:** PyTorch for NVIDIA (10-50× speedup on batch)

## 🎓 Quick Tips

1. **Always use the alignment-optimized pipeline** for production workloads
2. **Enable GPU** only for batch operations (>100 samples)
3. **ZK proofs are CPU-bound** - GPU doesn't help
4. **Reference pool must have k genomes** for k-anonymity
5. **Blockchain is opt-in** - disabled by default for performance
6. **REST API requires reference pool setup** - run setup script before first use

## 🆘 Getting Help

- **Issues:** Check `TROUBLESHOOTING.md` or GitHub Issues
- **Performance:** See `docs/reports/OPTIMIZATION_RESULTS_SUMMARY.md`
- **Security:** Review `docs/guides/HYPERVECTOR_SECURITY.md`
- **Blockchain:** Read `docs/reports/BLOCKCHAIN_INTEGRATION_COMPLETE.md`
- **Academic Details:** See paper in `docs/GenomeVault_Paper_Current/`
- **API Setup:** See `docs/api-docs/GETTING_STARTED_API.md` for step-by-step guide

---

**Last Updated:** October 2025  
**Version:** 1.0.0 (Production Ready)
