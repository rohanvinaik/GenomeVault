# CLAUDE.md

Quick reference for Claude Code when working with the GenomeVault codebase.

---

# 🚨🚨🚨 CRITICAL WARNING - READ BEFORE ANY FILE OPERATIONS 🚨🚨🚨

## ⛔ NEVER DELETE ANYTHING IN benchmark_results/ WITHOUT EXPLICIT CONFIRMATION ⛔

**LAYER 2 GUIDE STRANDS ARE STORED IN benchmark_results/enhanced_privacy_k13_phase123_optimized/layer2_reference_pool/**

**These files represent ~350 GB and 4+ DAYS of processing:**
- **12 BAM files** (ref1-12.sorted.bam): 25-29 GB each
- **12 FASTA files** (ref1-12.fa.gz): ~830 MB each
- **IRREPLACEABLE - Takes 30+ hours to regenerate**

### 🔴 BEFORE deleting benchmark_results/ directories:
1. **STOP** - Check if it contains Layer 2 guide strands
2. **ASK USER** for explicit confirmation before any deletion
3. **VERIFY** files can be recovered from Time Machine
4. **NEVER assume** benchmark data is disposable

### ✅ Proper Layer 2 Storage Location (TODO - Migration needed):
- **Current (WRONG):** `benchmark_results/*/layer2_reference_pool/`
- **Future (CORRECT):** `data/guide_strands/` (permanent storage, excluded from benchmarks)

**Nov 6, 2025: Near-catastrophic deletion prevented only by Time Machine recovery**

---

## ⚠️ CRITICAL: Core Privacy Architecture (READ THIS FIRST)

**THIS IS THE FOUNDATION OF THE ENTIRE SYSTEM. VIOLATING THESE PRINCIPLES INVALIDATES 8 YEARS OF WORK.**

### The Iron Law of Privacy

**ANY CONTACT BETWEEN EXPERIMENTAL DATA AND PUBLIC REFERENCE/CONSENSUS DATA IS ASSUMED TO INVALIDATE ALL PRIVACY GUARANTEES.**

### 3-Layer Architecture

**Layer 1: Reference Superposition Consensus**
- Public genome data: hg38 + hg19 + chm13
- Byzantine consensus with positional uncertainty
- Creates `consensus.fa` (2.9 GB)
- **EXPERIMENTAL DATA MUST NEVER TOUCH THIS DIRECTLY**

**Layer 2: Guide Strands (Blind Middleman)**
- Real genomic samples: k=12 diverse whole-genome FASTQ samples
- **Pipeline workflow:**
  1. Guide FASTQ → align to consensus → Guide BAM files (ref1.sorted.bam, ref2.sorted.bam, etc.)
  2. Guide BAM → `samtools consensus` → Guide FASTA files (ref1.fa.gz, ref2.fa.gz, etc., ~828 MB each)
- **Guide FASTA files are the actual blind middleman** - rearranged genomic sequences, NOT variant calls
- Random cycling between guides per chunk = information-theoretic privacy

**Layer 3: Experimental Strand (Patient/Query Data)**
- Example: ERR3239334 FASTQ (23 GB)
- **CORRECT workflow:**
  1. Align experimental FASTQ to GUIDE FASTA sequences (NOT consensus!)
  2. Generate GDiff differential encoding (sequence-level differences)
  3. Create privacy-preserving hypervector with random guide cycling

### Terminology (Use Exactly)

- **Consensus**: Public genome superposition (hg38 + hg19 + chm13), Layer 1
- **Guide strands**: Real genomic samples (k=12) serving as blind middleman, Layer 2
  - Stored as guide FASTA files (ref1.fa.gz, ref2.fa.gz, etc.)
  - Rearranged genomic sequences aligned to consensus coordinates
- **Experimental strand**: Patient/query data being encoded, Layer 3
- **GDiff**: Differential encoding format capturing sequence-level differences
- **k-anonymity**: Number of guide strands (k=3 for dev, k=12 for production)

### Privacy Guarantee

```
Experimental strand → Guide strands → Consensus
```

The guide strands act as a cryptographic blind - experimental data never creates a traceable link to public references.

### What NOT To Do

❌ Align experimental FASTQ to consensus directly
❌ Use VCF format for differential encoding (outdated - use GDiff)
❌ Create any direct link between experimental and public data
❌ Skip the guide FASTA extraction step

### What To Do

✅ **Layer 2 (Guide Strand Creation):**
```bash
# For each guide sample (ref1-ref12)
minimap2 -ax sr -t 10 consensus.fa ${sample}_R1.fastq.gz ${sample}_R2.fastq.gz | \
    sambamba sort -t 10 -m 8G -o ${sample}.sorted.bam /dev/stdin

samtools consensus --threads 10 --show-del yes --show-ins yes \
    ${sample}.sorted.bam | pigz -p 8 > ${sample}.fa.gz
```

✅ **Layer 3 (Experimental Processing):**
```python
from genomevault.differential_encoding.align_to_reference_pool import PrivacyPreservingReferencePoolAligner
from genomevault.differential_encoding.gdiff import GDiffEncoder
from genomevault.hypervector_transform.gdiff_encoder import SelectiveHDVEncoder

# 1. Align experimental FASTQ to guide FASTA pool
aligner = PrivacyPreservingReferencePoolAligner(
    guide_fasta_files=[Path("ref1.fa.gz"), Path("ref2.fa.gz"), Path("ref3.fa.gz")],
    threads=8
)
aligner.align_query_to_pool(
    query_fastq_1=Path("experimental_R1.fastq.gz"),
    query_fastq_2=Path("experimental_R2.fastq.gz"),
    output_bam=Path("experimental.bam"),
    privacy_preserving=True
)

# 2. Generate GDiff differential encoding
encoder = GDiffEncoder(
    query_bam="experimental.bam",
    pool_bams=["ref1.bam", "ref2.bam", "ref3.bam"],
    reference_fasta="consensus.fa",
    min_base_quality=20,
    min_mapping_quality=20,
)
gdiff = encoder.compute_differential_encoding()
gdiff.save("experimental.gdiff.gz", compress=True)

# 3. Create privacy-preserving hypervector
encoder = SelectiveHDVEncoder(schema="clinical_risk", dimension=10000)
hdv = encoder.encode_from_gdiff("experimental.gdiff.gz")
```

### Critical Implementation Details

**@SQ Header Fix (minimap2 multi-part index issue):**
```bash
# ❌ This FAILS - no @SQ headers for multi-part index
minimap2 -ax sr guide_pool.mmi reads.fq | samtools sort -o out.bam -

# ✅ This WORKS - rebuild @SQ headers from reference FASTA
minimap2 -ax sr guide_pool.mmi reads_R1.fq reads_R2.fq > aligned.sam
samtools view -h -bt guide_pool_reference.fa aligned.sam | samtools sort -o sorted.bam -
```

**Time estimates for whole genome (22.5 GB FASTQ):**
- Minimap2 alignment: ~1-2 hours
- SAM→BAM conversion: ~10-15 minutes
- **Total: ~2.5-3 hours per reference**

---

## 🧬 GDiff Format: Purpose-Built Differential Encoding

**VCF is being replaced with GDiff** - purpose-built for GenomeVault's differential encoding.

### Architecture: GDiff as Local Database

```
┌─────────────────────────────────────────────────────────────┐
│ USER HARDWARE (Private, Encrypted at Rest)                 │
│  GDiff = Comprehensive "Source of Truth"                   │
│  Size: ~150 MB (uncompressed), ~15 MB (gzipped)           │
│  Security: AES-256 encrypted, NEVER transmitted            │
└─────────────────┬───────────────────────────────────────────┘
                  │ Analysis Schema Selection (10-300ms)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ HDV GENERATOR (On-Demand, Analysis-Specific)               │
│  Size: 512 bytes - 10 KB                                   │
└─────────────────┬───────────────────────────────────────────┘
                  │ Privacy-Preserving Query (Only HDV transmitted)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ GENOMEVAULT NETWORK (Public)                               │
│  Receives: HDV (1-10 KB)                                   │
│  Network traffic: 2000-20000× less than VCF approach       │
└─────────────────────────────────────────────────────────────┘
```

### Analysis Schemas (Pre-Configured Feature Sets)

| Schema | HDV Size | Use Case |
|--------|----------|----------|
| **simple_snp_lookup** | 512 B | Basic variant queries |
| **clinical_risk** | 2 KB | Clinical genomics |
| **pharmacogenomics** | 3 KB | Precision medicine |
| **ancestry_inference** | 5 KB | Ancestry analysis |
| **nanopore_structural_inference** | 10 KB | Long-read sequencing |
| **epigenetic_landscape** | 8 KB | Epigenomics |
| **full_research_profile** | 15 KB | Comprehensive research |

**Documentation:**
- `docs/GDIFF_RATIONALE.md` - Why GDiff is necessary
- `docs/ERROR_AWARE_ENCODING_GUIDE.md` - User guide for error-aware features
- `genomevault/differential_encoding/gdiff/` - Implementation

---

## 🔒 Production Pipeline (REAL ZK + PIR) ✅ PRODUCTION READY

**As of Oct 30, 2025: Both ZK proofs and PIR use REAL cryptographic implementations.**

### Quick Benchmark

```bash
python3 benchmarks/gdiff_minimal_benchmark.py
```

**What it runs:**
- GDiff streaming (1.2 GB file, 78.96M variants)
- HDC encoding (10,000D, Metal GPU)
- **REAL ZK proof** (Groth16, 0.40s, 739 bytes, 128-bit security)
- **REAL IT-PIR** (2-server, 12.75ms, 0 bits leaked)

**Security Guarantees:**
- k=3 anonymity
- HDC: 10,000D irreversible projection (39 KB)
- ZK: 128-bit security, reveals NOTHING
- PIR: Information-theoretic (quantum-resistant)

**Validation:** `benchmark_results/k3_whole_genome_benchmark/COMPLETE_PRODUCTION_VALIDATION_REPORT.md`

---

## 🚀 Quick Start

```bash
# Setup
pip install -e ".[dev]"
pytest tests/

# Hardware detection & optimization recommendations
python3 scripts/check_hardware_and_recommend.py

# Optimized k=13 pipeline (5× faster)
python3 scripts/run_enhanced_privacy_pipeline_optimized.py \
    --output-dir benchmark_results/enhanced_privacy_k13_phase1 \
    --num-references 12 \
    --use-sambamba --sambamba-threads 10 --sambamba-memory 8G \
    --parallel-bcftools --bcftools-threads 5 \
    --gpu-backend metal --threads 10

# Privacy-preserving query (ZK + PIR)
python genomevault/cli/privacy_query.py \
    --vcf query.vcf.gz --chrom chr22 --pos 4169 --ref C --alt A \
    --output query_results.json

# REST API
uvicorn genomevault.api.app:app --host 0.0.0.0 --port 8000
```

## 📊 Expected Performance

### GenomeVault Core (Layer 4)

| Stage | Duration | Details |
|-------|----------|---------|
| **Differential Encoding** | 1.36s | 120 variants, k=3 anonymity |
| **HDC Integration** | 0.5ms | 264× compression |
| **ZK Proof (Groth16)** | 0.74s | 743 bytes, 117,143 constraints |
| **PIR Query (IT-PIR)** | 4.33ms | 0.25% breach probability |
| **⚡ TOTAL** | **2.11s** | **5.92× speedup vs baseline** |

### k=13 Pipeline (Complete 4-Layer System)

| Phase | Time (12 refs) | Speedup | Effort |
|-------|---------------|---------|--------|
| Baseline | 12 hours | 1× | - |
| Phase 1 (sambamba, parallel bcftools, Metal GPU) | 6.4 hours | 1.9× | 30 min ⭐⭐⭐ |
| Phase 2 (+ index caching, AMX) | 4.0 hours | 3.0× | 5 hours ⭐⭐ |
| Phase 3 (+ chromosome-parallel sorting) | 2.4 hours | 5.0× | 8 hours ⭐ |

**Recommendation:** Deploy Phase 1 immediately (11× ROI).

---

## 🗺️ Navigation Guide

| What | Where |
|------|-------|
| **Production Pipeline** | `genomevault/pipelines/production_pipeline.py` |
| **GDiff Format** | `genomevault/differential_encoding/gdiff/` |
| **Secure Guide Reference** | `genomevault/differential_encoding/gdiff/secure_guide_reference_builder.py` |
| **Nucleotide Resolver** | `genomevault/query/nucleotide_resolver.py` |
| **Resolution-Aware HDV** | `genomevault/hypervector_transform/resolution_aware_encoder.py` |
| **Privacy Query CLI** | `genomevault/cli/privacy_query.py` |
| **ZK Circuits** | `genomevault/zk_proofs/circuits/` |
| **HDC Encoding** | `genomevault/hypervector_transform/` |
| **Hardware Backends** | `genomevault/compute/backend.py` |
| **Benchmarks** | `benchmarks/run_alignment_optimized_pipeline.py` |
| **Latest Results** | `benchmark_results/pipeline_results.json` |
| **Validation Proofs** | `benchmark_results/FINAL_VALIDATION_SUMMARY.md` |
| **SGRS Documentation** | `docs/SECURE_GUIDE_REFERENCE_SYSTEM.md` |

## 🔧 Essential Commands

```bash
# Development
pytest                                    # Run all tests
pytest tests/test_compute_backend.py      # Test hardware backends

# Hardware-aware optimization
python3 scripts/check_hardware_and_recommend.py

# Privacy-preserving query (ZK + PIR)
python genomevault/cli/privacy_query.py \
    --vcf query.vcf.gz --chrom chr22 --pos 4169 --ref C --alt A \
    --output query_results.json

# Production pipeline (GDiff → HDC → ZK → PIR)
python -m genomevault.cli.main pipeline production \
    experimental.gdiff.gz --dimension 10000 --zk --sample 1000

# ZK proofs
./benchmarks/setup_groth16_enhanced.sh   # One-time setup
python benchmarks/zk_groth16_benchmark.py
```

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Check `genomevault/core/exceptions.py` |
| Slow HDC batch | Enable GPU: `GENOMEVAULT_BACKEND=auto` |
| GPU not detected | Run `python tests/test_compute_backend.py` |
| ZK setup fails | Run `./benchmarks/setup_groth16_enhanced.sh` |

## 📚 Key Documentation

### Core Validation
- `benchmark_results/FINAL_VALIDATION_SUMMARY.md` - Complete system validation
- `benchmark_results/GENOMEVAULT_COMPLETE_SYSTEM_VALIDATION_PROOF_PACKAGE.md` - Full proof (1,930+ lines)

### Guides
- `docs/GDIFF_RATIONALE.md` - Why GDiff format
- `docs/ERROR_AWARE_ENCODING_GUIDE.md` - Error-aware features user guide
- `docs/guides/PROBABILISTIC_ALIGNMENT_PRIVACY_STACK.md` - Complete privacy stack
- `docs/optimization/MASTER_OPTIMIZATION_ROADMAP.md` - 4-phase optimization roadmap
- `data/acquisition_plan/QUICK_START_GUIDE.md` - Data acquisition (k=3→k=12)

### API & Integration
- `docs/API_USAGE_GUIDE.md` - REST API guide (550+ lines)
- `docs/api-docs/GETTING_STARTED_API.md` - Step-by-step for end users
- `docs/reports/BLOCKCHAIN_INTEGRATION_COMPLETE.md` - Blockchain integration

### Academic
- `docs/GenomeVault_Paper_Current/GenomeVault_Academic_Paper.pdf` - 31-page paper

## 🎓 Quick Tips

1. **Use hardware-aware optimized pipeline** for production (5× faster)
2. **Run hardware detection first** - `python3 scripts/check_hardware_and_recommend.py`
3. **Phase 1 is critical** - 30 min effort for 5.6 hours savings (11× ROI)
4. **Enable GPU** - Metal (Apple Silicon) or CUDA (NVIDIA) for 43× HDC speedup
5. **ZK proofs are CPU-bound** - GPU doesn't help with proving
6. **Reference pool must have k genomes** for k-anonymity
7. **Privacy-preserving queries** - Use `genomevault/cli/privacy_query.py` for cryptographically secure queries (0 bits leaked)

---

**Last Updated:** November 2025
**Version:** 1.2.0 (Error-Aware Encoding, Production Ready)
