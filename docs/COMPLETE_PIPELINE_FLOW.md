# GenomeVault Complete Pipeline Flow

**Last Updated:** November 23, 2025

This document traces the entire GenomeVault pipeline from raw data acquisition through privacy-preserving encoding to zero-knowledge queries.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Data Acquisition & Alignment                              │
│ ├── Public References (hg38, hg19, chm13)                          │
│ ├── Guide Samples (k=12 diverse whole genomes)                     │
│ └── Experimental Sample (patient/query data)                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Privacy-Preserving 3-Layer Architecture                   │
│ ├── Layer 1: Consensus (hg38+hg19+chm13 Byzantine consensus)       │
│ ├── Layer 2: Guide FASTAs (k=12 blind middleman references)        │
│ └── Layer 3: GDiff Encoding (experimental vs guides)               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: HDC Encoding (3-Bank Split Architecture)                  │
│ ├── Bank 1: Hydrophobic (T vs A, transparent to G/C)               │
│ ├── Bank 2: Major Groove (G vs C, transparent to A/T)              │
│ └── Bank 3: Hinge (Y-R vs R-Y structural flexibility)              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Split Ternary Quantization                                │
│ ├── Vector 1 (GC-dominant): [AT=0, GC, Hinge]                      │
│ └── Vector 2 (AT-dominant): [AT, GC=0, Hinge]                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Query & Validation Experiments                            │
│ ├── Lens-Aware SIMD Query Engine                                   │
│ ├── Multi-Stage Query Architecture                                 │
│ └── Biophysical Context Validation                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 6: Zero-Knowledge Proofs & Private Queries                   │
│ ├── ZK Proofs (Groth16, 128-bit security)                          │
│ └── IT-PIR (Information-theoretic, quantum-resistant)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## STAGE 1: Data Acquisition & Alignment

### 1.1 Reference Genome Download

**Purpose:** Obtain public reference genomes for Layer 1 consensus building.

**Files:**
- `scripts/download_references.sh` - Downloads hg38, hg19, chm13
- Output: `data/reference_genomes/`
  - `hg38.fa.gz`
  - `hg19.fa.gz`
  - `chm13v2.0.fa.gz`

### 1.2 Guide Sample Acquisition

**Purpose:** Download k=12 diverse whole-genome samples for Layer 2 blind middleman.

**Files:**
- `data/acquisition_plan/guide_samples.txt` - Sample accession IDs
- `scripts/download_whole_genomes_sequential.sh` - Download FASTQ files
- Output: `data/downloaded/fastq/`
  - ERR3239276, ERR3239334, ERR3239454, ERR3239475 (African)
  - ERR3239548, ERR3239590, ERR3239920 (European)
  - ERR3239578, ERR3239612 (East Asian)
  - ERR3239756, ERR3239778 (African/European admixed)
  - ERR3239912, ERR3239934 (South Asian/European admixed)

**Data Size:** ~22.5 GB FASTQ per sample (paired-end)

### 1.3 Experimental Sample

**Purpose:** Patient/query data to be encoded privately.

**Example:** ERR3239334 (used throughout validation)
- Location: `data/downloaded/fastq/ERR3239334/`
- Files: `ERR3239334_1.fastq.gz`, `ERR3239334_2.fastq.gz`

---

## STAGE 2: Privacy-Preserving 3-Layer Architecture

### 2.1 Layer 1: Consensus Building

**Purpose:** Create superposition of public references with Byzantine consensus.

**Key File:**
- `genomevault/alignment/superposition_consensus_builder.py`

**Process:**
```bash
# Align each reference to create consensus
# Output: consensus.fa (2.9 GB)
```

**Location:** `data/consensus/consensus.fa`

**Algorithm:** Byzantine consensus with positional uncertainty to prevent direct linkage.

### 2.2 Layer 2: Guide FASTA Creation

**Purpose:** Create k=12 blind middleman references aligned to consensus coordinates.

**Key File:**
- `scripts/run_enhanced_privacy_pipeline_optimized.py` (PRIMARY)

**Process:**
```bash
# For each guide sample (ref1-ref12):
# 1. Align guide FASTQ to consensus
minimap2 -ax sr consensus.fa guide_R1.fastq.gz guide_R2.fastq.gz | \
    sambamba sort -o ref${i}.sorted.bam

# 2. Extract consensus sequence (guide FASTA)
samtools consensus ref${i}.sorted.bam | pigz > ref${i}.fa.gz

# 3. Re-align guide FASTQ to guide FASTA (for GDiff comparison)
minimap2 -ax sr ref${i}.fa.gz guide_R1.fastq.gz guide_R2.fastq.gz | \
    sambamba sort -o ref${i}_gdiff.bam
```

**Output:**
- `data/guide_strands/ref1.fa.gz` through `ref12.fa.gz` (~828 MB each)
- `data/guide_strands/ref1.sorted.bam` through `ref12.sorted.bam` (25-30 GB each)
- **Total:** ~338 GB (4+ days of processing)

**Critical:** These files are IRREPLACEABLE. Never delete without explicit confirmation.

### 2.3 Layer 3: GDiff Differential Encoding

**Purpose:** Encode experimental data as differences from guide pool (privacy-preserving).

**Key Files:**
- `genomevault/differential_encoding/align_to_reference_pool.py`
- `genomevault/differential_encoding/gdiff/gdiff_encoder.py`
- `genomevault/differential_encoding/gdiff/secure_guide_reference_builder.py`

**Process:**
```python
from genomevault.differential_encoding.align_to_reference_pool import PrivacyPreservingReferencePoolAligner
from genomevault.differential_encoding.gdiff import GDiffEncoder

# 1. Align experimental FASTQ to guide FASTA pool
aligner = PrivacyPreservingReferencePoolAligner(
    guide_fasta_files=[Path(f"ref{i}.fa.gz") for i in range(1, 13)],
    threads=8
)
aligner.align_query_to_pool(
    query_fastq_1=Path("ERR3239334_R1.fastq.gz"),
    query_fastq_2=Path("ERR3239334_R2.fastq.gz"),
    output_bam=Path("experimental.bam")
)

# 2. Generate GDiff differential encoding
encoder = GDiffEncoder(
    query_bam="experimental.bam",
    pool_bams=[f"ref{i}_gdiff.bam" for i in range(1, 13)],
    reference_fasta="consensus.fa"
)
gdiff = encoder.compute_differential_encoding()
gdiff.save("experimental.gdiff.gz", compress=True)
```

**Output:**
- `data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz` (29 MB)

**Format:** Custom GDiff format captures sequence-level differences without exposing absolute positions.

**Documentation:** `docs/GDIFF_RATIONALE.md`

---

## STAGE 3: HDC Encoding (3-Bank Split Architecture)

### 3.1 Core Encoder Implementation

**Key File:**
- `genomevault/hypervector_transform/complementary_pair_encoder.py`

**Features:**
- Watson-Crick complementary pairs (AT/GC)
- Sparse position codebook (1 dimension per position)
- SNR = 2D/N formula
- 99.92% baseline accuracy
- No imports from genomevault (self-contained)

**Parameters:**
```python
N = 1024      # Chunk size (bp)
D = 5120      # Dimension (bits)
OVERLAP = 128 # 12.5% overlap
STRIDE = 896  # N - OVERLAP
```

### 3.2 3-Bank Split Architecture

**Key File:**
- `genomevault/hdv_validation/hdc_experimentation/encoders/encode_3bank_split_architecture.py`

**Architecture:**
```
Bank 1 (Hydrophobic):
  T = +1, A = -1, G = 0, C = 0
  Captures hydrophobic/hydrophilic distinctions

Bank 2 (Major Groove):
  G = +1, C = -1, A = 0, T = 0
  Captures major groove shape differences

Bank 3 (Hinge):
  YR (CT, TG, CA, AG) = +1
  RY (TC, GT, AC, GA) = -1
  Captures structural flexibility at dinucleotide junctions
```

**Process:**
```bash
cd genomevault/hdv_validation/hdc_experimentation
python3 encoders/encode_3bank_split_architecture.py
```

**Input:** `experimental.gdiff.gz` (29 MB)

**Output:**
- `output/encoded_genome_3banks.h5` (5.3 GB)
- Shape: `[3,370,053 chunks, 3 banks, 5120 dimensions]`

**Why 3 Banks:** Each bank captures orthogonal biophysical properties, allowing transparent encoding where irrelevant nucleotides map to zero.

---

## STAGE 4: Split Ternary Quantization

### 4.1 Quantization Implementation

**Key File:**
- `genomevault/hdv_validation/hdc_experimentation/quantization/split_ternary_quantizer.py`

**Purpose:** Compress 3-bank float32 encoding to 6-bank ternary (-1, 0, +1) with √2 SNR improvement.

**Architecture:**
```python
# Split into two orthogonal 3D vectors
Vector 1 (GC-dominant): [AT=0, GC, Hinge]
Vector 2 (AT-dominant): [AT, GC=0, Hinge]

# Each vector independently captures signal
# √2 SNR improvement per vector
```

**Process:**
```bash
cd genomevault/hdv_validation/hdc_experimentation
python3 quantization/split_ternary_quantizer.py
```

**Input:** `output/encoded_genome_3banks.h5` (5.3 GB)

**Output:**
- `output/encoded_genome_6banks_split_ternary.h5` (6.1 GB)
- Shape: `[3,370,053 chunks, 6 banks, 5120 dimensions]`
- Values: {-1, 0, +1} (ternary)

**Storage:** Each ternary value = 2 bits (4 states: -1, 0, +1, unused)

---

## STAGE 5: Query & Validation Experiments

### 5.1 Lens-Aware SIMD Query Engine

**Key File:**
- `genomevault/hdv_validation/hdc_experimentation/query/lens_aware_simd_query_engine.py`

**Features:**
- SIMD-optimized dot products (Numba JIT)
- 1.92 μs median query time
- Lens-aware decoding with texture classification
- Binary search for optimal lens confidence

**Process:**
```python
from genomevault.hdv_validation.hdc_experimentation.query.lens_aware_simd_query_engine import LensAwareSIMDQueryEngine

engine = LensAwareSIMDQueryEngine(
    encoded_file="output/encoded_genome_6banks_split_ternary.h5",
    gdiff_file="data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz"
)

# Query for variant at specific position
result = engine.query_position(chrom="chr22", pos=16050607)
```

**Performance:**
- Query latency: 1.92 μs (median)
- Throughput: 520,833 queries/sec
- Accuracy: 99.89% (nucleotide-level resolution)

### 5.2 Threshold Grid Search Optimization

**Key File:**
- `genomevault/hdv_validation/hdc_experimentation/query/threshold_grid_search.py`

**Purpose:** Optimize biophysical layer thresholds for texture classification.

**Search Space:**
- AT_DOMINANT: ratio [1.3-1.7], percentile [60-80]
- GC_DOMINANT: ratio [1.1-1.3], percentile [50-70]
- EXTREME_AT: percentile [95-99]
- EXTREME_GC: percentile [96-99]
- Total: 12,500 configurations tested

**Best Configuration (Nov 23, 2025):**
```python
at_dominant_ratio = 1.7
at_dominant_percentile = 75
gc_dominant_ratio = 1.1
gc_dominant_percentile = 50
extreme_at_percentile = 97  # 3.00% frequency, 0.0% error - PERFECT
extreme_gc_percentile = 98  # 2.00% frequency, 0.0% error - PERFECT
```

**Run:**
```bash
cd genomevault/hdv_validation/hdc_experimentation
python3 query/threshold_grid_search.py
```

### 5.3 Experimental Validation

**Documentation:**
- `genomevault/hdv_validation/hdc_experimentation/docs/theory/MULTI_STAGE_QUERY_ARCHITECTURE_EXPERIMENTS.md`
- `genomevault/hdv_validation/hdc_experimentation/docs/theory/EXPERIMENTAL_DATA_COLLECTION.md`

**Validation Files:**
- `genomevault/hdv_validation/hdc_experimentation/query/experiment_0_biophysical_context_validation.py`

**Key Experiments:**

1. **Biophysical Context Validation**
   - Verify texture classification accuracy
   - Test layer-specific query strategies
   - Validate transparent encoding hypothesis

2. **Multi-Stage Query Architecture**
   - Stage 1: Magnitude-based texture detection
   - Stage 2: Lens selection (AT-dominant, GC-dominant, extreme)
   - Stage 3: Fine-grained nucleotide resolution

3. **Performance Benchmarking**
   - Query latency distribution
   - Accuracy vs confidence thresholds
   - SNR analysis per bank

---

## STAGE 6: Zero-Knowledge Proofs & Private Queries

### 6.1 Zero-Knowledge Proofs (Groth16)

**Key Files:**
- `genomevault/zk_proofs/prover.py` - Main prover interface
- `genomevault/zk_proofs/backends/real_circom_backend.py` - Circom/SnarkJS integration
- `genomevault/zk_proofs/circuits/` - Circuit definitions

**Purpose:** Prove possession of variant without revealing position or nucleotide.

**Process:**
```python
from genomevault.zk_proofs.prover import ZKProver

prover = ZKProver(backend="circom")

# Generate proof for variant query
proof = prover.prove(
    variant_position=16050607,
    nucleotide="A",
    hdv_encoding=hdv_vector
)

# Verify proof (anyone can verify without learning private data)
is_valid = prover.verify(proof)
```

**Performance:**
- Proving time: 0.40s
- Proof size: 739 bytes
- Security: 128-bit
- Circuit constraints: 117,143

**Benchmark:**
```bash
python3 benchmarks/zk_proof_real_benchmark.py
```

### 6.2 Information-Theoretic PIR

**Key Files:**
- `genomevault/pir/advanced/it_pir.py` - IT-PIR implementation
- `genomevault/pir/advanced/robust_it_pir.py` - Robust variant with error correction

**Purpose:** Query database without revealing which record was queried (0 bits leaked).

**Process:**
```python
from genomevault.pir.advanced.it_pir import ITPIRClient, ITPIRServer

# Client creates PIR query
client = ITPIRClient(database_size=1000000)
query = client.create_query(index=42)  # Want record 42

# Server processes query (learns nothing about index)
server = ITPIRServer(database)
response = server.process_query(query)

# Client extracts result
record = client.extract_result(response)
```

**Properties:**
- Information leakage: 0 bits (information-theoretic security)
- Quantum-resistant: Yes (no computational assumptions)
- Query latency: 4.33ms (2-server model)
- Breach probability: 0.25%

**Benchmark:**
```bash
python3 benchmarks/pir_performance_benchmark.py
```

### 6.3 End-to-End Privacy-Preserving Query

**Key File:**
- `benchmarks/run_k3_gdiff_production_pipeline.py`

**Complete Flow:**
```python
# 1. Load GDiff differential encoding
gdiff = GDiff.load("experimental.gdiff.gz")

# 2. Encode to HDC
encoder = UnifiedGenomicEncoder(dimension=10000)
hdv = encoder.encode_from_gdiff(gdiff)

# 3. Generate ZK proof
proof = prover.prove(variant="chr22:16050607:A", hdv=hdv)

# 4. Query with PIR
pir_query = pir_client.create_query(proof)
result = pir_server.process(pir_query)
```

**Performance (Complete Pipeline):**
- GDiff streaming: 1.36s (120 variants)
- HDC integration: 0.5ms
- ZK proof: 0.74s
- PIR query: 4.33ms
- **Total: 2.11s** (5.92× speedup vs baseline)

**Benchmark:**
```bash
python3 benchmarks/run_k3_gdiff_production_pipeline.py
```

---

## Data Flow Summary

### Size Progression

```
Stage 1: Raw FASTQ Data
├── Experimental sample: 22.5 GB (paired-end)
├── Guide samples (k=12): 270 GB total
└── Reference genomes: 3 × 3 GB = 9 GB

Stage 2: Aligned Data
├── Consensus FASTA: 2.9 GB
├── Guide FASTAs (k=12): 12 × 828 MB = 9.9 GB
├── Guide BAMs (k=12): 12 × 27 GB = 324 GB
└── GDiff encoding: 29 MB ✓ (1,100× compression)

Stage 3: HDC Encoding
└── 3-bank encoding: 5.3 GB

Stage 4: Quantization
└── 6-bank ternary: 6.1 GB

Stage 5: Query
├── Query latency: 1.92 μs
└── Accuracy: 99.89%

Stage 6: ZK/PIR
├── ZK proof: 739 bytes
└── PIR query: 4.33ms
```

### Critical Files (Production Pipeline)

**Data Generation:**
1. `scripts/run_enhanced_privacy_pipeline_optimized.py` - k=12 privacy pipeline
2. `data/guide_strands/ref1-12.fa.gz` - Guide FASTAs (338 GB total)
3. `data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz` - GDiff (29 MB)

**HDC Encoding:**
4. `genomevault/hypervector_transform/complementary_pair_encoder.py` - Core encoder
5. `genomevault/hdv_validation/hdc_experimentation/encoders/encode_3bank_split_architecture.py` - 3-bank encoder
6. `genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5` - Encoded genome (5.3 GB)

**Quantization:**
7. `genomevault/hdv_validation/hdc_experimentation/quantization/split_ternary_quantizer.py` - Quantizer
8. `genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_ternary.h5` - Quantized (6.1 GB)

**Query & Validation:**
9. `genomevault/hdv_validation/hdc_experimentation/query/lens_aware_simd_query_engine.py` - Query engine
10. `genomevault/hdv_validation/hdc_experimentation/query/threshold_grid_search.py` - Optimization
11. `genomevault/hdv_validation/hdc_experimentation/query/experiment_0_biophysical_context_validation.py` - Validation

**ZK/PIR:**
12. `genomevault/zk_proofs/prover.py` - ZK proofs
13. `genomevault/pir/advanced/it_pir.py` - PIR queries
14. `benchmarks/run_k3_gdiff_production_pipeline.py` - End-to-end validation

---

## Performance Summary

### Processing Time (Full Pipeline)

| Stage | Time | Details |
|-------|------|---------|
| **Data Acquisition** | 4+ days | k=12 samples, 338 GB |
| **Layer 1 (Consensus)** | ~30 min | Byzantine consensus |
| **Layer 2 (Guides)** | ~30 hours | 12 × 2.5 hours alignment |
| **Layer 3 (GDiff)** | ~3 hours | Experimental alignment + encoding |
| **HDC Encoding** | ~45 min | 3-bank split architecture |
| **Quantization** | ~20 min | Split ternary |
| **Query Optimization** | ~40 min | 12,500 configurations |
| **ZK Proof** | 0.40s | Per query |
| **PIR Query** | 4.33ms | Per query |

### Storage Requirements

| Component | Size | Compressible | Critical |
|-----------|------|--------------|----------|
| Guide FASTAs | 9.9 GB | No | ⚠️ IRREPLACEABLE |
| Guide BAMs | 324 GB | No | ⚠️ IRREPLACEABLE |
| GDiff | 29 MB | Yes | ✓ Regenerable |
| 3-bank H5 | 5.3 GB | No | ✓ Regenerable |
| 6-bank H5 | 6.1 GB | No | ✓ Regenerable |

---

## Key Documentation

### Architecture & Theory
- `CLAUDE.md` - Project overview and quick start
- `docs/GDIFF_RATIONALE.md` - Why GDiff format is necessary
- `docs/SECURE_GUIDE_REFERENCE_SYSTEM.md` - SGRS documentation
- `docs/guides/PROBABILISTIC_ALIGNMENT_PRIVACY_STACK.md` - Complete privacy stack

### HDC Experimentation
- `genomevault/hdv_validation/hdc_experimentation/docs/theory/MULTI_STAGE_QUERY_ARCHITECTURE_EXPERIMENTS.md`
- `genomevault/hdv_validation/hdc_experimentation/docs/theory/EXPERIMENTAL_DATA_COLLECTION.md`
- `genomevault/hdv_validation/hdc_experimentation/README.md`

### Validation & Proofs
- `benchmark_results/FINAL_VALIDATION_SUMMARY.md` - Complete system validation
- `benchmark_results/GENOMEVAULT_COMPLETE_SYSTEM_VALIDATION_PROOF_PACKAGE.md` - Full proof (1,930+ lines)
- `benchmark_results/k3_whole_genome_benchmark/COMPLETE_PRODUCTION_VALIDATION_REPORT.md`

### Migration & Cleanup
- `docs/ENCODER_MIGRATION_2025-11-23.md` - Encoder cleanup summary
- `genomevault/hypervector_transform/encoders/README_ENCODERS.md` - Encoder directory status

---

## Recovery & Troubleshooting

### Regenerating Data

**If GDiff is lost:**
```bash
cd /Users/rohanvinaik/genomevault
python3 scripts/run_enhanced_privacy_pipeline_optimized.py \
    --output-dir data/experimental_strands/ERR3239334 \
    --num-references 12 \
    --experimental-sample ERR3239334
```

**If 3-bank encoding is lost:**
```bash
cd genomevault/hdv_validation/hdc_experimentation
python3 encoders/encode_3bank_split_architecture.py
```

**If quantization is lost:**
```bash
cd genomevault/hdv_validation/hdc_experimentation
python3 quantization/split_ternary_quantizer.py
```

### Critical Files (NEVER DELETE)

⚠️ **Guide FASTAs:** `data/guide_strands/ref1-12.fa.gz` (338 GB, 4+ days to regenerate)
⚠️ **Guide BAMs:** `data/guide_strands/ref1-12.sorted.bam` (324 GB, 4+ days to regenerate)

These files are backed up via Time Machine. Always verify Time Machine backups before any deletion.

---

## Current State (November 23, 2025)

### Completed
✅ Full project cleanup (34 deprecated files archived)
✅ k=12 privacy pipeline with guide FASTAs
✅ 3-bank split architecture with biophysical transparency
✅ Split ternary quantization (√2 SNR improvement)
✅ Lens-aware SIMD query engine (1.92 μs median)
✅ Threshold grid search optimization (PERFECT extreme accuracy)
✅ Real ZK proofs (Groth16, 128-bit security)
✅ IT-PIR (information-theoretic, quantum-resistant)

### Active Development
🔄 Multi-stage query architecture experiments
🔄 Biophysical context validation
🔄 Experimental data collection for publication

### Production Ready
✓ Complete end-to-end pipeline verified
✓ All production files intact and tested
✓ Performance benchmarks documented
✓ Security guarantees validated

---

**For questions or issues, see `CLAUDE.md` for quick start guide and key commands.**
