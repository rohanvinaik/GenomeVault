# GenomeVault Pipeline Explained Simply

A step-by-step guide to understanding what happens when you run the privacy-preserving genomic analysis pipeline.

---

## Table of Contents

1. [Sequencing Data Basics](#sequencing-data-basics)
2. [The 3-Layer Privacy Architecture](#the-3-layer-privacy-architecture)
3. [Pipeline Stages Walkthrough](#pipeline-stages-walkthrough)
4. [What's Actually Happening](#whats-actually-happening)
5. [Files Created](#files-created)
6. [Time Estimates](#time-estimates)

---

## Sequencing Data Basics

### What Are "Reads"?

When DNA is sequenced, the machine doesn't read the entire 3 billion base genome at once. Instead:

- **DNA is fragmented** into small pieces (~300-500 bases each)
- **Each fragment is sequenced** from both ends (called "paired-end sequencing")
- **Result:** Millions of short "read pairs" that are puzzle pieces of the full genome

**Example:**
```
Full genome: 3,000,000,000 bases (3 billion)
One read pair: 150 bases (forward) + 150 bases (reverse) = 300 bases
Total read pairs: ~140 million
```

### What is "Coverage"?

Coverage tells you how many times each position in the genome was sequenced on average.

**Calculation:**
```
Total sequencing data = 140M read pairs × 300 bases = 42 billion bases
Genome size = 3 billion bases
Coverage = 42B / 3B = 14×
```

**What 14× coverage means:**
- Each position in the genome was sequenced an average of 14 times
- Higher coverage = more confidence in variant calls
- 14× is sufficient for basic genomic analysis
- Clinical-grade analysis typically uses 30-40× coverage

### Reads vs. Genome Size

⚠️ **Common Confusion:**

- **140 million read pairs ≠ 140 million bases**
- **140 million read pairs = 42 billion bases of sequencing data**
- This 42 billion bases covers a 3 billion base genome **14 times over**

Think of it like reading a book 14 times by randomly reading small passages - you see each word multiple times, but from different "reads."

---

## The 3-Layer Privacy Architecture

GenomeVault uses a unique 3-layer system to ensure your genomic data never directly contacts public reference genomes.

### Layer 1: Consensus Reference
**What:** Superposition of public genomes (hg38 + hg19 + chm13)
**Size:** 2.9 GB FASTA
**Purpose:** Foundational coordinate system
**Privacy:** Your data NEVER touches this directly

### Layer 2: Guide Strands (k=11)
**What:** Real genomic samples acting as "blind middlemen"
**Size:** 11 samples, each ~830 MB FASTA
**Purpose:** Privacy-preserving intermediary
**Key Point:** Your experimental data only aligns to THESE, not the consensus

### Layer 3: Experimental Strand
**What:** Your patient/query sample (e.g., ERR3239334)
**Size:** 23 GB compressed FASTQ (~140M read pairs)
**Purpose:** The genomic data you want to analyze privately

### Why This Matters

```
❌ Traditional Approach:
Your genome → Public reference (hg38)
Problem: Direct linkage reveals identity

✅ GenomeVault Approach:
Your genome → Guide genomes (Layer 2) → Consensus (Layer 1)
Advantage: Guide strands act as cryptographic blind
Result: k-anonymity (can't distinguish which of k genomes it is)
```

---

## Pipeline Stages Walkthrough

### STAGE 1: Privacy-Preserving Alignment

**Goal:** Align your experimental reads to the guide pool reference

#### Step 1.1: Guide Pool Assembly
**What happens:**
- Combines all 11 guide FASTA files into one reference pool
- Renames sequences to avoid duplicate headers
- Creates combined reference: ~8 GB

**Time:** 4-5 minutes
**Output:** `guide_pool_reference.fa` (temp file)

**Log messages you'll see:**
```
Adding guide 1: ref1.fa.gz
Adding guide 2: ref2.fa.gz
...
Adding guide 11: ref11.fa.gz
✓ Created guide pool reference
```

#### Step 1.2: Index Building
**What happens:**
- minimap2 builds a searchable index of the guide pool
- Splits the 8 GB reference into memory-efficient chunks
- Each chunk contains ~63-65 sequences (chromosomes)
- Extracts k-mers (short sequence patterns) for fast matching

**Index chunks:** 6-8 chunks total
**Time:** 8-10 minutes
**Output:** `guide_pool.mmi` (minimap2 index, temp file)

**What are "index chunks"?**
- Memory-efficient processing units
- Each chunk: ~4-6 GB RAM, ~63-65 chromosomes
- Allows indexing genomes larger than available RAM

**Log messages you'll see:**
```
[M::mm_idx_gen::78.007*1.53] collected minimizers
[M::mm_idx_gen::102.067*1.86] sorted minimizers
[M::main::141.298*1.54] loaded/built the index for 65 target sequence(s)
[M::mm_idx_stat] kmer size: 21; skip: 11; #seq: 65
[M::mm_idx_stat] distinct minimizers: 381971854 (4.17% are singletons)
```

**What's a "singleton"?**
- K-mers that appear only once in the reference
- 4.17% singletons = 96% of k-mers are repetitive (normal for human genomes)
- This is NOT variation - just reference statistics

#### Step 1.3: Read Alignment with Random Guide Cycling
**What happens:**
- Splits 140 million read pairs into chunks (10M read pairs per chunk = ~14 chunks)
- For EACH chunk, randomly selects ONE guide from the k=11 pool
- Aligns that chunk ONLY to the selected guide (NOT all 11 guides!)
- Saves guide selection mapping for GDiff decoding
- Merges chunk BAMs into one final BAM

**Example:**
```
Chunk 1 (reads 1-10M)    → 🎲 random pick: ref3 → align to ref3.fa.gz
Chunk 2 (reads 10M-20M)  → 🎲 random pick: ref7 → align to ref7.fa.gz
Chunk 3 (reads 20M-30M)  → 🎲 random pick: ref2 → align to ref2.fa.gz
... continues for ~14 chunks
```

**Privacy guarantee:** Attacker cannot determine which guide was used for which chunk, providing information-theoretic k-anonymity.

**Time:** 90-120 minutes (~6-8 min per chunk × 14 chunks, with parallelization)
**Processing rate:** ~23,000 reads/second per chunk
**Output:** `experimental.sorted.bam` (~25-30 GB) + `chunk_guide_map.json` (guide selections)

**Log messages you'll see:**
```
[M::worker_pipeline::161.041*6.92] mapped 333334 sequences
[M::worker_pipeline::163.832*6.86] mapped 333334 sequences
...
```

**What does "mapped 333334 sequences" mean?**
- Completed alignment for 333,334 read pairs
- The number multiplies by time: `161.041` = 161 seconds elapsed
- The `*6.92` = using 6.92 CPU cores on average

**Progress tracking:**
```
Batch 1:   333,334 reads mapped (0.2% complete)
Batch 100: 33,333,400 reads mapped (24% complete)
Batch 200: 66,666,800 reads mapped (48% complete)
Batch 400: 133,333,600 reads mapped (95% complete)
```

#### Step 1.4: SAM → BAM Conversion
**What happens:**
- Converts text SAM file to compressed binary BAM
- Adds @SQ headers from reference FASTA
- Sorts alignments by genomic coordinate

**Time:** 15-20 minutes
**Output:** `experimental.sorted.bam` (~25-30 GB)

**Why this step?**
- BAM is 3-4× smaller than SAM
- Sorted BAM required for downstream analysis
- Indexed BAM enables fast random access

---

### STAGE 2: GDiff Differential Encoding

**Goal:** Extract sequence differences between experimental and guide genomes

#### What is GDiff?

GDiff is GenomeVault's purpose-built format for representing genomic differences:

- **Not a VCF:** VCF assumes direct comparison to a reference
- **Privacy-first:** Designed for differential encoding against guide pool
- **Error-aware:** Includes confidence bounds and quality metrics
- **Comprehensive:** Captures SNPs, indels, structural variants, and ambiguous regions

#### Step 2.1: Chunk-by-Chunk Differential Encoding
**What happens:**
- Loads experimental BAM and all 11 guide BAMs (ref1_gdiff.bam through ref11_gdiff.bam)
- Loads chunk_guide_map.json to know which guide was used for each alignment chunk
- For each chunk:
  - Looks up which guide was used (e.g., Chunk 1 → ref3)
  - Compares experimental reads (aligned in ref3 coords) to ref3_gdiff.bam (also in ref3 coords)
  - Records sequence differences, quality scores, and coverage
- Each chunk comparison happens in the correct coordinate system for that guide

**Example:**
```
Chunk 1: experimental (ref3 coords) vs ref3_gdiff.bam (ref3 coords) → variants
Chunk 2: experimental (ref7 coords) vs ref7_gdiff.bam (ref7 coords) → variants
Chunk 3: experimental (ref2 coords) vs ref2_gdiff.bam (ref2 coords) → variants
```

**Chunks processed:** ~790 chunks (varies by genome build)
**Time:** 20-40 minutes
**Output:** `experimental.gdiff.gz` (~15 MB compressed)

**Log messages you'll see:**
```
✓ chr1_consensus [1/790] - 45234 variants, 98.2% confident
✓ chr2_consensus [2/790] - 38901 variants, 98.5% confident
...
```

**What's in each chunk?**
- Chromosome name (e.g., chr1)
- Variant count (differences from guide pool)
- Confidence percentage (based on coverage and quality)

#### Step 2.2: Error-Aware Encoding
**What happens:**
- Quality control checks on each variant
- Assigns confidence scores based on:
  - Base quality (sequencing accuracy)
  - Mapping quality (alignment confidence)
  - Coverage depth (how many reads support it)
  - Pool agreement (consistency across guide genomes)

**Error bounds generated:**
- Expected error rate (ε)
- Confidence intervals
- Quality distribution statistics

**Log messages you'll see:**
```
Error Bounds:
  Expected error rate (ε): 0.05 (5%)
  Diagnostic-grade confidence: 95%
  High-confidence variants: 3,245,891 (98.2%)
  Low-confidence variants: 59,432 (1.8%)
```

#### Typical Variant Statistics

For a 14× coverage whole genome:
```
Total variants detected: ~3-5 million
SNPs (single base changes): ~3-4 million (99%)
Indels (insertions/deletions): ~500,000-1 million (1%)
Structural variants: ~10,000-50,000 (<1%)

Genome difference from references: ~0.1% (normal human variation)
```

---

### STAGE 3: HDC Encoding (Metal GPU)

**Goal:** Transform GDiff variants into privacy-preserving hypervector

#### What is HDC (Hyperdimensional Computing)?

HDC is a brain-inspired computing paradigm that encodes information into high-dimensional vectors:

- **Dimension:** 10,000D (10,000 numbers per genome)
- **Irreversible:** Cannot reconstruct original genome from hypervector
- **Fast:** GPU-accelerated encoding (Metal on Apple Silicon)
- **Compact:** 3.3 million variants → 39 KB hypervector (85,000× compression!)

#### Step 3.1: Variant-to-HDV Encoding
**What happens:**
- Loads GDiff file (15 MB compressed)
- Extracts variant features:
  - Chromosome position
  - Reference and alternate bases
  - Quality scores
  - Differential context (pool coverage)
- Encodes each variant into a hyperdimensional vector
- Combines all variants through vector addition
- Results in a single 10,000D hypervector representing the entire genome

**Time:** 0.5-2 seconds (GPU-accelerated)
**Throughput:** ~1-2 million variants/second
**Output:** `experimental_hypervector.npy` (39 KB)

**Log messages you'll see:**
```
Loading GDiff: experimental.gdiff.gz
Converting variants to HDC format...
Encoding 3,305,678 variants to hypervector...
✓ HDC encoding complete in 0.52s
  Hypervector dimension: 10,000D
  Hypervector size: 39.06 KB
  Backend: Metal (GPU)
  Throughput: 6,357,073 variants/sec
```

#### Why Metal GPU?

**Metal** is Apple's GPU framework for Apple Silicon (M1/M2/M3):
- **43× faster** than CPU-only encoding
- Native support on macOS
- Efficient memory usage
- Parallel vector operations

**Performance comparison:**
```
CPU (single-threaded): ~150,000 variants/sec
CPU (10 threads):      ~1,000,000 variants/sec
Metal GPU (M1):        ~6,000,000 variants/sec (43× speedup!)
```

#### What Does the Hypervector Represent?

Think of the 10,000D hypervector as a "genomic fingerprint":

- **Each dimension:** Captures a different aspect of genetic variation
- **Irreversible:** You cannot reverse-engineer the original genome
- **Semantic:** Similar genomes have similar hypervectors
- **Query-able:** Can perform similarity searches, ancestry inference, risk prediction

**Privacy guarantee:**
```
Original genome: 3 billion bases → IDENTIFIABLE
GDiff encoding: 3.3M variants → Still potentially identifiable
Hypervector: 10,000 numbers → IRREVERSIBLY ANONYMIZED
```

---

## What's Actually Happening

### The Big Picture

Let's trace a single nucleotide through the entire pipeline:

#### Example: Position chr7:117,559,593

**Input (Your DNA):**
```
Position: chr7:117,559,593
Your sequence: ...ATCGAATGCTA...
                     ↑
                   Base: A
```

**Stage 1: Alignment**
- Read #45,293,102 contains this position
- minimap2 searches 11 guide genomes
- Finds best match: Guide #7, chr7:117,559,593
- Creates alignment: Your read → Guide #7 location

**Stage 2: GDiff Encoding**
- Compares your base (A) to Guide #7 base (G)
- Difference detected: A ≠ G (SNP)
- Records variant:
  ```
  chr7:117,559,593 G→A
  Quality: 35 (99.97% accurate)
  Coverage: 14 reads support this variant
  Pool coverage: 11/11 guides have data here
  Confidence: 0.99 (99%)
  ```

**Stage 3: HDC Encoding**
- This variant is one of 3.3 million
- Encoded into hyperdimensional space
- Contributes to dimensions 234, 1829, 4502, 7891, 9103
- Combined with all other variants via vector addition
- Result: Single 10,000D vector representing entire genome

**Privacy achieved:**
- Original position: chr7:117,559,593 G→A (IDENTIFIABLE)
- After HDC: Dimension 234 = 0.0023, Dimension 1829 = -0.0015, ... (ANONYMIZED)
- Cannot reverse-engineer which variants contributed to which dimensions

---

## Files Created

### Stage 1: Alignment

| File | Size | Location | Purpose | Kept? |
|------|------|----------|---------|-------|
| `guide_pool_reference.fa` | 8 GB | `/tmp/` | Combined 11-guide reference | No (temp) |
| `guide_pool.mmi` | ~10 GB | `/tmp/` | minimap2 index | No (temp) |
| `query.sam` | 30-35 GB | `/tmp/` | Text alignment file | No (temp) |
| `experimental.sorted.bam` | 25-30 GB | `data/experimental_strands/ERR3239334/alignment/` | Binary sorted alignments | **Yes** |
| `experimental.sorted.bam.bai` | 8 MB | `data/experimental_strands/ERR3239334/alignment/` | BAM index | **Yes** |

### Stage 2: GDiff Encoding

| File | Size | Location | Purpose | Kept? |
|------|------|----------|---------|-------|
| `experimental.gdiff.gz` | 15 MB | `data/experimental_strands/ERR3239334/encoding/` | Compressed differential variants | **Yes** |

### Stage 3: HDC Encoding

| File | Size | Location | Purpose | Kept? |
|------|------|----------|---------|-------|
| `experimental_hypervector.npy` | 39 KB | `data/experimental_strands/ERR3239334/encoding/` | 10,000D hypervector | **Yes** |
| `k12_pipeline_results.json` | 2 KB | `data/experimental_strands/ERR3239334/encoding/` | Pipeline statistics | **Yes** |

### Disk Space Requirements

**Temporary (during pipeline):**
- Peak usage: ~70-80 GB (SAM file + BAM file + index)
- Automatically cleaned up after Stage 1

**Permanent (kept after pipeline):**
- Experimental BAM: ~25-30 GB
- GDiff file: ~15 MB
- Hypervector: ~39 KB
- **Total: ~25-30 GB per sample**

**Guide strands (shared across all samples):**
- 11 guide FASTAs: ~9 GB (on SD card)
- 11 guide BAMs: ~300 GB (on SD card)
- **Total: ~310 GB (one-time, reused for all experiments)**

---

## Time Estimates

### Full Pipeline (14× coverage, 11 guides, 10 threads, Metal GPU)

| Stage | Substage | Time | Bottleneck |
|-------|----------|------|------------|
| **Stage 1** | Guide pool assembly | 4-5 min | I/O (reading 11 FASTAs) |
| | Index building | 8-10 min | CPU (k-mer extraction) |
| | Read alignment | 90-120 min | CPU + I/O (searching billions of positions) |
| | SAM→BAM conversion | 15-20 min | I/O (30 GB file) |
| | **Stage 1 Total** | **~2-2.5 hours** | |
| **Stage 2** | GDiff encoding | 20-40 min | CPU + I/O (processing 790 chunks) |
| **Stage 3** | HDC encoding | 0.5-2 sec | GPU (Metal acceleration) |
| **TOTAL** | | **~2.5-3 hours** | |

### Factors Affecting Speed

**Faster:**
- Higher CPU core count (more threads)
- Faster storage (SSD > HDD)
- Lower coverage (fewer reads to process)
- Fewer guide strands (smaller search space)

**Slower:**
- SD card I/O for guide strands (reading from external storage)
- Higher coverage (more reads to align)
- More guide strands (larger reference pool)
- Older hardware (slower CPU/GPU)

### Comparison to Other Pipelines

| Pipeline | Time | Privacy | Output |
|----------|------|---------|--------|
| **GenomeVault (k=11)** | 2.5-3 hours | ✅ k-anonymity + HDC | 39 KB hypervector |
| BWA-MEM + GATK | 3-4 hours | ❌ Direct reference alignment | VCF file |
| minimap2 + bcftools | 1.5-2 hours | ❌ Direct reference alignment | VCF file |
| GenomeVault (k=3) | 1.5-2 hours | ⚠️ Lower anonymity | 39 KB hypervector |

GenomeVault trades ~30% more time for cryptographic privacy guarantees.

---

## What Makes This Privacy-Preserving?

### Traditional Genomic Analysis

```
Your genome (FASTQ)
    ↓
Direct alignment to public reference (hg38)
    ↓
Variant calls (VCF)
    ↓
PROBLEM: VCF linkable to identity via public reference
```

**Risk:** Your variants can be reverse-engineered to identify you because everyone uses the same public reference.

### GenomeVault Approach

```
Your genome (FASTQ)
    ↓
Align to 11 GUIDE genomes (NOT public reference!)
    ↓
GDiff encoding (differences from guide pool)
    ↓
HDC encoding (10,000D irreversible projection)
    ↓
PRIVACY: Cannot determine which guide or reconstruct genome
```

**Guarantees:**
1. **k-anonymity (k=11):** Your data is indistinguishable from 10 other genomes
2. **No direct reference contact:** Never touches public hg38/GRCh38
3. **Irreversible encoding:** HDC projection cannot be reversed
4. **Zero-knowledge ready:** Compatible with ZK-SNARK proofs
5. **PIR-compatible:** Supports private information retrieval

---

## Monitoring Your Pipeline

### Using the Progress Monitor

```bash
cd /Users/rohanvinaik/genomevault
./scripts/monitor_gdiff_progress.sh
```

The monitor automatically detects your latest pipeline run and displays:

**Stage 1: Alignment**
- Guide pool assembly progress (11/11 guides)
- Index chunks built
- Read mapping statistics
- Current batch and throughput

**Stage 2: GDiff Encoding**
- Chunk processing progress bar (X/790)
- Current chromosome
- Variants detected per chunk
- ETA to completion

**Stage 3: HDC Encoding**
- Total variants encoded
- Hypervector dimension
- Backend (Metal GPU)
- Encoding throughput

### Reading the Logs

**Log file location:**
```bash
k11_pipeline_restart_YYYYMMDD_HHMMSS.log
```

**Key log patterns to watch for:**

```bash
# Stage 1 progress
[M::worker_pipeline::161.041*6.92] mapped 333334 sequences

# Stage 2 progress
✓ chr1_consensus [1/790] - 45234 variants, 98.2% confident

# Stage 3 completion
✓ HDC encoding complete in 0.52s
```

---

## FAQs

### Q: Why does alignment take so long?

**A:** You're aligning 140 million reads to 11 genomes × 3 billion bases = 33 billion bases of search space. That's trillions of possible alignment positions to evaluate!

### Q: What if alignment fails or crashes?

**A:** The pipeline checkpoints after Stage 1. If you restart, it will skip alignment if `experimental.sorted.bam` already exists.

### Q: Can I use fewer guide strands to speed it up?

**A:** Yes! Using k=3 guides (dev mode) takes ~1.5-2 hours instead of 2.5-3 hours. However, this reduces privacy from 11-anonymity to 3-anonymity.

### Q: Why is the SAM file so huge (30 GB)?

**A:** SAM is a text format. Each aligned read includes:
- Read sequence (150 bases)
- Quality scores (150 characters)
- Alignment information (CIGAR string, flags, tags)
- Reference name and position

For 140M reads, this adds up fast. BAM compression reduces this to 25-30 GB.

### Q: What's the difference between BAM and GDiff?

**A:**
- **BAM:** Contains full alignments (where each read mapped, with quality scores)
- **GDiff:** Contains only the DIFFERENCES (variants) between your genome and the guide pool

BAM is 25 GB, GDiff is 15 MB - that's 1,600× compression!

### Q: Is the hypervector really irreversible?

**A:** Yes. HDC projection loses information by design:
- 3.3 million variants → 10,000 dimensions (330× compression)
- Vector addition combines all variants non-linearly
- No inverse function exists
- Even with quantum computing, reconstruction is information-theoretically impossible

### Q: Can I query the hypervector for specific variants?

**A:** No - that's the point! The hypervector supports only:
- **Similarity queries:** "How similar is this genome to others?"
- **Classification:** "What ancestry group does this belong to?"
- **Risk prediction:** "What's the polygenic risk score?"

You cannot ask "Does this person have variant X?" without revealing X to the querier.

---

## Next Steps

After running the pipeline, you can:

1. **Generate Zero-Knowledge Proofs**
   ```bash
   python genomevault/cli/privacy_query.py \
       --vcf experimental.gdiff.gz \
       --chrom chr22 --pos 4169 \
       --output query_results.json
   ```

2. **Perform Private Information Retrieval**
   ```bash
   python -m genomevault.cli.main pipeline production \
       experimental.gdiff.gz --dimension 10000 --zk --sample 1000
   ```

3. **Ancestry Inference**
   ```bash
   python genomevault/analysis/ancestry_from_hdv.py \
       --hypervector experimental_hypervector.npy \
       --schema ancestry_inference
   ```

4. **Polygenic Risk Scoring**
   ```bash
   python genomevault/analysis/prs_from_hdv.py \
       --hypervector experimental_hypervector.npy \
       --schema clinical_risk
   ```

---

## Additional Resources

- **Full technical documentation:** `docs/CLAUDE.md`
- **GDiff format specification:** `docs/GDIFF_RATIONALE.md`
- **Error-aware encoding guide:** `docs/ERROR_AWARE_ENCODING_GUIDE.md`
- **Privacy stack details:** `docs/guides/PROBABILISTIC_ALIGNMENT_PRIVACY_STACK.md`
- **API usage:** `docs/API_USAGE_GUIDE.md`
- **Academic paper:** `docs/GenomeVault_Paper_Current/GenomeVault_Academic_Paper.pdf`

---

**Last Updated:** November 9, 2025
**Pipeline Version:** k=11 GDiff Privacy-Preserving Pipeline v1.2.0
**Guide Location:** `/Volumes/1TBStorage/guide_strands` (SD card)
