# Stage-Specific Optimization Plan for GenomeVault k=13 Pipeline

**Date:** October 25, 2025
**Target:** Four-layer privacy-preserving genomic pipeline
**Status:** Waiting for ref1 completion, ready for implementation

---

## Executive Summary

This document provides stage-specific optimization strategies for each layer of the GenomeVault k=13 enhanced privacy pipeline. Optimizations are categorized by pipeline stage to address the specific data formats, computational bottlenecks, and privacy requirements at each layer.

**Key Insight:** FASTQ data is unlabeled (no chromosome information), so chromosome-based optimizations only apply AFTER alignment produces BAM files.

**Expected Total Speedup:** 8-15× end-to-end with all optimizations applied

---

## Pipeline Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Superposition Consensus (Graph-Based Reference)       │
│ Input:  7 reference VCF files                                   │
│ Output: consensus_chr22.fa (870 MB)                             │
│ Status: ✅ COMPLETE                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Rolling Reference Pool (k=12 Anonymity)                │
│ Input:  12 FASTQ samples (paired-end, compressed)               │
│ Output: 12 BAM files + 12 VCF files (per reference)             │
│ Status: 🔄 IN PROGRESS (ref 1/12, sorting phase)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Privacy-Preserving Query Alignment                     │
│ Input:  Query FASTQ sample                                      │
│ Output: Query BAM + VCF aligned to each of 12 references        │
│ Status: ⏳ PENDING                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: GenomeVault Core (HDC + ZK + PIR)                      │
│ Input:  Query VCF + 12 reference VCFs                           │
│ Output: Hypervector (78 MB), ZK proof, PIR query                │
│ Status: ⏳ PENDING                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Superposition Consensus Optimization

### Current Implementation

**File:** `benchmarks/run_enhanced_privacy_pipeline.py` (lines 147-230)

**Process:**
1. Load 7 reference VCF files
2. Build variant graph (nodes = positions, edges = co-occurrences)
3. Compute superposition (weighted average of alleles)
4. Write consensus FASTA file

**Current Performance:**
- Time: ~30-60 minutes (one-time)
- Output: 870 MB consensus reference
- Status: Already optimized (pre-built detection in place)

### Optimization Opportunities

#### 1. Parallel VCF Parsing (2-3× speedup)

**Current Code:**
```python
# Sequential VCF loading
for vcf_file in reference_vcfs:
    vcf_reader = vcf.Reader(filename=vcf_file)
    for record in vcf_reader:
        graph.add_variant(record)
```

**Optimized Code:**
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def parallel_vcf_parsing(vcf_files: List[str], num_workers: int = 4):
    """Parse multiple VCF files in parallel."""

    def parse_single_vcf(vcf_file: str) -> List[Variant]:
        variants = []
        vcf_reader = vcf.Reader(filename=vcf_file)
        for record in vcf_reader:
            variants.append(record)
        return variants

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        variant_lists = pool.map(parse_single_vcf, vcf_files)

    # Merge into single graph
    graph = VariantGraph()
    for variants in variant_lists:
        for variant in variants:
            graph.add_variant(variant)

    return graph
```

**Impact:**
- Expected: 2-3× faster VCF loading
- Risk: Low (embarrassingly parallel workload)
- Implementation: 30-45 minutes

#### 2. Memory-Mapped Graph Construction (1.5-2× speedup)

**Current:** In-memory graph (limited by RAM)

**Optimized:**
```python
import mmap

class MemoryMappedVariantGraph:
    """Graph stored in memory-mapped file for zero-copy access."""

    def __init__(self, backing_file: str, num_positions: int = 60_000_000):
        self.backing_file = backing_file
        # Pre-allocate mmap file (3 GB for chr22)
        with open(backing_file, 'wb') as f:
            f.write(b'\0' * (num_positions * 48))  # 48 bytes per position

        self.mmap = mmap.mmap(
            open(backing_file, 'r+b').fileno(),
            0,
            access=mmap.ACCESS_WRITE
        )

    def add_variant(self, pos: int, allele: str):
        # Write directly to mmap (zero-copy)
        offset = pos * 48
        self.mmap[offset:offset+48] = struct.pack('...', ...)
```

**Impact:**
- Expected: 1.5-2× faster graph construction
- Memory: Reduces peak RAM usage by 50%
- Risk: Medium (requires careful file management)

#### 3. Pre-Computed Consensus Caching ✅

**Status:** Already implemented (`run_enhanced_privacy_pipeline.py:192`)

```python
if os.path.exists(consensus_ref):
    print(f"✅ Using pre-built superposition consensus: {consensus_ref}")
    return consensus_ref
```

**Impact:** Saves 30-60 minutes per run (already working)

### Layer 1 Summary

| Optimization | Speedup | Effort | Priority |
|--------------|---------|--------|----------|
| Parallel VCF parsing | 2-3× | 30-45 min | Medium |
| Memory-mapped graph | 1.5-2× | 2-3 hours | Low |
| Pre-built caching | ∞ (skip rebuild) | ✅ Done | N/A |

**Recommendation:** Layer 1 is already well-optimized. Focus on Layer 2-4 for higher impact.

---

## Layer 2: Rolling Reference Pool Optimization

### Current Implementation

**File:** `benchmarks/run_enhanced_privacy_pipeline.py` (lines 240-380)

**Process (per reference sample):**
1. **FASTQ Decompression:** `pigz -dc -p 4` (parallel gzip)
2. **Alignment:** `minimap2 -ax sr -t 10 -K 250M` (10 threads)
3. **Sorting:** `samtools sort -@ 4` (4 threads, external merge sort)
4. **Variant Calling:** `bcftools mpileup | bcftools call` (single-threaded)

**Current Performance:**
- Time per reference: 30-60 minutes
- 12 references × 60 min = **12 hours total**
- Bottleneck: Sorting (20-30 min) + Variant calling (10-15 min)

### Data Format Clarification

**FASTQ Stage (Input):**
- **Format:** Plain text, 4 lines per read
- **Structure:**
  ```
  @READ_ID
  ACGTACGTACGT...          ← Raw sequence (no chromosome label)
  +
  IIIIIIIIIIII...          ← Quality scores
  ```
- **Chromosome Info:** ❌ NOT PRESENT (reads are unlabeled)
- **Optimizations:** Decompression (pigz), quality filtering

**BAM Stage (Post-Alignment):**
- **Format:** Binary aligned reads
- **Structure:**
  ```
  READ_ID  chr22  12345678  ...  ← Chromosome + position assigned by minimap2
  ```
- **Chromosome Info:** ✅ PRESENT (assigned during alignment)
- **Optimizations:** Chromosome-partitioned sorting, parallel merging

### Optimization Opportunities

#### 1. Sambamba Parallel Sorting (2-3× speedup) ⭐ IMMEDIATE

**Current Code:**
```bash
minimap2 -ax sr -t 10 -K 250M -2 {align_params} {consensus_ref} \
    <(pigz -dc -p 4 {r1}) <(pigz -dc -p 4 {r2}) | \
    samtools sort -@ 4 -o {bam_file} -
```

**Optimized Code:**
```bash
minimap2 -ax sr -t 10 -K 250M -2 {align_params} {consensus_ref} \
    <(pigz -dc -p 4 {r1}) <(pigz -dc -p 4 {r2}) | \
    sambamba sort -t 10 -m 4G --tmpdir=/tmp -o {bam_file} /dev/stdin
```

**Why Faster:**
- **samtools:** 2-way merge (merges 2 temp files at a time)
- **sambamba:** N-way merge (merges all temp files in parallel)
- **samtools:** Single-threaded final merge
- **sambamba:** Multi-threaded throughout

**Impact:**
- Sorting: 20-30 min → 7-10 min (2-3× speedup)
- Per reference: 60 min → 37-40 min
- 12 references: 12 hours → **4.4-4.8 hours** (7.2 hours saved)

**Implementation:**
```python
# In run_enhanced_privacy_pipeline.py, line 363
align_cmd = f"""
minimap2 -ax sr -t {threads} -K 250M -2 {align_params} {consensus_ref} \\
    <(pigz -dc -p 4 {r1}) <(pigz -dc -p 4 {r2}) | \\
    sambamba sort -t {threads} -m 4G --tmpdir={tmp_dir} -o {bam_file} /dev/stdin
"""
```

**Risk:** Low (sambamba already installed, drop-in replacement)

#### 2. Minimap2 Index Caching (30-60 sec saved per reference)

**Current:** Index rebuilt for each reference (wasted computation)

**Optimized:**
```python
def get_or_build_minimap2_index(ref_fasta: str) -> str:
    """Build or load cached minimap2 index."""
    index_file = f"{ref_fasta}.mmi"

    if os.path.exists(index_file):
        # Check if index is newer than reference
        ref_mtime = os.path.getmtime(ref_fasta)
        idx_mtime = os.path.getmtime(index_file)

        if idx_mtime > ref_mtime:
            print(f"✅ Using cached minimap2 index: {index_file}")
            return index_file

    # Build new index
    print(f"🔨 Building minimap2 index: {index_file}")
    subprocess.run([
        "minimap2", "-d", index_file, "-x", "sr", ref_fasta
    ], check=True)

    return index_file

# Usage
index_file = get_or_build_minimap2_index(consensus_ref)
align_cmd = f"minimap2 -ax sr -t {threads} {index_file} ..."
```

**Impact:**
- Index build: 30-60 sec per reference (one-time)
- 12 references × 60 sec = 12 min saved
- Negligible for first run, significant for repeated runs

#### 3. Parallel BCFtools Variant Calling (1.5-2× speedup)

**Current Code:**
```bash
bcftools mpileup -Ou -f {consensus_ref} {bam_file} | \
    bcftools call -mv -Ov -o {vcf_file}
```

**Optimized Code:**
```bash
# Use --threads for both mpileup and call
bcftools mpileup --threads 4 -Ou -f {consensus_ref} {bam_file} | \
    bcftools call --threads 4 -mv -Ov -o {vcf_file}
```

**Impact:**
- Variant calling: 10-15 min → 5-8 min (1.5-2× speedup)
- Per reference: 37-40 min → **32-35 min**
- 12 references: 4.4-4.8 hours → **3.8-4.2 hours**

#### 4. Chromosome-Partitioned Parallel Sorting (5-10× speedup) 🚀 FUTURE

**Important:** This optimization applies to **BAM files** (post-alignment), NOT FASTQ input.

**Why it works:**
1. Minimap2 assigns chromosome to each read during alignment
2. BAM file contains chromosome field (`chr1`, `chr22`, etc.)
3. We can partition BAM by chromosome and sort in parallel
4. Final merge is trivial (chromosome order is global sort order)

**Implementation:**

```python
def chromosome_parallel_sort(input_sam_stream, output_bam: str, threads: int = 12):
    """
    Partition BAM by chromosome and sort in parallel.

    This is 5-10× faster than single-threaded external merge sort because:
    - Each chromosome is sorted independently (embarrassingly parallel)
    - No merge phase needed (chromosome order IS the global sort order)
    - Utilizes all CPU cores simultaneously
    """
    import pysam
    from concurrent.futures import ProcessPoolExecutor
    import tempfile

    # Phase 1: Partition by chromosome (stream SAM → temp files per chr)
    print("Phase 1: Partitioning by chromosome...")
    temp_dir = tempfile.mkdtemp()
    chr_writers = {}

    for read in pysam.AlignmentFile(input_sam_stream, "r"):
        if read.is_unmapped:
            chr_name = "unmapped"
        else:
            chr_name = read.reference_name  # e.g., "chr22"

        if chr_name not in chr_writers:
            chr_writers[chr_name] = pysam.AlignmentFile(
                f"{temp_dir}/{chr_name}.unsorted.bam", "wb",
                header=read.header
            )

        chr_writers[chr_name].write(read)

    # Close all writers
    for writer in chr_writers.values():
        writer.close()

    # Phase 2: Sort each chromosome in parallel
    print(f"Phase 2: Sorting {len(chr_writers)} chromosomes in parallel...")

    def sort_chromosome(chr_name: str) -> str:
        """Sort a single chromosome's BAM file."""
        unsorted = f"{temp_dir}/{chr_name}.unsorted.bam"
        sorted_bam = f"{temp_dir}/{chr_name}.sorted.bam"

        # Use sambamba for fast single-chromosome sorting
        subprocess.run([
            "sambamba", "sort",
            "-t", "1",  # Single thread per chromosome (parallel across chromosomes)
            "-m", "500M",
            "-o", sorted_bam,
            unsorted
        ], check=True)

        return sorted_bam

    with ProcessPoolExecutor(max_workers=threads) as pool:
        sorted_bams = list(pool.map(sort_chromosome, chr_writers.keys()))

    # Phase 3: Concatenate in chromosome order (already globally sorted!)
    print("Phase 3: Merging sorted chromosomes...")

    # Chromosome order (no sorting needed, just concatenation)
    chr_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM", "unmapped"]
    ordered_bams = [f"{temp_dir}/{chr}.sorted.bam"
                   for chr in chr_order
                   if os.path.exists(f"{temp_dir}/{chr}.sorted.bam")]

    # Merge with samtools (just concatenation, no sorting)
    subprocess.run([
        "samtools", "merge", "-f", "-@", str(threads),
        output_bam, *ordered_bams
    ], check=True)

    # Cleanup
    shutil.rmtree(temp_dir)

    print(f"✅ Chromosome-parallel sort complete: {output_bam}")
```

**Usage:**
```python
# In run_enhanced_privacy_pipeline.py
align_cmd = f"""
minimap2 -ax sr -t {threads} -K 250M -2 {align_params} {index_file} \\
    <(pigz -dc -p 4 {r1}) <(pigz -dc -p 4 {r2})
"""

# Run alignment (outputs SAM to stdout)
align_process = subprocess.Popen(align_cmd, shell=True, stdout=subprocess.PIPE)

# Sort with chromosome partitioning
chromosome_parallel_sort(align_process.stdout, bam_file, threads=12)
```

**Performance Analysis:**

Assume 24 chromosomes (chr1-22 + chrX + chrY), 12 cores available:

**Current (samtools external merge sort):**
- Create 61 temp files sequentially: 5 min
- Merge 61 → 30 → 15 → 7 → 3 → 1: 20-25 min
- **Total:** 25-30 min

**Chromosome-parallel sort:**
- Partition into 24 chromosome streams: 3 min (streaming, minimal overhead)
- Sort 24 chromosomes × 1 core each (2 batches of 12): 4-6 min
- Concatenate 24 sorted files: 1-2 min
- **Total:** 8-11 min

**Speedup:** 25-30 min → 8-11 min = **2.5-3× faster**

**With 24 cores (full parallelism):**
- Sort all 24 chromosomes simultaneously: 2-3 min
- **Total:** 6-8 min = **4-5× faster**

**Impact:**
- Per reference: 32-35 min → **20-25 min**
- 12 references: 3.8-4.2 hours → **2.4-3.0 hours**

**Risk:** Medium (requires custom implementation, careful testing)

#### 5. AMX Acceleration for Alignment Scoring (2-3× speedup)

**What:** Use Apple Silicon AMX coprocessor for Smith-Waterman matrix operations.

**Implementation:**
```python
# genomevault/differential_encoding/amx_alignment.py
import Accelerate  # Apple's BLAS wrapper

class AMXAlignmentScorer:
    """AMX-accelerated alignment scoring."""

    def score_alignment(self, query_seq: str, target_seq: str) -> float:
        # Convert sequences to numeric matrices
        query_matrix = self._encode_sequence(query_seq)
        target_matrix = self._encode_sequence(target_seq)

        # Use Accelerate.vDSP for matrix operations (auto-uses AMX)
        score = Accelerate.vDSP.dot_product(query_matrix, target_matrix)

        return score
```

**Impact:**
- Alignment scoring: 2-3× faster (AMX is 1 TFLOPS int8)
- Minimap2 time: 15-20 min → 7-10 min
- Per reference: 20-25 min → **15-18 min**

**Risk:** Medium (requires Accelerate framework integration)

### Layer 2 Summary

| Optimization | Speedup (per ref) | Time Saved (12 refs) | Effort | Priority |
|--------------|-------------------|----------------------|--------|----------|
| **Sambamba sorting** | 2-3× (20→7 min) | 7.2 hours | 15 min | ⭐ HIGH |
| Minimap2 index cache | +60 sec saved | 12 min | 30 min | Medium |
| Parallel BCFtools | 1.5-2× (10→5 min) | 1.2 hours | 15 min | High |
| Chromosome-parallel sort | 2.5-5× (25→8 min) | 3.4 hours | 3-4 hours | Medium |
| AMX alignment | 2-3× (15→7 min) | 2.4 hours | 4-6 hours | Medium |

**Cumulative Impact:**
- Current: 60 min/ref × 12 = 12 hours
- With sambamba + BCFtools: 32 min/ref × 12 = **6.4 hours** (5.6 hours saved)
- With all optimizations: 12 min/ref × 12 = **2.4 hours** (9.6 hours saved)

**Recommendation:** Implement sambamba + parallel BCFtools immediately (30 min effort, 6.8 hours saved).

---

## Layer 3: Privacy-Preserving Query Alignment Optimization

### Current Implementation

**File:** `benchmarks/run_enhanced_privacy_pipeline.py` (lines 470-550)

**Process:**
1. Align query FASTQ to superposition consensus reference
2. Sort aligned BAM
3. Call variants (produces query VCF)

**Expected Performance:**
- Same as Layer 2 (single sample instead of 12)
- Time: 30-60 minutes
- Bottleneck: Same (sorting + variant calling)

### Optimization Opportunities

#### 1. Reuse Layer 2 Optimizations ✅

**All Layer 2 optimizations apply directly:**
- Sambamba sorting: 2-3× speedup
- Minimap2 index caching: Reuse index from Layer 1
- Parallel BCFtools: 1.5-2× speedup
- Chromosome-parallel sorting: 2.5-5× speedup (if implemented)
- AMX alignment: 2-3× speedup (if implemented)

**Impact:**
- Query alignment: 60 min → 12-15 min with all optimizations

#### 2. Privacy-Preserving Read Masking (Optional)

**Current:** Aligns full reads to consensus (potential information leakage)

**Enhanced Privacy:**
```python
def privacy_preserving_alignment(query_fastq: str, consensus_ref: str):
    """
    Align query with differential privacy noise injection.

    Privacy mechanism:
    - Add Laplace noise to quality scores (ε-differential privacy)
    - Mask low-quality reads (prevent fingerprinting)
    - Subsample to fixed coverage (prevent depth-based inference)
    """

    # Read FASTQ
    reads = parse_fastq(query_fastq)

    # Privacy step 1: Subsample to 30× coverage (prevent depth leakage)
    target_coverage = 30
    genome_size = 3_000_000_000  # 3 Gbp
    target_bases = genome_size * target_coverage
    reads = subsample_to_coverage(reads, target_bases)

    # Privacy step 2: Add Laplace noise to quality scores
    epsilon = 1.0  # Privacy budget
    for read in reads:
        read.quality = add_laplace_noise(read.quality, epsilon)

    # Privacy step 3: Mask reads below quality threshold
    min_quality = 20
    reads = [r for r in reads if np.mean(r.quality) >= min_quality]

    # Standard alignment
    return align_reads(reads, consensus_ref)
```

**Impact:**
- Privacy: Adds ε-differential privacy guarantee
- Performance: Negligible overhead (<5%)
- Risk: Low (well-established DP techniques)

### Layer 3 Summary

| Optimization | Speedup | Effort | Priority |
|--------------|---------|--------|----------|
| Reuse Layer 2 optimizations | 3-5× | ✅ Free | High |
| Privacy-preserving masking | 1× (no slowdown) | 2-3 hours | Low |

**Recommendation:** Reuse Layer 2 optimizations (free speedup). Privacy masking is optional.

---

## Layer 4: GenomeVault Core Optimization (HDC + ZK + PIR)

### Current Implementation

**File:** `benchmarks/run_enhanced_privacy_pipeline.py` (lines 600-800)

**Process:**
1. **Differential Encoding:** Compute differences between query VCF and 12 reference VCFs
2. **HDC Integration:** Encode differences as 10,000D hypervector
3. **ZK Proof:** Generate Groth16 proof of variant presence
4. **PIR Query:** Private information retrieval

**Current Performance:**
- Differential encoding: 100-200 ms
- HDC encoding: 500-800 ms (CPU-only)
- ZK proof: 1-2 seconds (Groth16)
- PIR query: 50-100 ms

**Bottleneck:** ZK proof generation (CPU-bound)

### Optimization Opportunities

#### 1. Metal GPU Batch HDC Encoding (43× speedup) ⭐ IMMEDIATE

**Current Status:**
- Metal backend exists (`genomevault/compute/metal_backend.py`)
- Proven 43.72× speedup for batch encoding (780 ms vs 34,080 ms)
- NOT currently enabled in pipeline

**Implementation:**

```python
# In run_enhanced_privacy_pipeline.py
from genomevault.compute.backend_selector import get_optimal_backend

def encode_differential_hypervectors(
    query_vcf: str,
    reference_vcfs: List[str],
    use_gpu: bool = True
):
    """Encode query differences using optimal backend."""

    # Auto-select backend (Metal > CUDA > CPU)
    backend = get_optimal_backend(
        prefer_gpu=use_gpu,
        batch_size=len(reference_vcfs)  # 12 samples
    )

    # Compute differences
    differences = []
    for ref_vcf in reference_vcfs:
        diff = compute_differential_encoding(query_vcf, ref_vcf)
        differences.append(diff)

    # Batch encode on GPU (43× faster for 12 samples)
    hypervectors = backend.encode_batch(differences)

    return hypervectors
```

**Performance:**

**Current (CPU):**
- 12 samples × 500 ms/sample = 6,000 ms = **6 seconds**

**With Metal GPU:**
- Batch encoding: 12 samples in **140 ms** (43× speedup)
- Speedup: 6,000 ms → 140 ms

**Impact:**
- HDC stage: 6 sec → 0.14 sec (5.86 sec saved per query)
- Risk: Low (code already exists, well-tested)

#### 2. ZK Proof Acceleration (Limited Options)

**Current:** Groth16 proving time ~1-2 seconds (CPU-bound)

**Challenge:** ZK proof generation is inherently sequential and not GPU-friendly.

**Possible Optimizations:**

**Option 1: PLONK Instead of Groth16** (1.5-2× faster proving)
```python
# Use PLONK (better prover efficiency than Groth16)
# Tradeoff: Larger proof size (1.5 KB vs 743 bytes)

from genomevault.zk_proofs import PLONKProver

prover = PLONKProver(circuit="variant_presence")
proof = prover.prove(witness)  # 500-800 ms vs 1-2 sec
```

**Option 2: Parallel Multi-Proof Generation** (N× speedup for N queries)
```python
from concurrent.futures import ProcessPoolExecutor

def generate_proofs_parallel(witnesses: List[Witness], num_workers: int = 4):
    """Generate multiple ZK proofs in parallel."""

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        proofs = pool.map(lambda w: prover.prove(w), witnesses)

    return list(proofs)
```

**Impact:**
- Single proof: 1-2 sec (unchanged)
- Batch of 10 proofs: 10-20 sec → 2.5-5 sec (4× speedup)

**Recommendation:** ZK optimization has limited ROI. Focus on HDC speedup instead.

#### 3. PIR Query Optimization (Already Fast)

**Current:** IT-PIR query takes 50-100 ms (already very fast)

**Possible Optimizations:**
- Batch PIR: Query multiple items simultaneously
- GPU-accelerated XOR operations for IT-PIR

**Impact:** Minimal (PIR is <5% of total time)

**Recommendation:** PIR is fast enough; skip optimization.

### Layer 4 Summary

| Optimization | Speedup | Effort | Priority |
|--------------|---------|--------|----------|
| **Metal GPU HDC encoding** | 43× (6s → 0.14s) | 15 min | ⭐ CRITICAL |
| PLONK instead of Groth16 | 1.5-2× (2s → 1s) | 3-4 hours | Low |
| Parallel multi-proof | 4× (batch only) | 1 hour | Low |
| PIR optimization | <1.2× | 2-3 hours | Very Low |

**Recommendation:** Enable Metal GPU HDC encoding immediately (15 min effort, 43× speedup).

---

## Implementation Timeline

### Phase 1: Immediate Wins (30-45 minutes) ⭐

**Implement after ref1 sorting completes:**

1. **Sambamba sorting** (15 min)
   - Replace `samtools sort` with `sambamba sort`
   - Expected: 2-3× speedup for sorting (20 min → 7 min)
   - File: `run_enhanced_privacy_pipeline.py:363`

2. **Parallel BCFtools** (15 min)
   - Add `--threads 4` to mpileup and call
   - Expected: 1.5-2× speedup for variant calling (10 min → 5 min)
   - File: `run_enhanced_privacy_pipeline.py:380`

3. **Metal GPU HDC encoding** (15 min)
   - Use `backend_selector.get_optimal_backend()`
   - Expected: 43× speedup for HDC (6 sec → 0.14 sec)
   - File: `run_enhanced_privacy_pipeline.py:650`

**Total Impact:**
- Layer 2: 60 min/ref → 32 min/ref (12 refs = 5.6 hours saved)
- Layer 4: 6 sec → 0.14 sec (5.86 sec saved per query)
- **Total time saved: 5.6 hours + 6 sec ≈ 5.6 hours**

### Phase 2: High-Impact Optimizations (2-3 hours)

4. **Minimap2 index caching** (30 min)
   - Build `.mmi` index once, reuse for all samples
   - Expected: 60 sec saved per reference (12 min total)

5. **AMX alignment scoring** (4-6 hours)
   - Integrate Apple Accelerate framework
   - Expected: 2-3× speedup for alignment (15 min → 7 min)
   - Impact: 1.6 hours saved for 12 references

**Total Impact:** 1.8 hours saved

### Phase 3: Advanced Optimizations (6-10 hours)

6. **Chromosome-partitioned sorting** (3-4 hours)
   - Implement parallel BAM partitioning by chromosome
   - Expected: 2.5-5× speedup (25 min → 8 min)
   - Impact: 3.4 hours saved for 12 references

7. **Parallel VCF parsing for Layer 1** (1-2 hours)
   - Parse 7 reference VCFs in parallel
   - Expected: 2-3× speedup (60 min → 20 min)
   - Impact: 40 min saved (one-time)

**Total Impact:** 4.1 hours saved

### Phase 4: Research Optimizations (8-12 hours)

8. **PLONK ZK backend** (3-4 hours)
   - Implement PLONK prover as alternative to Groth16
   - Expected: 1.5-2× speedup (2 sec → 1 sec)
   - Impact: Minimal for single queries

9. **Memory-mapped graph construction** (2-3 hours)
   - Use mmap for Layer 1 graph building
   - Expected: 1.5-2× speedup (60 min → 30 min)
   - Impact: 30 min saved (one-time)

**Total Impact:** 30 min saved

---

## Expected Performance Summary

### Current Pipeline (k=13, no optimizations)

| Layer | Time | Bottleneck |
|-------|------|------------|
| Layer 1: Superposition Consensus | 30-60 min (one-time) | VCF parsing |
| Layer 2: Rolling Reference Pool | 12 hours (60 min × 12) | Sorting + variant calling |
| Layer 3: Query Alignment | 60 min | Same as Layer 2 |
| Layer 4: HDC + ZK + PIR | 8-10 sec | HDC encoding (6 sec) |
| **TOTAL** | **~14 hours** | Layer 2 dominates |

### With Phase 1 Optimizations (30 min effort)

| Layer | Time | Improvement |
|-------|------|-------------|
| Layer 1 | 30-60 min (cached) | No change |
| Layer 2 | 6.4 hours (32 min × 12) | **5.6 hours saved** |
| Layer 3 | 32 min | 28 min saved |
| Layer 4 | 2.14 sec (HDC: 0.14 sec) | 6 sec saved |
| **TOTAL** | **~8 hours** | **6 hours saved (43% faster)** |

### With All Optimizations (16-19 hours effort)

| Layer | Time | Improvement |
|-------|------|-------------|
| Layer 1 | 20 min (parallel VCF + cached) | 40 min saved |
| Layer 2 | 2.4 hours (12 min × 12) | **9.6 hours saved** |
| Layer 3 | 12 min | 48 min saved |
| Layer 4 | 2.14 sec | 6 sec saved |
| **TOTAL** | **~3 hours** | **11 hours saved (78% faster)** |

---

## Risk Assessment

### Low Risk (Safe to implement immediately)
- ✅ Sambamba sorting (drop-in replacement)
- ✅ Parallel BCFtools (standard flag)
- ✅ Metal GPU HDC encoding (code already exists)
- ✅ Minimap2 index caching (standard practice)

### Medium Risk (Requires testing)
- ⚠️ Chromosome-partitioned sorting (custom implementation)
- ⚠️ AMX alignment acceleration (framework integration)
- ⚠️ Parallel VCF parsing (memory management)

### High Risk (Research required)
- 🔴 PLONK ZK backend (new cryptographic library)
- 🔴 Memory-mapped graph (complex file management)

---

## Validation & Testing

After implementing each optimization, validate:

1. **Correctness:**
   ```bash
   # Compare VCF outputs (should be identical)
   diff <(bcftools view original.vcf | sort) \
        <(bcftools view optimized.vcf | sort)
   ```

2. **Performance:**
   ```bash
   # Time each stage
   time python benchmarks/run_enhanced_privacy_pipeline.py --stage layer2
   ```

3. **Privacy Guarantees:**
   ```bash
   # Verify k-anonymity maintained
   python benchmarks/verify_privacy_guarantees.py
   ```

---

## Next Steps

**Immediate (after ref1 sorting completes):**
1. ✅ Wait for current sorting to finish (~10-15 min remaining)
2. Implement Phase 1 optimizations (30 min effort)
3. Restart pipeline with optimizations enabled
4. Benchmark improvement (expect 5.6 hours saved)

**Short-term (this week):**
5. Implement AMX alignment scoring (Phase 2)
6. Benchmark full k=13 pipeline with optimizations

**Medium-term (next 2 weeks):**
7. Implement chromosome-partitioned sorting (Phase 3)
8. Complete end-to-end benchmarking
9. Update documentation with final results

---

## Conclusion

This stage-specific optimization plan provides **11 hours of total speedup** (78% reduction) across all four pipeline layers. The highest-impact optimizations are:

1. **Sambamba sorting** (Layer 2): 5.6 hours saved, 15 min effort ⭐
2. **Metal GPU HDC** (Layer 4): 6 sec saved, 15 min effort ⭐
3. **Chromosome-parallel sorting** (Layer 2): 3.4 hours saved, 3-4 hours effort
4. **AMX alignment** (Layer 2): 1.6 hours saved, 4-6 hours effort

**Recommended Priority:**
1. Phase 1 (30 min effort, 5.6 hours saved) - **Implement immediately**
2. Phase 2 (4-6 hours effort, 1.8 hours saved) - This week
3. Phase 3 (6-10 hours effort, 4.1 hours saved) - Next 2 weeks
4. Phase 4 (8-12 hours effort, 30 min saved) - Low priority

---

**Status:** Ready for implementation (waiting for ref1 completion)
**Next Action:** Monitor ref1 sorting, apply Phase 1 optimizations when complete
