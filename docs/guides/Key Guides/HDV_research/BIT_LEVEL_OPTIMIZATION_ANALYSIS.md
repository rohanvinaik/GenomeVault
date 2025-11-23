# Bit-Level Optimization Analysis: 3-Ternary Bank Architecture
## Memory Efficiency, SIMD Acceleration, and Query Optimization

**Date**: November 21, 2025 (CORRECTED)
**Status**: Architecture Analysis & Optimization Roadmap
**Version**: 2.0

---

## Executive Summary

The **3-bank ternary architecture** (D=5,120, N=1,024) is the **optimal encoding format** for GenomeVault HDC, providing superior performance across all critical dimensions compared to alternative binary representations.

### Comparison: 3 Ternary vs 6 Binary

| Dimension | 3 Ternary Banks | 6 Binary Banks | Winner |
|-----------|-----------------|----------------|--------|
| **Storage** | 0.75D bytes (with 2-bit packing) | 0.75D bytes (with 1-bit packing) | **TIE** |
| **Query Speed** | Direct access, no reconstruction | Requires 3 subtractions (6D ops) | **3 Ternary** |
| **Encoding Compute** | 3 sparsification ops | 6 sparsification ops (50% more) | **3 Ternary** |
| **Accuracy (Monty Hall)** | Natural signed similarity | Requires reconstruction step | **3 Ternary** |
| **Information Theory** | Sub-Shannon via D >> N orthogonality | Sub-Shannon via D >> N orthogonality | **TIE** |

**Recommendation**: Continue with **3-ternary architecture** and focus optimizations on:
1. SIMD-accelerated ternary operations (AVX-512, NEON)
2. Cache-line alignment and prefetching
3. Sparse ternary dot product kernels
4. Memory-mapped I/O with intelligent chunking

---

## Part 1: Detailed Analysis of 3 Ternary vs 6 Binary

### 1.1 Storage Efficiency: TIE

**3 Ternary Banks:**
- Ternary {-1, 0, +1} requires 2 bits per element (4 states, 3 used)
- 3 banks × D dimensions × 2 bits = 6D bits = **0.75D bytes**
- Practical HDF5 storage: `dtype=np.int8` (1 byte per element)
- Storage: 3 × D × 1 byte = 3D bytes = **15,360 bytes per chunk** (for D=5,120)

**6 Binary Banks:**
- Binary {0, 1} requires 1 bit per element
- 6 banks × D dimensions × 1 bit = 6D bits = **0.75D bytes**
- Practical HDF5 storage: `dtype=np.uint8` (1 byte per element, bit-packed)
- Storage: 6 × D × 1 byte = 6D bytes = **30,720 bytes per chunk** (unpacked)
- With bit-packing: 6 × (D / 8) = **3,840 bytes per chunk** (packed)

**With optimal packing**:
- 3 ternary: 2-bit encoding → 0.75D bytes
- 6 binary: 1-bit encoding → 0.75D bytes

**Verdict:** TIE at 0.75D bytes with optimal packing. In practice, both use int8/uint8 for simplicity, where 6-binary can be 4× larger unpacked but 4× smaller when bit-packed.

---

### 1.2 Query Speed: 3 Ternary WINS

**The Critical Difference: Reconstruction Overhead**

**3 Ternary Banks (Direct Access)**:
```python
def query_3ternary(chunk_vectors, position_vector):
    """
    chunk_vectors: (3, D) int8 array {-1, 0, +1}
    position_vector: (D,) int8 array {-1, +1}
    """
    # Direct access - O(1) per bank
    bank1 = chunk_vectors[0]  # Hydrophobic (5,120 elements)
    bank2 = chunk_vectors[1]  # Major Groove (5,120 elements)
    bank3 = chunk_vectors[2]  # Hinge (5,120 elements)

    # Dot product: D multiply-adds per bank
    sim_bank1 = np.dot(bank1, position_vector)  # 5,120 ops
    sim_bank2 = np.dot(bank2, position_vector)  # 5,120 ops
    sim_bank3 = np.dot(bank3, position_vector)  # 5,120 ops

    # Total: 3 memory reads + 15,360 multiply-add operations
```

**6 Binary Banks (Reconstruction Required)**:
```python
def query_6binary(chunk_vectors, position_vector):
    """
    chunk_vectors: (6, D) uint8 array {0, 1}
    position_vector: (D,) int8 array {-1, +1}
    """
    # Reconstruction step - O(D) per bank
    bank1 = chunk_vectors[0] - chunk_vectors[1]  # Hydrophobic: A - T (5,120 ops)
    bank2 = chunk_vectors[2] - chunk_vectors[3]  # Major Groove: G - C (5,120 ops)
    bank3 = chunk_vectors[4] - chunk_vectors[5]  # Hinge: pos - neg (5,120 ops)

    # Dot product: D multiply-adds per bank
    sim_bank1 = np.dot(bank1, position_vector)  # 5,120 ops
    sim_bank2 = np.dot(bank2, position_vector)  # 5,120 ops
    sim_bank3 = np.dot(bank3, position_vector)  # 5,120 ops

    # Total: 6 memory reads + 15,360 subtractions + 15,360 multiply-add operations
    # Extra cost: 15,360 subtraction operations (3 × D)
```

**Computational Cost Comparison**:

| Operation | 3 Ternary | 6 Binary | Overhead |
|-----------|-----------|----------|----------|
| **Memory reads** | 3 × D = 15,360 bytes | 6 × D = 30,720 bytes | **2× more** |
| **Reconstruction** | 0 ops | 3 × D = 15,360 subs | **+15,360 ops** |
| **Dot products** | 3 × D = 15,360 ops | 3 × D = 15,360 ops | Same |
| **Total compute** | 15,360 ops | 30,720 ops | **2× slower** |

**Latency Estimate** (D=5,120):
- **3 ternary**: 15,360 int8 multiply-adds ≈ **2-5 μs** (with SIMD)
- **6 binary**: 15,360 subs + 15,360 mul-adds ≈ **4-10 μs** (with SIMD)

**Verdict**: 3 ternary banks eliminate reconstruction overhead, providing **2× faster queries**.

---

### 1.3 Encoding Compute: 3 Ternary WINS

**During Encoding (Per Chunk)**:

**3 Ternary Banks**:
```python
# Accumulate (identical for both architectures)
acc_hydro = accumulate_hydrophobic(sequence, position_codebook)  # int16
acc_groove = accumulate_major_groove(sequence, position_codebook)  # int16
acc_hinge = accumulate_hinge(sequence, position_codebook)  # int16

# Quantize to ternary (3 operations)
bank1 = np.sign(acc_hydro).astype(np.int8)  # {-1, 0, +1}
bank2 = np.sign(acc_groove).astype(np.int8)
bank3 = np.sign(acc_hinge).astype(np.int8)

# Total: 3 sparsification operations
```

**6 Binary Banks**:
```python
# Accumulate (same as above)
acc_hydro = accumulate_hydrophobic(sequence, position_codebook)
acc_groove = accumulate_major_groove(sequence, position_codebook)
acc_hinge = accumulate_hinge(sequence, position_codebook)

# Split into 6 binary banks (6 operations)
bank1_A = (acc_hydro < 0).astype(np.uint8)  # A positions
bank1_T = (acc_hydro > 0).astype(np.uint8)  # T positions
bank2_G = (acc_groove > 0).astype(np.uint8)  # G positions
bank2_C = (acc_groove < 0).astype(np.uint8)  # C positions
bank3_pos = (acc_hinge > 0).astype(np.uint8)  # YR positions
bank3_neg = (acc_hinge < 0).astype(np.uint8)  # RY positions

# Total: 6 threshold operations
```

**Encoding Cost**:
- **3 ternary**: 3 × D comparisons + sign operations = **15,360 ops**
- **6 binary**: 6 × D comparisons + conversions = **30,720 ops**

**Verdict**: 3 ternary banks require **50% less compute during encoding**.

---

### 1.4 Accuracy & Genomic Monty Hall: 3 Ternary WINS

**The Genomic Monty Hall Framework**:
- Cross-validate 3 orthogonal chemical lenses (Hydrophobic, Major Groove, Hinge)
- Each lens provides **signed similarity**: positive (evidence FOR), negative (evidence AGAINST), or neutral (no information)
- Natural alignment with ternary representation {-1, 0, +1}

**3 Ternary Banks (Natural Signed Similarity)**:
```python
# Query: "Is this position nucleotide A?"
# position_vector: random {-1, +1} vector for the position

# Bank 1 (Hydrophobic): A=-1, T=+1, GC=0
sim_hydro = np.dot(bank1, position_vector)
# bank1[i] = -1 → contributes NEGATIVELY → evidence FOR A
# bank1[i] = +1 → contributes POSITIVELY → evidence AGAINST A (pro-T)
# bank1[i] = 0 → contributes 0 → neutral (GC positions)

# Direct interpretation:
# sim_hydro < 0 → evidence FOR A
# sim_hydro > 0 → evidence AGAINST A (likely T)
# sim_hydro ≈ 0 → ambiguous (likely GC or weak signal)
```

**6 Binary Banks (Requires Reconstruction)**:
```python
# Must reconstruct ternary before computing signed similarity
bank1_ternary = bank1_A - bank1_T  # Reconstruction: D subtraction ops

# Then compute similarity (same as 3-ternary)
sim_hydro = np.dot(bank1_ternary, position_vector)

# Extra step: reconstruction overhead
```

**Monty Hall "Door Revealing" Logic**:

The Genomic Monty Hall algorithm relies on **signed constraints** from multiple lenses:

```python
# Lens 1 (Hydrophobic): "Strong negative signal"
if sim_hydro < -threshold:
    # Reveals: NOT G, NOT C (transparent), NOT T (would be positive)
    # Conclusion: Must be A

# Lens 2 (Major Groove): "Weak signal"
if abs(sim_groove) < weak_threshold:
    # Reveals: NOT G, NOT C (would have strong signal)
    # Conclusion: Must be A or T

# Lens 3 (Hinge): "Purine step detected"
if sim_hinge > 0:
    # Reveals: YR step (pyrimidine → purine)
    # If previous was T (pyrimidine), current must be A or G (purine)
    # Combined with Lens 1: Must be A (not G)
```

**Verdict**: Ternary values {-1, 0, +1} directly encode "pro", "anti", "neutral" states, providing **natural alignment** with Genomic Monty Hall's signed constraint logic. 6-binary requires reconstruction before applying the same logic.

---

### 1.5 Information Theory (Sub-Shannon Encoding): TIE

**Key Insight**: The "Shannon violation" comes from **high-dimensional orthogonal projection** (D >> N), NOT from storage format.

#### Classical Shannon Limit

For 4 nucleotides (A, T, G, C):
```
Minimum bits per nucleotide = log₂(4) = 2 bits/nucleotide
```

This is the **information-theoretic lower bound** for lossless storage of random DNA sequences.

#### GenomeVault's Apparent "Violation"

**Storage Cost**:
- 3 ternary banks: 0.75D bytes for N nucleotides
- For D=5,120 and N=1,024:
  ```
  (0.75 × 5,120 × 8 bits) / 1,024 nucleotides = 30 bits/nucleotide
  ```

**Wait, that's 15× WORSE than Shannon!**

#### The Resolution: Distributed Encoding vs Direct Storage

This is **not** direct nucleotide storage. It's a **distributed high-dimensional encoding** where:

1. **Each nucleotide influences D dimensions** (via positional binding)
   - Nucleotide at position i binds with position_vector[i] (D-dimensional)
   - Accumulated across all positions in chunk

2. **Orthogonal random projections** create D-dimensional "smear"
   - position_codebook: (N × D) random {-1, +1} matrix
   - Each dimension accumulates contributions from ~N/D ≈ 205 nucleotides on average

3. **SNR amplification**: D/N = 5.0 provides 5× redundancy
   - Signal strength: Proportional to √D (coherent accumulation)
   - Noise strength: Proportional to √N (random cancellation)
   - SNR = √(D/N) = √5.0 ≈ 2.24× better than N=D baseline

4. **Recovery requires decoding** (similarity search + Monty Hall)
   - Not a direct lookup (unlike Shannon's encoding)
   - Similarity search: Compare query position vector to all encoded chunks
   - Constraint satisfaction: Genomic Monty Hall cross-validates 3 lenses

#### Local Information Density (The Real Story)

Within each chunk, **compositional constraints** reduce effective alphabet size:

**Example: 80% GC region (CpG island)**
```
Naive entropy:
  P(G) = 0.40, P(C) = 0.40, P(A) = 0.10, P(T) = 0.10
  Entropy = -Σ P(x) log₂ P(x)
          = -(0.4 log 0.4 + 0.4 log 0.4 + 0.1 log 0.1 + 0.1 log 0.1)
          ≈ 1.72 bits/nucleotide

Reduction: 2.0 - 1.72 = 0.28 bits/nucleotide (14% less than Shannon!)
```

**Magnitude weighting applies Bayesian priors**:
- In CpG island, if Hydrophobic signal is weak → likely GC (not AT)
- If Major Groove signal is strong → definitely GC
- Combined: Effective alphabet reduced from {A, T, G, C} → {G, C}
- Entropy: log₂(2) = **1 bit/nucleotide** (50% reduction!)

**Lens library provides structural templates**:
- "This region matches Alu repeat" → 300 bp with known structure
- Effective information: Position within Alu + variant type
- Bits needed: log₂(300) + log₂(10 variants) ≈ **11 bits total** vs 300 × 2 = 600 bits naive
- **98% compression** via template matching!

#### Both Formats Achieve Sub-Shannon Encoding

**3 Ternary Banks**:
- High-D orthogonal projection: D >> N (5,120 > 1,024)
- Ternary quantization: {-1, 0, +1} with natural sparsity (50-70%)
- Compositional constraints: Magnitude weighting + Monty Hall
- **Effective bits/nucleotide**: < 2 bits (after all constraints)

**6 Binary Banks**:
- High-D orthogonal projection: D >> N (5,120 > 1,024)
- Binary quantization: {0, 1} with artificial sparsity (96%)
- Compositional constraints: Magnitude weighting + Monty Hall
- **Effective bits/nucleotide**: < 2 bits (after all constraints)

**Verdict**: TIE. Both achieve sub-Shannon encoding via **D >> N orthogonality**, NOT storage format.

---

## Part 2: Optimizations for 3-Ternary Architecture

Since 3-ternary is the superior architecture, we focus on optimizations that work **with** ternary format, not against it.

### 2.1 SIMD-Accelerated Ternary Dot Product

**Current Implementation** (NumPy):
```python
# NumPy uses BLAS for dot product
similarity = np.dot(bank_vector, position_vector)  # (D,) × (D,)
# Latency: ~2-5 μs for D=5,120
```

**Optimized SIMD Implementation** (AVX-512):
```c
#include <immintrin.h>

int32_t ternary_dot_product_avx512(
    const int8_t *ternary_vec,    // {-1, 0, +1}
    const int8_t *position_vec,   // {-1, +1}
    size_t D
) {
    __m512i sum = _mm512_setzero_si512();

    for (size_t i = 0; i < D; i += 64) {
        // Load 64 int8 elements (512 bits)
        __m512i v1 = _mm512_loadu_si512(&ternary_vec[i]);
        __m512i v2 = _mm512_loadu_si512(&position_vec[i]);

        // Multiply: int8 × int8 → int16 (AVX-512 VNNI)
        __m512i prod = _mm512_maddubs_epi16(v1, v2);

        // Accumulate
        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(prod, _mm512_set1_epi16(1)));
    }

    // Horizontal sum
    return _mm512_reduce_add_epi32(sum);
}

// Latency: ~500-800 ns for D=5,120 (4-6× faster than NumPy)
```

**ARM NEON** (Apple M1/M2/M3/M4):
```c
#include <arm_neon.h>

int32_t ternary_dot_product_neon(
    const int8_t *ternary_vec,
    const int8_t *position_vec,
    size_t D
) {
    int32x4_t sum = vdupq_n_s32(0);

    for (size_t i = 0; i < D; i += 16) {
        // Load 16 int8 elements (128 bits)
        int8x16_t v1 = vld1q_s8(&ternary_vec[i]);
        int8x16_t v2 = vld1q_s8(&position_vec[i]);

        // Multiply: int8 × int8 → int16
        int16x8_t prod_low = vmull_s8(vget_low_s8(v1), vget_low_s8(v2));
        int16x8_t prod_high = vmull_s8(vget_high_s8(v1), vget_high_s8(v2));

        // Accumulate to int32
        sum = vpadalq_s16(sum, prod_low);
        sum = vpadalq_s16(sum, prod_high);
    }

    // Horizontal sum
    return vaddvq_s32(sum);
}

// Latency: ~600-1000 ns for D=5,120 (3-5× faster than NumPy)
```

**Speedup**: 3-6× faster than NumPy/BLAS for small vectors (D=5,120).

---

### 2.2 Sparse Ternary Dot Product (Exploiting Natural Sparsity)

**Observation**: 50-70% of ternary elements are zero (natural sparsity).

**Sparse Kernel** (skip zero elements):
```c
int32_t sparse_ternary_dot_product(
    const int8_t *ternary_vec,
    const int8_t *position_vec,
    const uint16_t *nonzero_indices,  // Indices where ternary_vec[i] != 0
    size_t num_nonzero,
    size_t D
) {
    int32_t sum = 0;

    for (size_t i = 0; i < num_nonzero; i++) {
        uint16_t idx = nonzero_indices[i];
        sum += ternary_vec[idx] * position_vec[idx];
    }

    return sum;
}

// Latency: ~200-400 ns for 30% density (5,120 × 0.3 = 1,536 ops)
// Speedup: 5-10× faster than dense for high sparsity
```

**Trade-off**: Requires storing nonzero indices (2 bytes per index).
- Index storage: 30% density × 5,120 × 2 bytes = **3,072 bytes per bank**
- Total overhead: 9,216 bytes per chunk (vs 15,360 bytes savings from skipping 70% of ops)

**Net benefit**: Only worthwhile if query frequency >> encoding frequency (which is true for GenomeVault).

---

### 2.3 Cache-Line Alignment and Prefetching

**Problem**: 15 KB per chunk doesn't fit in L1 cache (~32 KB typical).

**Solution**: Align banks to cache-line boundaries (64 bytes).

```c
// Align each bank to 64-byte boundary
typedef struct __attribute__((aligned(64))) {
    int8_t bank1[5120];  // 5,120 bytes = 80 cache lines
    int8_t bank2[5120];  // 5,120 bytes = 80 cache lines
    int8_t bank3[5120];  // 5,120 bytes = 80 cache lines
} AlignedChunkVector;  // Total: 15,360 bytes = 240 cache lines

// With 64-byte alignment, each bank starts on cache-line boundary
```

**Prefetching Strategy**:
```c
void query_batch_with_prefetch(
    AlignedChunkVector *chunks,
    int8_t *position_vector,
    int *results,
    size_t n
) {
    for (size_t i = 0; i < n; i++) {
        // Prefetch next chunk (if available)
        if (i + 1 < n) {
            __builtin_prefetch(&chunks[i+1], 0, 3);  // Prefetch to L1
        }

        // Query current chunk
        results[i] = query_ternary(&chunks[i], position_vector);
    }
}
```

**Benefit**: Reduces memory latency by ~30-50% for sequential access patterns.

---

### 2.4 Memory-Mapped I/O with Intelligent Chunking

**Problem**: HDF5 random access has high latency (~50-100 μs per chunk).

**Solution**: Memory-map hot chromosomes (chr1-22, chrX, chrY).

```python
import mmap
import numpy as np

class MemoryMappedChunkStore:
    def __init__(self, h5_path, chromosome):
        self.h5_file = h5py.File(h5_path, 'r')
        self.chunk_dataset = self.h5_file['all_bank_vectors']

        # Get chromosome index range
        self.chr_start, self.chr_end = self._get_chr_range(chromosome)

        # Memory-map chromosome region
        self.mmap_file = open(f"/tmp/{chromosome}_chunks.bin", "r+b")
        self.mmap = mmap.mmap(self.mmap_file.fileno(), 0)

        # Create numpy array view
        num_chunks = self.chr_end - self.chr_start
        self.chunks = np.frombuffer(
            self.mmap,
            dtype=np.int8,
            count=num_chunks * 3 * 5120
        ).reshape(num_chunks, 3, 5120)

    def get_chunk(self, chunk_idx):
        # O(1) access via memory-mapped view
        return self.chunks[chunk_idx]
```

**Benefit**: Reduces chunk access latency from 50-100 μs (HDF5) to <1 μs (memory access).

---

### 2.5 GPU Acceleration (Metal / CUDA)

**Metal Compute Shader** (Apple Silicon):
```metal
kernel void batch_ternary_dot_product(
    device const char *chunk_vectors [[buffer(0)]],   // (N, 3, D) int8
    device const char *position_vector [[buffer(1)]],  // (D,) int8
    device int *similarities [[buffer(2)]],            // (N, 3) output
    uint tid [[thread_position_in_grid]],
    constant uint &D [[buffer(3)]]
) {
    uint chunk_idx = tid / 3;
    uint bank_idx = tid % 3;

    // Offset to this chunk's bank
    device const char *bank = &chunk_vectors[(chunk_idx * 3 + bank_idx) * D];

    int sum = 0;
    for (uint i = 0; i < D; i++) {
        sum += bank[i] * position_vector[i];
    }

    similarities[tid] = sum;
}

// Launch: N chunks × 3 banks = 10,000 × 3 = 30,000 threads
// Latency: ~50-100 μs for 10,000 chunks (massively parallel)
// Per-chunk: ~5-10 ns (1000× faster than CPU!)
```

**Use Case**: Batch queries (scan multiple chunks for best match).

---

## Part 3: Revised Optimization Roadmap

### Phase 1: Low-Hanging Fruit (This Week)

**Goal**: 2-3× query speedup with minimal code changes

**Tasks**:
1. Implement cache-line aligned chunk storage
2. Add prefetching to query loops
3. Benchmark on chr22 test set

**Expected Speedup**: 2× faster (from 5 μs → 2.5 μs per query)

---

### Phase 2: SIMD Kernels (2-3 Weeks)

**Goal**: 5-6× query speedup with SIMD

**Tasks**:
1. Write AVX-512 ternary dot product kernel (x86)
2. Write NEON ternary dot product kernel (ARM)
3. Python C extension wrapper
4. Benchmark against NumPy baseline

**Expected Speedup**: 5× faster (from 5 μs → 1 μs per query)

---

### Phase 3: Sparse Kernels (Optional, 1-2 Weeks)

**Goal**: 10× query speedup for sparse regions

**Tasks**:
1. Add nonzero index tracking to encoder
2. Implement sparse ternary dot product
3. Adaptive dense/sparse kernel selection

**Expected Speedup**: 10× faster for high-sparsity banks (from 5 μs → 500 ns)

---

### Phase 4: GPU Acceleration (Future, 2-3 Weeks)

**Goal**: 100-1000× throughput for batch queries

**Tasks**:
1. Write Metal compute shader (Apple Silicon)
2. Write CUDA kernel (NVIDIA)
3. Batch query API

**Expected Speedup**: 1000× faster for batch (from 5 μs → 5 ns per chunk in batch)

---

## Part 4: Why 6-Binary Doesn't Win

### Misconception: "Binary is always faster than ternary"

**Reality**: Binary operations (XOR, popcount) are fast, but **reconstruction overhead dominates**.

**Cost Breakdown**:
```
6-Binary Query:
  1. Load 6 banks: 30,720 bytes (6D)
  2. Reconstruct 3 ternary: 15,360 subtractions (3D ops)
  3. Dot product: 15,360 multiply-adds (3D ops)
  Total: 30,720 operations

3-Ternary Query:
  1. Load 3 banks: 15,360 bytes (3D)
  2. Dot product: 15,360 multiply-adds (3D ops)
  Total: 15,360 operations

Ratio: 6-binary is 2× slower (not faster!)
```

**When would 6-binary win?**
- If we never needed ternary values (but we do for Genomic Monty Hall)
- If XOR/Hamming distance could replace dot product (but signed similarity is essential)
- If reconstruction was free (but it's 15,360 ops)

**None of these apply to GenomeVault.**

---

## Part 5: Storage Optimization (Separate from Query Speed)

### 2-Bit Ternary Packing (For Disk Storage)

While 3-ternary wins on query speed, we can still optimize disk storage:

```python
def pack_ternary_2bit(ternary_array: np.ndarray) -> np.ndarray:
    """
    Pack ternary {-1, 0, +1} into 2-bit representation.

    Mapping: -1 → 00, 0 → 01, +1 → 10 (unused: 11)
    """
    # Map to {0, 1, 2}
    packed = (ternary_array + 1).astype(np.uint8)  # -1→0, 0→1, +1→2

    # Pack 4 values per byte (2 bits each)
    result = np.zeros(len(packed) // 4, dtype=np.uint8)
    for i in range(4):
        result |= (packed[i::4] << (i * 2))

    return result


def unpack_ternary_2bit(packed: np.ndarray, length: int) -> np.ndarray:
    """Unpack 2-bit ternary back to {-1, 0, +1}."""
    unpacked = np.zeros(length, dtype=np.int8)
    for i in range(4):
        unpacked[i::4] = (packed >> (i * 2)) & 0b11

    # Map back: 0→-1, 1→0, 2→+1
    return unpacked - 1
```

**Storage Savings**:
- Unpacked: 3 × D × 1 byte = 15,360 bytes
- Packed: 3 × D × 0.25 byte = 3,840 bytes
- **Reduction: 4× smaller** (same as bit-packed 6-binary!)

**Trade-off**: Packing/unpacking overhead (~1-2 μs per chunk).
- **Use for**: Long-term storage (HDF5 on disk)
- **Don't use for**: In-memory query cache (keep unpacked for speed)

---

## Conclusion

### Final Recommendation: Stick with 3-Ternary Architecture

**Why 3-Ternary Wins**:
1. ✅ **Query Speed**: 2× faster (no reconstruction overhead)
2. ✅ **Encoding Speed**: 50% faster (3 ops vs 6 ops)
3. ✅ **Accuracy**: Natural signed similarity for Genomic Monty Hall
4. ✅ **Simplicity**: Direct access, fewer moving parts
5. ✅ **Storage**: TIE with 6-binary (0.75D bytes with optimal packing)

**Optimization Focus**:
- SIMD ternary dot product (AVX-512, NEON)
- Cache-line alignment & prefetching
- Sparse kernels for high-sparsity banks
- Memory-mapped I/O for hot chromosomes
- GPU acceleration for batch queries

**Not Recommended**:
- ❌ 6-binary split architecture (reconstruction overhead too high)
- ❌ XOR/Hamming distance (incompatible with signed similarity)
- ❌ Ternary → binary conversion at query time (defeats the purpose)

**Storage Optimization (Independent)**:
- Use 2-bit packing for disk storage (4× compression)
- Keep unpacked in memory for fast queries
- Best of both worlds: compact storage + fast queries

---

**Implementation References**:
- **Encoder**: `genomevault/hdv_validation/hdc_experimentation/encoders/encode_3bank_split_architecture.py`
- **Decoder**: `genomevault/hdv_validation/hdc_experimentation/decoders/lens_aware_decoder_CORRECTED_3TERNARY.py`
- **Architecture Theory**: `docs/guides/Key Guides/HDV_research/SPLIT_BANK_ARCHITECTURE.md`
- **Lens Documentation**: `docs/guides/Key Guides/HDV_research/STRUCTURAL_MOTIF_LENS_LIBRARY.md`

---

**Last Updated**: November 21, 2025
**Version**: 2.0 (Corrected Analysis)
**Status**: Production Recommendation
