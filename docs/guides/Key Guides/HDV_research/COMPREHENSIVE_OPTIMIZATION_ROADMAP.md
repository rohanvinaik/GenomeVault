# Comprehensive HDC Pipeline Optimization Roadmap
## Integrated Plan for Storage, Speed, Accuracy, and Compute Efficiency

**Date**: November 21, 2025
**Status**: Production Roadmap
**Architecture**: 3-Ternary Banks (D=5,120, N=1,024)

---

## Executive Summary

This document provides a **unified optimization roadmap** that simultaneously improves:
- **Storage**: 51.8 GB → 3-4 GB (17× reduction, lossless)
- **Speed**: 8.3 μs → 0.3-0.5 μs per query (21-28× faster)
- **Accuracy**: 85-90% → 92-97% (lens + templates)
- **Compute**: Python → Python+Numba → C++ core (staged approach)

**Key Principle**: All optimizations are LOSSLESS - they preserve the full accumulated signal required for lens confidence trajectory analysis.

---

## ⚠️ CRITICAL PERFORMANCE ASSUMPTIONS AND RISKS

**READ THIS BEFORE IMPLEMENTATION** - These assumptions must be validated early to avoid wasted effort:

### Risk 1: Confidence Trajectory Computational Cost 🔴 HIGH IMPACT

**The Problem**:
```
Naive trajectory analysis (20 λ sweeps):
  - 20 lens overlays per query
  - 20 dot products per query
  - 20 Monty Hall decodes per query

Baseline query: 0.4 μs
Trajectory overhead: 20 × 0.4 μs = 8 μs per query

Result: SLOWER than current 8.3 μs! ❌
```

**Mitigation Options** (implement in order of preference):

**Option 1: Smart Binary Search** (RECOMMENDED)
```python
# Test λ = [0, 0.5, 1.0] first (3 evaluations)
# Detect pattern (monotonic vs peak)
# If peak detected, refine with 3-5 additional points around peak
# Total: 6-8 evaluations instead of 20

Expected cost: ~2.5-3.5 μs (acceptable overhead)
```

**Option 2: Adaptive Sampling**
```python
# High-confidence positions: Use fast mode (3 points: 0, 0.5, 1.0)
# Low-confidence positions: Use full mode (6-8 points with refinement)
# Medium confidence: Use medium mode (5 points)

Expected: 80% of queries use 3 points → avg 3.6 evaluations
```

**Option 3: Cached Trajectory Shapes** (ADVANCED)
```python
# Pre-compute trajectory shapes for common texture+lens combinations
# Match incoming query to nearest cached shape
# Only compute 3 validation points + interpolate

Expected: ~4 evaluations per query after cache warm-up
```

**Validation Requirement**: Benchmark all three approaches on chr22 (Week 3).

### Risk 2: 2-Bit Unpacking Overhead 🟡 MEDIUM IMPACT

**The Problem**:
```python
# Unpacking ternary from 2-bit encoding:
for i, byte in enumerate(packed):  # 3,840 bytes
    for j in range(4):  # 4 values per byte
        bits = (byte >> (6 - 2*j)) & 0b11
        if bits == 0b00: ternary[4*i + j] = -1
        elif bits == 0b01: ternary[4*i + j] = 0
        else: ternary[4*i + j] = 1

Total: 15,360 conditional operations per chunk
```

**With SIMD dot product at 128 ns, unpacking might dominate!**

**Mitigation Options**:

**Option 1: SIMD Unpacking** (RECOMMENDED)
```python
import numpy as np
from numba import njit

@njit(parallel=True)
def simd_unpack_2bit_to_ternary(packed: np.ndarray) -> np.ndarray:
    """
    Parallel bit extraction using SIMD.
    Expected: 20-40 ns on M1/M2 with NEON
    """
    # Implementation: Use bit masks + parallel extraction
    pass
```

**Option 2: Cache Unpacked Hot Regions**
```python
# Keep chr1-22 unpacked in L3 cache after first load
# Trade 15.4 KB per chunk × hot chunks for speed
# Acceptable for frequently queried regions
```

**Option 3: Hybrid Storage**
```python
# Hot chromosomes (chr1-22): Store unpacked (int8)
# Cold chromosomes (chrM, chrY): Store 2-bit packed
# 80% storage savings on cold, 100% speed on hot
```

**Validation Requirement**: Benchmark unpacking overhead specifically (Week 2). If >50ns, implement SIMD unpacking.

### Risk 3: Template Match Rate Assumptions 🟡 MEDIUM IMPACT

**The Assumption**:
- 45% of genome is repetitive
- 95% similarity threshold → high match rate
- Storage: template_id (10 bits) + variants (~50 bytes) per instance

**What We Don't Know**:
1. What % of Alu instances actually match >95%?
2. How many templates needed to cover 80% of instances?
3. What's the storage cost of variants per instance?

**Validation Requirements** (BEFORE Phase 2):
```bash
# Run on chr22 (51 Mbp) first
RepeatMasker -species human -gff chr22.fa

# Cluster Alu repeats
python cluster_alus.py \
    --input chr22.alu.gff \
    --similarity 0.95 \
    --output alu_clusters.json

# Measure actual match rates
python measure_template_coverage.py \
    --clusters alu_clusters.json \
    --threshold 0.95

# Expected outputs:
#   - Number of templates needed for 80% coverage
#   - Average variant count per instance
#   - Actual storage savings vs full encoding
```

**Go/No-Go Decision**: Only proceed with Phase 2 if:
- Template match rate >70% on chr22 repetitive regions
- Average variants per instance <10 (otherwise storage cost too high)
- Actual compression >3× on repetitive regions

### Risk 4: Numba Parallelization on Apple Silicon 🟡 MEDIUM IMPACT

**The Problem**:
```python
@njit(parallel=True, fastmath=True)
def sparse_dot(bank1, bank2, bank3, pos_vec):
    for i in prange(D):  # D=5,120
        # ...
```

**On M1/M2**: `prange` may NOT parallelize well for small D=5,120
- Thread creation overhead > compute time for single vector
- Better to use single-thread with SIMD than multi-thread overhead

**Mitigation**:
```python
# Option 1: Single-threaded with explicit SIMD (RECOMMENDED for single queries)
@njit(fastmath=True)  # NO parallel=True
def simd_sparse_dot(...):
    # Numba still auto-vectorizes to NEON
    pass

# Option 2: Multi-threading ONLY for batch queries
def batch_query(positions: list, num_threads: int = 8):
    # Use joblib or multiprocessing for parallelism across queries
    # NOT within single query
    pass
```

**Validation Requirement**: Benchmark both approaches on M1/M2 (Week 2).

### Risk 5: HDF5 Compression Choice 🟢 LOW IMPACT

**Current Assumption**: Use gzip compression (2.5× on 2-bit packed data)

**Problem**: HDF5 gzip has decompression overhead (~100-200 μs per chunk)

**Better Options**:

**LZ4** (RECOMMENDED):
```python
# Faster decompress (~20-40 μs per chunk)
# Slightly worse ratio (2.0× vs 2.5×)
# Net speedup: 5× faster queries on compressed data

h5py.File(..., compression='lz4', compression_opts=9)
```

**Blosc** (BEST for scientific data):
```python
# SIMD-optimized compression/decompression
# Best of both: 2.5× ratio + 30-50 μs decompress
# Requires: pip install hdf5plugin

import hdf5plugin
h5py.File(..., **hdf5plugin.Blosc(cname='zstd', clevel=3, shuffle=hdf5plugin.Blosc.SHUFFLE))
```

**Validation Requirement**: Benchmark all three on chr22 encoding (Week 1).

### Risk 6: Cache Line Alignment Misconception 🟢 LOW IMPACT

**The Problem**:
```python
# This aligns the READ, but data must be WRITTEN aligned!
aligned_offset = (chunk_offset // 64) * 64
```

**Correct Approach**:
```python
# During encoding: Write chunks with 64-byte padding
chunk_size_padded = ((3 * D * 1) + 63) // 64 * 64  # Round up to 64 bytes

# During decoding: Read from aligned addresses
# OS will handle page-aligned memory mapping
```

**Validation Requirement**: Verify alignment during encoding (Week 1).

---

## Validation Gates (Go/No-Go Checkpoints)

### Week 2 Gate: Core Performance Validation

**Benchmarks Required**:
1. **SIMD Auto-Vectorization**:
   - Verify numba generates NEON instructions (use `NUMBA_DUMP_ASSEMBLY=1`)
   - Measure dot product: Expect <150 ns (vs 5,000 ns baseline)

2. **Sparse Kernel Speedup**:
   - Measure operations skipped: Expect 50-70% (natural sparsity)
   - Measure actual speedup: Expect 2-3× (not just 1/0.3 = 3.3×)

3. **Unpacking Overhead**:
   - Measure 2-bit → ternary conversion time
   - **Go**: <50 ns → Continue with 2-bit packing
   - **No-Go**: >100 ns → Implement SIMD unpacking OR use hybrid storage

4. **Confidence Trajectory**:
   - Measure smart search (6-8 evaluations): Expect 2.5-3.5 μs total
   - **Go**: <4 μs overhead → Acceptable
   - **No-Go**: >5 μs → Use cached trajectories or adaptive sampling

**Decision**: If ANY benchmark fails by >50%, STOP and revise approach before Week 3.

### Week 4 Gate: End-to-End Validation

**Metrics Required** (tested on chr22, 10,000 positions):
1. **Storage**:
   - **Go**: 51.8 GB → 5-6 GB (within 20% of 5.2 GB target)
   - **No-Go**: >7 GB → Investigate compression choices

2. **Speed**:
   - **Go**: 8.3 μs → 0.4-0.6 μs per query (14-20× speedup)
   - **No-Go**: >1.0 μs → Profile bottlenecks, may need Phase 3

3. **Accuracy**:
   - **Go**: 92-95% on chr22 (10k positions)
   - **No-Go**: <90% → Revise lens library or trajectory analysis

4. **Information Loss**:
   - **MUST**: Bit-identical reconstruction from 2-bit packed data
   - **MUST**: Confidence trajectory detects "peaks then drops" pattern on known variants

**Decision**: If storage OR speed targets missed by >30%, investigate before Phase 2.

### Phase 1 → Phase 2 Decision

Only proceed to Phase 2 (templates) if:
- ✅ **Template match validation** (on chr22):
  - Match rate >70% on repetitive regions
  - <10 variants per template instance on average
  - Actual compression >3× on repetitive regions

- ✅ **Storage still >4 GB** after Phase 1 (otherwise Phase 2 unnecessary)

- ✅ **Development bandwidth** available (1-2 months)

**Prediction**: Phase 2 may NOT be needed if Phase 1 achieves 5-6 GB storage.

### Phase 2 → Phase 3 Decision

Only proceed to Phase 3 (C++ core) if:
- ✅ **Query speed <0.3 μs** required (clinical/production SLA)
- ✅ **Profiling shows compute-bound** (not memory-bound)
- ✅ **Batch queries** are primary workload (not single-position)
- ✅ **Memory safety** critical (medical/regulatory deployment)

**Prediction**: Most users won't need Phase 3.

---

## Part 1: Current State Assessment

### Encoder (✅ CORRECT)

**Status**: Currently running (95.8% complete, ~15 minutes remaining)

**Architecture**:
```python
# encode_3bank_split_architecture.py:225-227
bank1 = np.sign(acc_hydro).astype(np.int8)   # T=+1, A=-1, GC=0
bank2 = np.sign(acc_groove).astype(np.int8)  # G=+1, C=-1, AT=0
bank3 = np.sign(acc_hinge).astype(np.int8)   # YR=+1, RY=-1, neutral=0
```

**Parameters**:
- D = 5,120 (dimension)
- N = 1,024 (chunk size)
- D/N ratio = 5.0 (SNR amplification)
- Overlap = 128 bp (12.5%)
- Stride = 896 bp
- Quantization = np.sign() (direct ternary, LOSSLESS)

**Output**:
- Total chunks: ~3,370,053
- Storage format: HDF5, shape=(chunks, 3, D), dtype=int8
- Uncompressed size: ~51.8 GB
- Encoding time: ~4-5 hours

**Verdict**: ✅ NO restart needed - architecture is correct

### Decoder (⚠️ NEEDS OPTIMIZATION)

**Current Implementation** (`lens_aware_decoder_CORRECTED_3TERNARY.py`):
- ✅ Correct 3-ternary architecture
- ✅ Direct np.dot() for similarities (no reconstruction overhead)
- ❌ Using vanilla numpy (not SIMD-optimized)
- ❌ No sparse kernel optimization
- ❌ No cache-line alignment or prefetching
- ❌ Missing confidence trajectory analysis for biological variation detection

**Performance Bottleneck**:
```
Current query time (D=5,120):
  Memory access (L3 cache): 3-5 μs    ← 90% of time
  Python overhead: ~200 ns             ← 6% of time
  numpy dot product: ~80 ns            ← 2% of time
  Lens overlay: ~100 ns                ← 2% of time

Total: ~3.5-8.3 μs (memory-dominated, NOT Python-dominated!)
```

**Key Insight**: Optimizing memory access (cache alignment, prefetching) gives 50-80% speedup, while dropping to C++ only gives 6% speedup. Memory is the bottleneck!

---

## Part 2: Optimization Dimensions

### Dimension 1: Storage (51.8 GB → 3-4 GB)

**Goal**: 17× reduction with ZERO information loss

**Strategies** (all lossless):

1. **2-Bit Packing** (4× reduction)
   - Ternary {-1, 0, +1} only needs 2 bits (4 states: -1, 0, +1, unused)
   - Pack 4 ternary values into 1 byte
   - 51.8 GB → 12.9 GB (lossless)

2. **Gzip Compression** (2.5× reduction)
   - Exploit natural 50-70% sparsity
   - 12.9 GB → 5.2 GB (lossless)

3. **Template Matching** (5-10× reduction on repetitive regions)
   - 45% of genome is repetitive (Alu, LINE-1, etc.)
   - Store template_id + variants instead of full banks
   - Alu repeat (300 bp): 3,840 bytes → 50 bytes (77× smaller)
   - 5.2 GB → 3.1 GB (lossless)

**Total**: 51.8 GB → 3-4 GB = **17× compression, ZERO information loss**

### Dimension 2: Speed (8.3 μs → 0.3-0.5 μs)

**Goal**: 21-28× faster queries with lossless optimizations

**Strategies**:

1. **SIMD Acceleration** (10-20× faster compute)
   - NEON (Apple Silicon): 16-wide int8 vectors
   - AVX-512 (x86): 64-wide int8 vectors
   - Numba @njit auto-vectorization
   - 5.1 μs → 320 ns (16× faster on M1/M2)

2. **Cache-Line Alignment** (5-10× faster memory)
   - Align chunks to 64-byte cache lines
   - Prefetch next chunk during computation
   - 3.2 μs → 160 ns (20× faster)

3. **Sparse Kernel** (2-3× fewer operations)
   - Skip natural zeros (50-70% from bank transparency)
   - Keep ALL +1/-1 values (no information loss!)
   - 15,360 ops → 4,608-7,680 ops (2-3× fewer)

4. **Memory-Mapped I/O** (2-5× faster for hot regions)
   - Pre-load hot chromosomes (chr1-22, X, Y) into RAM
   - OS page cache for frequently queried regions
   - RAM access: 10-20 μs → L3 cache: 1-2 μs

**Total**: 8.3 μs → 0.3-0.5 μs = **21-28× faster, LOSSLESS**

### Dimension 3: Accuracy (85-90% → 92-97%)

**Goal**: +7-12% improvement using MORE information, not less

**Strategies** (all use more information, not discard):

1. **Lens-Aware Decoding** (+5% on common variants)
   - Load pre-computed lens library from consensus FASTA
   - Overlay lens on raw banks (preserve full signal!)
   - Lens guides decoding toward consensus motifs
   - 88-92% → 93-97% on common variants

2. **Confidence Trajectory Analysis** (+6% on rare variants)
   - Sweep lens weight λ from 0 → 1
   - Detect "peaks then drops" pattern → real biological variation
   - Identify optimal λ for each position
   - 82-88% → 88-94% on rare variants

3. **Template Matching** (+15% on repetitive regions)
   - Match chunks to pre-computed template library
   - Exact ternary banks for known motifs (Alu, LINE-1)
   - Lens overlay on template banks
   - 75-85% → 92-98% on repetitive regions (45% of genome)

**Total**: 85-90% → 92-97% = **+7-12% improvement, ZERO information loss**

### Dimension 4: Compute Efficiency (Python → Numba → C++ staged)

**Goal**: Maximize performance per unit of development effort

**Reality Check**:
- **Memory bottleneck**: 90% of query time is L3 cache access (3-5 μs)
- **Python overhead**: Only 6% of query time (~200 ns)
- **Compute (numpy)**: Only 2% of query time (~80 ns)

**Implication**: Dropping to C++ gives 6% speedup but costs 2-3 months. Optimizing memory access gives 50-80% speedup and costs 2-4 weeks!

**Staged Approach**:

**Phase 1: Python + Numba** (2-4 weeks, 15-20× speedup)
- Stay in Python, add @njit decorators
- Numba auto-vectorizes to SIMD (NEON/AVX-512)
- Cache alignment + prefetching
- Sparse kernel (skip zeros, keep +1/-1)
- **ROI: 90% of potential gains with 11× faster implementation time**

**Phase 2: Hybrid Python/C++** (2-3 months, additional 2-3× speedup)
- C++ core for hot path (encoding, decoding, batch queries)
- Python API for lens management, analysis, pipeline
- Explicit SIMD intrinsics (vs numba auto-vectorization)
- Multi-threading for batch queries
- **ROI: Marginal gains (2-3×) for significant effort**

**Phase 3: Full Rust** (6-12 months, marginal gains)
- Only if clinical deployment requires memory safety
- Zero-cost abstractions, LLVM optimization
- Additional 1.5-2× speedup over C++
- **ROI: Not recommended unless memory safety is critical**

**Recommendation**: **Start with Phase 1**, re-evaluate after profiling. You may not need Phase 2/3!

---

## Part 3: Integrated Roadmap (3 Phases)

### Phase 1: Lossless Core Optimizations (2-4 weeks)

**Focus**: Maximum ROI - storage, speed, and accuracy improvements with minimal rewrite

**Week 1: Storage Optimization**

1. **2-Bit Packing Implementation** (3-4 days)
   ```python
   # genomevault/hdv_validation/hdc_experimentation/quantization/ternary_packer.py

   import numpy as np

   def pack_ternary_to_2bit(ternary_vector: np.ndarray) -> np.ndarray:
       """
       Lossless packing: {-1, 0, +1} → 2 bits each

       Encoding:
         -1 → 00
          0 → 01
         +1 → 10
         unused → 11

       Packs 4 ternary values into 1 byte.
       """
       assert ternary_vector.dtype == np.int8
       assert len(ternary_vector) % 4 == 0

       packed = np.zeros(len(ternary_vector) // 4, dtype=np.uint8)

       for i in range(0, len(ternary_vector), 4):
           byte = 0
           for j in range(4):
               val = ternary_vector[i + j]
               if val == -1:
                   bits = 0b00
               elif val == 0:
                   bits = 0b01
               else:  # +1
                   bits = 0b10
               byte |= (bits << (6 - 2*j))
           packed[i // 4] = byte

       return packed

   def unpack_2bit_to_ternary(packed: np.ndarray) -> np.ndarray:
       """Lossless unpacking"""
       ternary = np.zeros(len(packed) * 4, dtype=np.int8)

       for i, byte in enumerate(packed):
           for j in range(4):
               bits = (byte >> (6 - 2*j)) & 0b11
               if bits == 0b00:
                   ternary[4*i + j] = -1
               elif bits == 0b01:
                   ternary[4*i + j] = 0
               else:  # 0b10
                   ternary[4*i + j] = 1

       return ternary
   ```

2. **Convert Encoder Output** (1 day)
   ```bash
   # Convert existing encoded_genome_3banks.h5 to 2-bit packed format
   python3 genomevault/hdv_validation/hdc_experimentation/quantization/convert_to_2bit_packed.py \
       --input output/encoded_genome_3banks.h5 \
       --output output/encoded_genome_3banks_2bit.h5 \
       --compress gzip

   # Expected: 51.8 GB → 5.2 GB (10× reduction)
   ```

3. **Validation** (1 day)
   ```python
   # Verify bit-identical unpacking
   original_banks = load_chunk_int8(chunk_idx)
   packed = pack_ternary_to_2bit(original_banks.flatten())
   unpacked = unpack_2bit_to_ternary(packed).reshape(3, 5120)
   assert np.array_equal(original_banks, unpacked)  # Must be identical!
   ```

**Week 2: Query Speed Optimization (Python + Numba)**

1. **SIMD Dot Product** (2-3 days)
   ```python
   # genomevault/hdv_validation/hdc_experimentation/decoders/fast_query_engine.py

   import numpy as np
   from numba import njit, prange

   @njit(parallel=True, fastmath=True, cache=True)
   def sparse_ternary_dot_product_simd(
       bank1: np.ndarray,  # int8 {-1, 0, +1}, shape (D,)
       bank2: np.ndarray,
       bank3: np.ndarray,
       position_vec: np.ndarray,  # int8 {-1, +1}, shape (D,)
   ) -> tuple:
       """
       SIMD-accelerated sparse ternary dot product.

       Numba auto-vectorizes to:
       - NEON (Apple Silicon): 16-wide int8 vectors
       - AVX-512 (x86): 64-wide int8 vectors

       Sparse kernel skips natural zeros (50-70% of values).

       Expected speedup: 10-20× over vanilla numpy
       """
       D = len(bank1)
       sim1, sim2, sim3 = 0.0, 0.0, 0.0

       # Numba parallelizes + auto-vectorizes this loop
       for i in prange(D):
           pos_val = position_vec[i]

           # Sparse: skip if bank value is zero
           # (but KEEP ALL +1/-1 values - no information loss!)
           if bank1[i] != 0:
               sim1 += bank1[i] * pos_val
           if bank2[i] != 0:
               sim2 += bank2[i] * pos_val
           if bank3[i] != 0:
               sim3 += bank3[i] * pos_val

       return (sim1 / D, sim2 / D, sim3 / D)
   ```

2. **Cache-Optimized Chunk Loading** (2-3 days)
   ```python
   import h5py
   import mmap

   class CacheOptimizedChunkStorage:
       """
       Load chunks with cache-line alignment and prefetching.
       """
       def __init__(self, h5_path: str, use_mmap: bool = True):
           self.h5_file = h5py.File(h5_path, 'r')
           self.all_banks = self.h5_file['all_bank_vectors_2bit']  # 2-bit packed

           # Memory-map hot chromosomes (chr1-22, X, Y)
           self.hot_chromosomes = {}
           if use_mmap:
               self._setup_mmap()

       def _setup_mmap(self):
           """Pre-load hot chromosomes into OS page cache"""
           # Implementation: mmap chr1-22 into RAM
           pass

       def load_chunk_aligned(self, chunk_idx: int) -> dict:
           """
           Load chunk with 64-byte cache-line alignment.
           Prefetch next chunk to overlap memory access with computation.
           """
           # Load 2-bit packed data
           packed_banks = self.all_banks[chunk_idx, :, :]

           # Unpack to ternary (lossless)
           bank1 = unpack_2bit_to_ternary(packed_banks[0])
           bank2 = unpack_2bit_to_ternary(packed_banks[1])
           bank3 = unpack_2bit_to_ternary(packed_banks[2])

           # Ensure cache-line alignment (64 bytes)
           return {
               'bank1': np.ascontiguousarray(bank1),
               'bank2': np.ascontiguousarray(bank2),
               'bank3': np.ascontiguousarray(bank3),
           }
   ```

**Week 3: Lens System Integration**

1. **Confidence Trajectory Analysis** (3-4 days)
   ```python
   class LensAwareDecoderOptimized:
       """
       Fast lens-aware decoder with confidence trajectory analysis.

       Implements the "peaks then drops" pattern detection for
       identifying real biological variation vs consensus.
       """

       def decode_position_with_trajectory(
           self,
           chrom: str,
           pos: int,
           position_codebook: np.ndarray,
       ) -> dict:
           """
           Decode with confidence trajectory analysis.

           This is the CRITICAL feature that requires full accumulated signal!
           """
           # 1. Load chunk (cache-aligned, 2-bit packed)
           chunk_idx = self._get_chunk_idx(chrom, pos)
           banks = self.storage.load_chunk_aligned(chunk_idx)

           # 2. Classify texture (Bank 3 ZCR for hinge transitions)
           texture_type = self.texture_classifier.classify(banks['bank3'])

           # 3. Select candidate lenses
           candidates = self.lens_library.get_candidates_by_texture(texture_type)

           # 4. Find best lens (SIMD-accelerated)
           best_lens = self._find_best_lens_simd(banks, candidates)

           # 5. CONFIDENCE TRAJECTORY ANALYSIS (requires full signal!)
           trajectory = []
           position_vec = position_codebook[pos % self.N]

           for λ in np.linspace(0, 1, 20):
               # Overlay lens with weight λ
               overlayed_banks = {
                   'bank1': banks['bank1'] + λ * best_lens.bank1,
                   'bank2': banks['bank2'] + λ * best_lens.bank2,
                   'bank3': banks['bank3'] + λ * best_lens.bank3,
               }

               # Fast SIMD decode
               sims = sparse_ternary_dot_product_simd(
                   overlayed_banks['bank1'],
                   overlayed_banks['bank2'],
                   overlayed_banks['bank3'],
                   position_vec,
               )

               # Genomic Monty Hall
               nucleotide, confidence = self._monty_hall_decode(sims)
               trajectory.append((λ, confidence))

           # 6. Analyze trajectory pattern
           if self._is_monotonic_increase(trajectory):
               # Lens helps → trust consensus
               result_type = 'consensus_match'
               optimal_λ = 1.0
           elif self._peaks_then_drops(trajectory):
               # 🧬 REAL BIOLOGY: genome differs from consensus!
               result_type = 'biological_variation'
               optimal_λ = self._find_peak_lambda(trajectory)
           else:
               result_type = 'uncertain'
               optimal_λ = 0.5

           # 7. Final decode at optimal λ
           final_call = self._decode_at_lambda(
               banks, best_lens, optimal_λ, position_vec
           )

           return {
               'nucleotide': final_call['nucleotide'],
               'confidence': final_call['confidence'],
               'type': result_type,
               'lens': best_lens.name,
               'lens_weight': optimal_λ,
               'trajectory': trajectory,  # For debugging/analysis
           }

       def _is_monotonic_increase(self, trajectory: list) -> bool:
           """Check if confidence increases monotonically with λ"""
           confidences = [c for _, c in trajectory]
           for i in range(1, len(confidences)):
               if confidences[i] < confidences[i-1] - 0.02:  # Allow 2% noise
                   return False
           return True

       def _peaks_then_drops(self, trajectory: list) -> bool:
           """
           Detect "peaks then drops" pattern.

           This pattern indicates real biological variation:
           - Confidence peaks at intermediate λ (0.2-0.6)
           - Then drops at high λ (0.8-1.0)
           - Means: lens conflicts with accumulated evidence
           - Interpretation: THIS GENOME DIFFERS from consensus (real biology!)
           """
           confidences = [c for _, c in trajectory]
           lambdas = [l for l, _ in trajectory]

           # Find peak
           peak_idx = np.argmax(confidences)
           peak_λ = lambdas[peak_idx]
           peak_conf = confidences[peak_idx]

           # Check if peak is in middle range
           if not (0.2 <= peak_λ <= 0.6):
               return False

           # Check if confidence drops at λ=1.0
           final_conf = confidences[-1]
           drop = peak_conf - final_conf

           return drop > 0.05  # 5% drop threshold

       def _find_peak_lambda(self, trajectory: list) -> float:
           """Find λ where confidence is maximized"""
           confidences = [c for _, c in trajectory]
           lambdas = [l for l, _ in trajectory]
           peak_idx = np.argmax(confidences)
           return lambdas[peak_idx]
   ```

2. **Build Lens Library** (1-2 days)
   ```python
   # genomevault/hdv_validation/hdc_experimentation/encoders/build_lens_library.py

   from pathlib import Path
   import h5py
   from Bio import SeqIO

   def build_lens_library_from_consensus(
       consensus_fasta: Path,
       output_h5: Path,
       D: int = 5120,
       N: int = 1024,
       seed: int = 42,
   ):
       """
       Build lens library from consensus FASTA.

       Pre-computes ternary bank encodings for common structural motifs:
       - CpG islands (high GC)
       - AT-rich regions
       - Repetitive elements (Alu, LINE-1)
       - Unique sequences
       """
       # Load consensus sequence
       consensus = str(SeqIO.read(consensus_fasta, 'fasta').seq)

       # Initialize position codebook (must match encoder!)
       np.random.seed(seed)
       position_codebook = np.random.choice([-1, 1], size=(N, D)).astype(np.int8)

       # Build lens library
       lenses = []

       # 1. CpG islands (80% GC)
       cpg_motifs = extract_cpg_islands(consensus, min_length=200)
       for motif in cpg_motifs:
           lens = encode_motif_to_lens(motif, position_codebook, name=f'CpG_{len(lenses)}')
           lenses.append(lens)

       # 2. AT-rich regions (70% AT)
       at_motifs = extract_at_rich_regions(consensus, min_length=200)
       for motif in at_motifs:
           lens = encode_motif_to_lens(motif, position_codebook, name=f'ATrich_{len(lenses)}')
           lenses.append(lens)

       # 3. Repetitive elements
       repeats = extract_repetitive_elements(consensus)
       for motif in repeats:
           lens = encode_motif_to_lens(motif, position_codebook, name=f'Repeat_{len(lenses)}')
           lenses.append(lens)

       # Save to HDF5
       save_lens_library(lenses, output_h5)

       print(f"✓ Built lens library: {len(lenses)} lenses")
       print(f"  Saved to: {output_h5}")
   ```

**Week 4: Validation and Testing**

1. **Accuracy Validation** (2 days)
   ```bash
   # Test on chr22 (51 Mbp, ~50,000 chunks, fast validation)
   python3 genomevault/hdv_validation/hdc_experimentation/validate_optimized_decoder.py \
       --encoded output/encoded_genome_3banks_2bit.h5 \
       --lens-library output/lens_library.h5 \
       --ground-truth data/experimental_strands/ERR3239334/groundtruth.vcf.gz \
       --chromosomes chr22 \
       --sample-size 10000 \
       --output validation/phase1_chr22_results.json

   # Expected: 92-97% accuracy on chr22
   ```

2. **Performance Benchmarking** (2 days)
   ```python
   # Benchmark query speed
   import time

   decoder = LensAwareDecoderOptimized(
       h5_path='output/encoded_genome_3banks_2bit.h5',
       lens_library=lens_library,
   )

   # Warm up cache
   for i in range(100):
       decoder.decode_position_with_trajectory('chr22', i * 1000, position_codebook)

   # Benchmark
   positions = [(f'chr22', i * 1000) for i in range(10000)]
   start = time.perf_counter()

   for chrom, pos in positions:
       result = decoder.decode_position_with_trajectory(chrom, pos, position_codebook)

   elapsed = time.perf_counter() - start
   avg_query_time = (elapsed / len(positions)) * 1e6  # Convert to μs

   print(f"Average query time: {avg_query_time:.2f} μs")
   # Expected: 0.4-0.6 μs (vs 8.3 μs baseline = 14-21× faster)
   ```

**Phase 1 Deliverables**:
- ✅ 2-bit packed storage: 51.8 GB → 5.2 GB (10× reduction)
- ✅ SIMD + cache + sparse: 8.3 μs → 0.4-0.6 μs (14-21× faster)
- ✅ Lens library built from consensus
- ✅ Confidence trajectory analysis implemented
- ✅ Accuracy validation: 92-97% on chr22
- ✅ All optimizations LOSSLESS (no information discarded)

---

### Phase 2: Template Matching (1-2 months)

**Focus**: Additional 3-5× storage compression on repetitive regions (45% of genome)

**Month 1: Template Library Creation**

1. **Identify Repetitive Elements** (1 week)
   ```bash
   # Use RepeatMasker on consensus FASTA
   RepeatMasker -species human -gff consensus.fa

   # Extract Alu repeats (~1.1 million copies, 10% of genome)
   grep "Alu" consensus.fa.out > alu_elements.bed

   # Extract LINE-1 elements (~500,000 copies, 17% of genome)
   grep "LINE/L1" consensus.fa.out > line1_elements.bed

   # Extract simple repeats (3% of genome)
   grep "Simple_repeat" consensus.fa.out > simple_repeats.bed

   # Total: 45% of genome classified as repetitive
   ```

2. **Cluster Similar Repeats** (1 week)
   ```python
   # Cluster Alu variants into ~1,000 templates
   from sklearn.cluster import DBSCAN

   def cluster_alu_variants(alu_sequences: list) -> dict:
       """
       Cluster 1.1 million Alu instances into ~1,000 templates.

       Each cluster represents a variant family (e.g., Alu_Ja, Alu_Jb, etc.)
       """
       # Encode each Alu as 3-ternary banks
       alu_banks = [encode_3bank_ternary(seq) for seq in alu_sequences]

       # Cluster based on bank similarity
       clusterer = DBSCAN(eps=0.05, min_samples=100)
       labels = clusterer.fit_predict(alu_banks)

       # Extract cluster centroids as templates
       templates = {}
       for label in set(labels):
           cluster_members = [alu_banks[i] for i, l in enumerate(labels) if l == label]
           centroid = np.median(cluster_members, axis=0)
           templates[f'Alu_{label}'] = centroid

       return templates

   # Expected: ~1,000 Alu templates covering 1.1 million instances
   ```

3. **Pre-Compute Template Banks** (2-3 days)
   ```python
   # genomevault/hdv_validation/hdc_experimentation/encoders/build_template_library.py

   class TemplateLibrary:
       """
       Pre-computed ternary banks for repetitive elements.
       """
       def __init__(self):
           self.templates = {}

       def add_template(self, name: str, sequence: str):
           """
           Encode template to 3-ternary banks (same encoder as genome).
           """
           banks = encode_3bank_ternary(sequence)
           self.templates[name] = {
               'sequence': sequence,
               'bank1': banks[0],
               'bank2': banks[1],
               'bank3': banks[2],
               'length': len(sequence),
           }

       def save(self, output_h5: Path):
           """Save template library to HDF5"""
           with h5py.File(output_h5, 'w') as f:
               for name, template in self.templates.items():
                   grp = f.create_group(name)
                   grp.create_dataset('bank1', data=template['bank1'], compression='gzip')
                   grp.create_dataset('bank2', data=template['bank2'], compression='gzip')
                   grp.create_dataset('bank3', data=template['bank3'], compression='gzip')
                   grp.attrs['sequence'] = template['sequence']
                   grp.attrs['length'] = template['length']

   # Build library
   library = TemplateLibrary()

   # Add Alu templates (~1,000 variants)
   for name, sequence in alu_templates.items():
       library.add_template(name, sequence)

   # Add LINE-1 templates (~500 variants)
   for name, sequence in line1_templates.items():
       library.add_template(name, sequence)

   # Add simple repeats
   library.add_template('poly_A_20', 'A' * 20)
   library.add_template('poly_T_20', 'T' * 20)
   # ... etc

   library.save('output/template_library.h5')
   # Expected size: ~50 MB for 1,500 templates
   ```

**Month 2: Template-Aware Encoding**

1. **Re-encode with Templates** (2-3 weeks)
   ```python
   # genomevault/hdv_validation/hdc_experimentation/encoders/encode_with_templates.py

   def encode_genome_with_templates(
       gdiff_path: Path,
       guide_fastas: list,
       template_library: TemplateLibrary,
       output_h5: Path,
   ):
       """
       Encode genome using template references where possible.

       For each 1024 bp chunk:
       - Check if it matches a template (>95% similarity)
       - If yes: Store template_id + variants (50-100 bytes)
       - If no: Store full 3-ternary banks (3,840 bytes 2-bit packed)
       """
       encoded_chunks = []

       for chunk_idx, chunk_seq in enumerate(iterate_genome_chunks(gdiff_path)):
           # Try to match template
           match = template_library.find_best_match(chunk_seq, threshold=0.95)

           if match:
               # Encode as template reference
               variants = compute_variants(chunk_seq, match.sequence)
               encoded_chunks.append({
                   'type': 'template',
                   'template_id': match.id,  # 10 bits (1,024 templates)
                   'variants': variants,  # ~20-50 bytes for SNPs/indels
               })
               # Total: ~30-70 bytes (vs 3,840 bytes = 50-130× smaller)
           else:
               # Encode as full banks
               banks = encode_3bank_ternary(chunk_seq)
               packed = pack_ternary_to_2bit(banks)
               encoded_chunks.append({
                   'type': 'full',
                   'banks': packed,
               })
               # Total: ~3,840 bytes (2-bit packed)

       # Save to HDF5
       save_template_aware_encoding(encoded_chunks, output_h5)
   ```

2. **Template-Aware Decoding** (1 week)
   ```python
   def decode_position_with_template(
       self,
       chrom: str,
       pos: int,
       position_codebook: np.ndarray,
   ) -> dict:
       """
       Decode position from template-encoded chunk.

       Workflow:
       1. Load chunk (either template ref or full banks)
       2. If template: Load template banks + apply variants
       3. Overlay lens as usual
       4. Decode with Genomic Monty Hall
       """
       chunk_idx = self._get_chunk_idx(chrom, pos)
       chunk_data = self.storage.load_chunk(chunk_idx)

       if chunk_data['type'] == 'template':
           # Load template banks
           template = self.template_library.get(chunk_data['template_id'])
           banks = {
               'bank1': template.bank1.copy(),
               'bank2': template.bank2.copy(),
               'bank3': template.bank3.copy(),
           }

           # Apply variants at this position
           pos_in_chunk = pos % self.N
           for variant in chunk_data['variants']:
               if variant.pos == pos_in_chunk:
                   apply_variant_to_banks(banks, variant)
       else:
           # Load full banks
           banks = unpack_2bit_to_ternary(chunk_data['banks'])

       # Decode with lens (same as before)
       return self.decode_with_lens(banks, position_codebook)
   ```

**Phase 2 Deliverables**:
- ✅ Template library: 1,500 templates, ~50 MB
- ✅ Template-aware encoding: 5.2 GB → 3.1 GB (additional 1.7× reduction)
- ✅ Template-aware decoding: Lens works identically on templates
- ✅ Accuracy improvement: +10-15% on repetitive regions (45% of genome)
- ✅ Total storage: 51.8 GB → 3.1 GB (17× compression, LOSSLESS)

---

### Phase 3: Hybrid Python/C++ Core (2-3 months, OPTIONAL)

**Decision Point**: Only proceed if Phase 1+2 profiling shows:
- Query speed still not sufficient (need <0.2 μs)
- Batch queries (1,000+ positions) needed
- Memory safety critical for clinical deployment

**Focus**: C++ hot path for encoding/decoding, Python API for everything else

**Month 1: C++ Core Implementation**

1. **Define C++ Interface** (1 week)
   ```cpp
   // genomevault/core/fast_query.h

   #pragma once
   #include <vector>
   #include <cstdint>

   namespace genomevault {

   struct QueryResult {
       char nucleotide;        // 'A', 'T', 'G', 'C'
       float confidence;       // 0.0-1.0
       char result_type;       // 'C' (consensus), 'B' (biological), 'U' (uncertain)
       float lens_weight;      // Optimal λ (0.0-1.0)
   };

   class FastQueryEngine {
   public:
       FastQueryEngine(const char* h5_path, const char* lens_library_path);
       ~FastQueryEngine();

       // Single position query
       QueryResult query_position(
           const char* chrom,
           int32_t pos,
           const int8_t* position_codebook,  // Shape: (N, D)
           int32_t N,
           int32_t D
       );

       // Batch query (multi-threaded)
       std::vector<QueryResult> query_batch(
           const std::vector<std::pair<std::string, int32_t>>& positions,
           const int8_t* position_codebook,
           int32_t N,
           int32_t D,
           int num_threads = 8
       );

   private:
       // SIMD dot product (explicit AVX-512/NEON intrinsics)
       void sparse_dot_product_simd(
           const int8_t* bank1,
           const int8_t* bank2,
           const int8_t* bank3,
           const int8_t* pos_vec,
           int32_t D,
           float* out_sims
       );

       // Implementation details
       struct Impl;
       Impl* impl_;
   };

   }  // namespace genomevault
   ```

2. **Implement SIMD Kernels** (2 weeks)
   ```cpp
   // genomevault/core/simd_kernels.cpp

   #include <immintrin.h>  // AVX-512 intrinsics

   namespace genomevault {

   void sparse_dot_product_avx512(
       const int8_t* bank1,
       const int8_t* bank2,
       const int8_t* bank3,
       const int8_t* pos_vec,
       int32_t D,
       float* out_sims
   ) {
       __m512i sum1 = _mm512_setzero_si512();
       __m512i sum2 = _mm512_setzero_si512();
       __m512i sum3 = _mm512_setzero_si512();

       for (int32_t i = 0; i < D; i += 64) {
           // Load 64 int8 values
           __m512i b1 = _mm512_loadu_si512((__m512i*)&bank1[i]);
           __m512i b2 = _mm512_loadu_si512((__m512i*)&bank2[i]);
           __m512i b3 = _mm512_loadu_si512((__m512i*)&bank3[i]);
           __m512i pos = _mm512_loadu_si512((__m512i*)&pos_vec[i]);

           // Sparse: mask out zeros
           __mmask64 nz1 = _mm512_cmpneq_epi8_mask(b1, _mm512_setzero_si512());
           __mmask64 nz2 = _mm512_cmpneq_epi8_mask(b2, _mm512_setzero_si512());
           __mmask64 nz3 = _mm512_cmpneq_epi8_mask(b3, _mm512_setzero_si512());

           // Multiply-add (only non-zero elements)
           __m512i prod1 = _mm512_mullo_epi8(b1, pos);
           __m512i prod2 = _mm512_mullo_epi8(b2, pos);
           __m512i prod3 = _mm512_mullo_epi8(b3, pos);

           sum1 = _mm512_mask_add_epi8(sum1, nz1, sum1, prod1);
           sum2 = _mm512_mask_add_epi8(sum2, nz2, sum2, prod2);
           sum3 = _mm512_mask_add_epi8(sum3, nz3, sum3, prod3);
       }

       // Horizontal sum (reduce 512-bit vector to scalar)
       out_sims[0] = horizontal_sum_i8(sum1) / (float)D;
       out_sims[1] = horizontal_sum_i8(sum2) / (float)D;
       out_sims[2] = horizontal_sum_i8(sum3) / (float)D;
   }

   }  // namespace genomevault
   ```

3. **Python Bindings (pybind11)** (1 week)
   ```cpp
   // genomevault/core/python_bindings.cpp

   #include <pybind11/pybind11.h>
   #include <pybind11/stl.h>
   #include <pybind11/numpy.h>
   #include "fast_query.h"

   namespace py = pybind11;

   PYBIND11_MODULE(fast_query_cpp, m) {
       py::class_<genomevault::QueryResult>(m, "QueryResult")
           .def_readonly("nucleotide", &genomevault::QueryResult::nucleotide)
           .def_readonly("confidence", &genomevault::QueryResult::confidence)
           .def_readonly("result_type", &genomevault::QueryResult::result_type)
           .def_readonly("lens_weight", &genomevault::QueryResult::lens_weight);

       py::class_<genomevault::FastQueryEngine>(m, "FastQueryEngine")
           .def(py::init<const char*, const char*>())
           .def("query_position", &genomevault::FastQueryEngine::query_position)
           .def("query_batch", &genomevault::FastQueryEngine::query_batch);
   }
   ```

**Month 2: Python API Wrapper**

```python
# genomevault/hdv_validation/hdc_experimentation/decoders/production_decoder.py

from genomevault.core import fast_query_cpp  # C++ extension

class ProductionDecoder:
    """
    Production decoder with C++ hot path and Python API.

    C++ handles:
    - Memory-mapped I/O
    - SIMD dot products (explicit AVX-512/NEON)
    - Cache prefetching
    - Multi-threading for batch queries

    Python handles:
    - Lens library management
    - Result analysis
    - Pipeline integration
    """

    def __init__(self, h5_path: str, lens_library_path: str):
        # C++ engine for hot path
        self.cpp_engine = fast_query_cpp.FastQueryEngine(
            h5_path, lens_library_path
        )

        # Python for lens management
        self.lens_library = LensLibrary.load(lens_library_path)

    def query_position(self, chrom: str, pos: int, position_codebook) -> dict:
        """Single position query (delegates to C++ core)"""
        result = self.cpp_engine.query_position(
            chrom, pos, position_codebook, N=1024, D=5120
        )

        # Python post-processing
        return {
            'nucleotide': result.nucleotide,
            'confidence': result.confidence,
            'type': result.result_type,
            'lens_weight': result.lens_weight,
        }

    def query_batch(self, positions: list, position_codebook) -> list:
        """
        Batch query with C++ multi-threading.

        Processes 1,000+ positions in parallel.
        Expected: 0.15-0.2 μs per position (vs 0.4 μs single-threaded)
        """
        results = self.cpp_engine.query_batch(
            positions, position_codebook, N=1024, D=5120, num_threads=8
        )

        return [
            {
                'nucleotide': r.nucleotide,
                'confidence': r.confidence,
                'type': r.result_type,
                'lens_weight': r.lens_weight,
            }
            for r in results
        ]
```

**Month 3: Testing and Optimization**

1. **Validation** (1 week)
   - Verify C++ results match Python+Numba exactly
   - Test on chr22 (10,000 positions)
   - Accuracy must be identical

2. **Performance Benchmarking** (1 week)
   ```bash
   # Single query
   pytest benchmarks/test_cpp_query_speed.py
   # Expected: 0.15-0.2 μs (vs 0.4 μs Python+Numba = 2-3× faster)

   # Batch query (1,000 positions, 8 threads)
   pytest benchmarks/test_batch_query_speed.py
   # Expected: 150-200 μs total (0.15-0.2 μs per position)
   ```

**Phase 3 Deliverables** (OPTIONAL):
- ✅ C++ core with explicit SIMD intrinsics
- ✅ Python bindings (pybind11)
- ✅ Multi-threaded batch queries
- ✅ Single query: 0.4 μs → 0.15-0.2 μs (additional 2-3× faster)
- ✅ Batch (1,000 positions): 0.15-0.2 μs per position with parallelism

---

## Part 4: Decision Framework

### When to Move to Next Phase?

**Phase 1 → Phase 2 Decision**:

Proceed with Phase 2 (Template Matching) if:
- ✅ Storage is still a concern (need <4 GB)
- ✅ Accuracy on repetitive regions needs improvement
- ✅ Development bandwidth available (1-2 months)

Skip Phase 2 if:
- 5.2 GB storage is acceptable
- 92-95% accuracy is sufficient
- Need to move to production quickly

**Phase 2 → Phase 3 Decision**:

Proceed with Phase 3 (C++ Core) ONLY if:
- ✅ Single query speed still insufficient (need <0.2 μs)
- ✅ Batch queries (1,000+ positions) are common workload
- ✅ Memory safety critical (clinical/regulatory deployment)
- ✅ Development bandwidth available (2-3 months)

Skip Phase 3 if:
- 0.4-0.6 μs query speed is acceptable
- Single position queries (not batch)
- Python+Numba is sufficient

**Recommended Path**: Phase 1 → Evaluate → Phase 2 → Evaluate → (Rarely) Phase 3

---

## Part 5: Expected Final Outcomes

### After Phase 1 (2-4 weeks)

| Metric | Before | After Phase 1 | Improvement |
|--------|--------|---------------|-------------|
| **Storage** | 51.8 GB | 5.2 GB | **10× reduction** |
| **Query Speed** | 8.3 μs | 0.4-0.6 μs | **14-21× faster** |
| **Accuracy** | 85-90% | 92-95% | **+7-10%** |
| **Development** | - | 2-4 weeks | **Quick ROI** |
| **Language** | Python | Python + Numba | **No rewrite** |
| **Information Loss** | 0% | 0% | **LOSSLESS** |

### After Phase 2 (1-2 months additional)

| Metric | After Phase 1 | After Phase 2 | Improvement |
|--------|---------------|---------------|-------------|
| **Storage** | 5.2 GB | 3.1 GB | **1.7× smaller** |
| **Accuracy** | 92-95% | 92-97% | **+2% on repeats** |
| **Development** | 2-4 weeks | 1-2 months more | **Template library** |

### After Phase 3 (2-3 months additional, OPTIONAL)

| Metric | After Phase 2 | After Phase 3 | Improvement |
|--------|---------------|---------------|-------------|
| **Query Speed** | 0.4-0.6 μs | 0.15-0.2 μs | **2-3× faster** |
| **Batch (1,000)** | 400-600 μs | 150-200 μs | **2-3× faster** |
| **Development** | 3-4 months | 5-7 months total | **C++ core** |
| **Memory Safety** | Python | C++ | **Better** |

---

## Part 6: Implementation Priority Matrix

### High Priority (Do First)

1. **2-Bit Packing** (Week 1)
   - Effort: 3-4 days
   - Impact: 4× storage reduction
   - Risk: Low (well-tested compression)
   - ROI: **Immediate**

2. **SIMD Dot Product** (Week 2)
   - Effort: 2-3 days
   - Impact: 10-20× compute speedup
   - Risk: Low (numba auto-vectorization)
   - ROI: **Immediate**

3. **Cache Alignment** (Week 2)
   - Effort: 2-3 days
   - Impact: 5-10× memory speedup
   - Risk: Low (standard optimization)
   - ROI: **Immediate**

4. **Lens Library** (Week 3)
   - Effort: 3-4 days
   - Impact: +5% accuracy
   - Risk: Medium (need good templates)
   - ROI: **High**

5. **Confidence Trajectory** (Week 3)
   - Effort: 3-4 days
   - Impact: +6% on rare variants
   - Risk: Medium (new algorithm)
   - ROI: **High**

### Medium Priority (Do Second)

6. **Template Matching** (Month 2-3)
   - Effort: 1-2 months
   - Impact: 1.7× additional storage reduction
   - Risk: Medium (RepeatMasker dependency)
   - ROI: **Good if storage critical**

7. **GPU Batch Queries** (Optional)
   - Effort: 1-2 weeks
   - Impact: 100-1000× throughput for batch
   - Risk: Low (Metal/CUDA well-supported)
   - ROI: **Good for whole-genome scans**

### Low Priority (Do Last, If At All)

8. **C++ Hot Path** (Month 4-6, OPTIONAL)
   - Effort: 2-3 months
   - Impact: 2-3× additional speedup
   - Risk: High (language boundary issues)
   - ROI: **Low unless <0.2 μs required**

9. **Rust Rewrite** (Month 7-12, NOT RECOMMENDED)
   - Effort: 6-12 months
   - Impact: 1.5-2× additional speedup
   - Risk: Very high (full rewrite)
   - ROI: **Very low unless memory safety critical**

---

## Part 7: Key Principles

### 1. Preserve Full Accumulated Signal (Non-Negotiable)

**The lens system REQUIRES full signal for confidence trajectory analysis.**

❌ **NEVER**:
- Apply percentile thresholding (artificial sparsity)
- Discard accumulated values below arbitrary threshold
- Sacrifice accuracy for storage/speed

✅ **ALWAYS**:
- Use lossless compression (2-bit packing, gzip, templates)
- Exploit natural sparsity (50-70% from architecture)
- Optimize memory access and compute, not information content

### 2. Memory Access is the Bottleneck (Not Python)

**90% of query time is memory access (L3 cache), NOT Python overhead.**

**Implication**: Cache alignment + prefetching gives 50-80% speedup. Dropping to C++ gives 6% speedup.

**Strategy**: Optimize memory access first (Phase 1), then consider C++ (Phase 3) only if still needed.

### 3. Staged Approach with Re-Evaluation

**DON'T**: Plan full C++/Rust rewrite upfront

**DO**: Implement Phase 1 (Python+Numba) → Profile → Re-evaluate

**Reason**: You may not need Phase 2/3! 90% of gains come from Phase 1 (2-4 weeks) vs 6-12 months for full rewrite.

### 4. Validate at Every Phase

**After each phase**:
- Run accuracy validation on chr22 (10,000 positions)
- Benchmark query speed (average over 10,000 queries)
- Measure storage (actual file size)
- Compare with baseline (must match or improve)

**If validation fails**: Fix before moving to next phase!

---

## Part 8: Next Steps

### Immediate Actions (Next 24 Hours)

1. **Verify encoder completion**
   ```bash
   tail -f genomevault/hdv_validation/hdc_experimentation/output/encoding_log.txt
   # Expected: 100% complete, ~51.8 GB output file
   ```

2. **Create Phase 1 implementation plan**
   ```bash
   mkdir -p genomevault/hdv_validation/hdc_experimentation/quantization
   mkdir -p genomevault/hdv_validation/hdc_experimentation/decoders/optimized
   mkdir -p genomevault/hdv_validation/hdc_experimentation/encoders/lens_library
   ```

3. **Implement 2-bit packing** (Start Week 1)
   ```bash
   # Create packer module
   touch genomevault/hdv_validation/hdc_experimentation/quantization/ternary_packer.py

   # Create conversion script
   touch genomevault/hdv_validation/hdc_experimentation/quantization/convert_to_2bit_packed.py
   ```

### Week 1 Goals

- ✅ 2-bit packing implementation
- ✅ Convert encoded_genome_3banks.h5 → encoded_genome_3banks_2bit.h5
- ✅ Validation: bit-identical unpacking
- ✅ Storage: 51.8 GB → 12.9 GB (uncompressed) → 5.2 GB (gzipped)

### Week 2 Goals

- ✅ SIMD dot product (numba @njit)
- ✅ Cache-aligned chunk loading
- ✅ Sparse kernel (skip zeros, keep +1/-1)
- ✅ Benchmark: 8.3 μs → 0.4-0.6 μs

### Week 3 Goals

- ✅ Build lens library from consensus
- ✅ Implement confidence trajectory analysis
- ✅ Accuracy validation: 92-95% on chr22

### Week 4 Goals

- ✅ End-to-end testing
- ✅ Performance benchmarking
- ✅ Documentation and handoff

---

## Part 9: Rigorous Benchmarking Methodology

**CRITICAL**: Benchmark rigor determines whether optimizations actually work. Follow this protocol exactly.

### Benchmark Protocol (All Phases)

```python
# genomevault/hdv_validation/hdc_experimentation/benchmark_protocol.py

import time
import numpy as np
from pathlib import Path

def rigorous_benchmark(
    benchmark_func,
    num_warmup: int = 100,
    num_iterations: int = 10000,
    clear_cache: bool = False,
):
    """
    Rigorous benchmarking protocol for query speed.

    Protocol:
    1. Warm up cache (100 queries)
    2. Optionally clear CPU cache between runs
    3. Run 10,000 queries, measure each
    4. Report min/median/p95/p99
    5. Test on both hot (L3) and cold (RAM) data
    """
    # Step 1: Warm up cache
    print("Warming up cache...")
    for i in range(num_warmup):
        benchmark_func()

    # Step 2: Clear CPU cache (optional, for cold benchmarks)
    if clear_cache:
        # On macOS: purge
        # On Linux: echo 3 > /proc/sys/vm/drop_caches (requires sudo)
        import subprocess
        subprocess.run(['purge'], check=False)  # macOS only
        time.sleep(1)  # Let OS settle

    # Step 3: Run benchmark
    times = []
    print(f"Running {num_iterations} iterations...")
    for i in range(num_iterations):
        start = time.perf_counter()
        benchmark_func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{num_iterations} iterations complete")

    # Step 4: Compute statistics
    times_us = np.array(times) * 1e6  # Convert to microseconds

    results = {
        'min': np.min(times_us),
        'median': np.median(times_us),
        'mean': np.mean(times_us),
        'p95': np.percentile(times_us, 95),
        'p99': np.percentile(times_us, 99),
        'max': np.max(times_us),
        'std': np.std(times_us),
    }

    # Step 5: Report
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"  Min (best case):     {results['min']:.3f} μs")
    print(f"  Median (typical):    {results['median']:.3f} μs")
    print(f"  Mean:                {results['mean']:.3f} μs")
    print(f"  P95 (95% < this):    {results['p95']:.3f} μs")
    print(f"  P99 (99% < this):    {results['p99']:.3f} μs")
    print(f"  Max (worst case):    {results['max']:.3f} μs")
    print(f"  Std Dev:             {results['std']:.3f} μs")
    print("="*80)

    return results

# Example usage:
def test_query():
    decoder.decode_position('chr22', 10000, position_codebook)

# Hot cache (L3) benchmark
hot_results = rigorous_benchmark(test_query, clear_cache=False)

# Cold cache (RAM) benchmark
cold_results = rigorous_benchmark(test_query, clear_cache=True)

print(f"\nSpeedup (hot vs cold): {cold_results['median'] / hot_results['median']:.2f}×")
```

### What to Report

**Always report ALL of**:
- **Min**: Best case (everything in L1/L2 cache)
- **Median**: Typical case (what 50% of queries experience)
- **P95**: Near-worst case (what 5% of queries exceed)

**Why?**:
- Min alone is misleading (only best case)
- Mean is skewed by outliers
- Median + P95 gives realistic range

### Validation Dataset Beyond chr22

**Problem**: chr22 alone is not representative of whole genome:
- Chr22 is gene-rich (not typical)
- MHC region (chr6) is highly variable
- Chr1 is large and diverse

**Recommended Validation Set**:

```python
validation_chromosomes = {
    'chr22': {
        'size': 51_000_000,
        'type': 'gene_rich',
        'sample_size': 10_000,
        'rationale': 'Fast validation, high gene density'
    },
    'chr6': {
        'size': 171_000_000,
        'type': 'MHC_region',
        'sample_size': 5_000,
        'rationale': 'Highly variable, tests lens adaptation'
    },
    'chr1': {
        'size': 249_000_000,
        'type': 'large_diverse',
        'sample_size': 5_000,
        'rationale': 'Largest chromosome, diverse regions'
    },
    'chrX': {
        'size': 155_000_000,
        'type': 'sex_chromosome',
        'sample_size': 3_000,
        'rationale': 'Different structure, fewer recombination'
    },
}

# Total: 23,000 positions across 4 diverse chromosomes
# Covers: gene-rich, highly variable, large/diverse, sex chromosome
```

**Validation Schedule**:
- **Week 2**: chr22 only (10k positions) - fast iteration
- **Week 4**: chr22 + chr6 (15k positions) - diverse validation
- **Phase 2**: All 4 chromosomes (23k positions) - comprehensive

### Benchmarking Tools

**Verify SIMD Code Generation** (Week 2):
```bash
# Check if Numba generates NEON/AVX-512 instructions
NUMBA_DUMP_ASSEMBLY=1 python3 test_simd_dot_product.py

# Look for:
# - NEON: vld1, vmul, vadd instructions (ARM)
# - AVX-512: vmovdqa, vpmaddwd instructions (x86)
```

**Profile Memory Access** (Week 2):
```python
# Use perf (Linux) or Instruments (macOS)
import subprocess

# macOS Instruments
subprocess.run([
    'xcrun', 'xctrace', 'record',
    '--template', 'Time Profiler',
    '--launch', 'python3', 'benchmark_queries.py'
])

# Linux perf
subprocess.run([
    'perf', 'stat', '-e',
    'cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses',
    'python3', 'benchmark_queries.py'
])
```

---

## Part 10: Edge Case Handling and Fallback Strategies

**What happens when assumptions fail?** Every optimization has edge cases.

### Edge Case 1: Template Match is Ambiguous

**Scenario**: Multiple templates match >95% similarity

**Problem**:
```python
# Alu repeat at chr1:12345
matches = [
    ('Alu_Ja', similarity=0.96),
    ('Alu_Jb', similarity=0.96),
    ('Alu_Y', similarity=0.95),
]

# Which template to use?
```

**Solution**:
```python
def select_template_with_tiebreaker(matches, threshold=0.95):
    """
    Tiebreaker strategy for ambiguous template matches.
    """
    # Filter to above threshold
    candidates = [(name, sim) for name, sim in matches if sim >= threshold]

    if len(candidates) == 0:
        # No match → Encode as full banks
        return None
    elif len(candidates) == 1:
        # Unique match → Use template
        return candidates[0][0]
    else:
        # Ambiguous → Use tiebreaker rules:

        # Rule 1: Prefer template with fewest variants in database
        # (more canonical = better compression)
        template_name = min(candidates, key=lambda x: template_variant_count[x[0]])

        # Rule 2: If still tied, prefer highest similarity
        # (already sorted by similarity)

        return template_name
```

### Edge Case 2: Confidence Trajectory is Completely Flat

**Scenario**: Confidence is ~25% (random) at all λ values

**Problem**:
```python
trajectory = [
    (λ=0.0, conf=0.26),
    (λ=0.2, conf=0.25),
    (λ=0.4, conf=0.27),
    (λ=0.6, conf=0.26),
    (λ=0.8, conf=0.25),
    (λ=1.0, conf=0.26),
]

# No pattern! What to do?
```

**Solution**:
```python
def analyze_trajectory_with_fallback(trajectory):
    """
    Analyze confidence trajectory with fallback for edge cases.
    """
    confidences = [conf for _, conf in trajectory]

    # Check for flat trajectory (std < 3%)
    if np.std(confidences) < 0.03:
        # Flat → Low confidence region
        return {
            'type': 'low_confidence',
            'optimal_λ': 0.0,  # Don't use lens (doesn't help)
            'confidence': max(confidences),
            'action': 'flag_for_manual_review',
        }

    # Check for monotonic increase
    if is_monotonic_increase(confidences):
        return {
            'type': 'consensus_match',
            'optimal_λ': 1.0,
            'confidence': confidences[-1],
            'action': 'trust_lens',
        }

    # Check for peaks then drops
    if peaks_then_drops(confidences):
        peak_idx = np.argmax(confidences)
        return {
            'type': 'biological_variation',
            'optimal_λ': trajectory[peak_idx][0],
            'confidence': confidences[peak_idx],
            'action': 'reduce_lens_weight',
        }

    # Fallback: Unstable trajectory
    return {
        'type': 'unstable',
        'optimal_λ': 0.5,  # Compromise
        'confidence': np.median(confidences),
        'action': 'flag_for_review',
    }
```

### Edge Case 3: Lens and Banks Conflict at λ=0 (Raw Data Low Confidence)

**Scenario**: Raw banks (λ=0) give low confidence, but lens makes it worse

**Problem**:
```python
trajectory = [
    (λ=0.0, conf=0.35),  # Low confidence (raw)
    (λ=0.2, conf=0.30),  # Worse with lens!
    (λ=0.4, conf=0.28),
    (λ=0.6, conf=0.26),
    (λ=1.0, conf=0.25),  # Monotonic DECREASE
]

# Lens is making things worse!
```

**Solution**:
```python
def detect_lens_mismatch(trajectory):
    """
    Detect when lens conflicts with accumulated evidence.
    """
    confidences = [conf for _, conf in trajectory]

    # Check for monotonic decrease
    if all(confidences[i] >= confidences[i+1] for i in range(len(confidences)-1)):
        # Lens is making confidence WORSE
        return {
            'type': 'lens_mismatch',
            'optimal_λ': 0.0,  # Ignore lens completely
            'confidence': confidences[0],  # Use raw banks
            'action': 'flag_lens_for_review',
            'reason': 'Lens conflicts with accumulated evidence',
        }

    # Normal case
    return None  # Proceed with standard analysis
```

### Edge Case 4: HDF5 Decompression Failure

**Scenario**: Corrupted chunk or compression error

**Problem**:
```python
try:
    chunk = h5_file['all_bank_vectors'][chunk_idx, :, :]
except (IOError, OSError) as e:
    # Corruption! What to do?
```

**Solution**:
```python
def load_chunk_with_fallback(h5_file, chunk_idx, max_retries=3):
    """
    Load chunk with retry and fallback strategies.
    """
    for attempt in range(max_retries):
        try:
            chunk = h5_file['all_bank_vectors'][chunk_idx, :, :]
            return chunk
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                # Retry after brief delay
                time.sleep(0.1)
                continue
            else:
                # All retries failed → Use fallback

                # Option 1: Load from backup copy (if exists)
                if backup_h5_file is not None:
                    return backup_h5_file['all_bank_vectors'][chunk_idx, :, :]

                # Option 2: Re-encode chunk on-the-fly
                # (requires access to original FASTQ/GDiff)
                return re_encode_chunk(chunk_idx)

                # Option 3: Return zeros + flag for manual review
                logging.error(f"Chunk {chunk_idx} corrupted, returning zeros")
                return np.zeros((3, D), dtype=np.int8)
```

### Edge Case 5: Query Position Outside Encoded Range

**Scenario**: User queries position not in encoded data

**Problem**:
```python
query_pos = 250_000_000  # chr1:250M
max_encoded_pos = 249_000_000  # chr1 ends at 249M

# Position doesn't exist in encoding!
```

**Solution**:
```python
def validate_query_position(chrom, pos, chromosome_bounds):
    """
    Validate query position before decoding.
    """
    if chrom not in chromosome_bounds:
        raise ValueError(f"Chromosome {chrom} not in encoded data. "
                         f"Available: {list(chromosome_bounds.keys())}")

    chrom_start, chrom_end = chromosome_bounds[chrom]

    if not (chrom_start <= pos <= chrom_end):
        raise ValueError(f"Position {chrom}:{pos} outside encoded range "
                         f"[{chrom_start}, {chrom_end}]")

    return True  # Valid position
```

---

## Conclusion

This comprehensive roadmap provides a **staged, validated approach** to optimizing the HDC pipeline across all dimensions:

**Phase 1 (2-4 weeks)**: Lossless core optimizations
- Storage: 51.8 GB → 5.2 GB (10× reduction)
- Speed: 8.3 μs → 0.4-0.6 μs (14-21× faster)
- Accuracy: 85-90% → 92-95% (+7-10%)
- **90% of potential gains with minimal rewrite**

**Phase 2 (1-2 months, OPTIONAL)**: Template matching
- Storage: 5.2 GB → 3.1 GB (additional 1.7× reduction)
- Accuracy: 92-95% → 92-97% (+2% on repetitive regions)

**Phase 3 (2-3 months, RARELY NEEDED)**: C++ core
- Speed: 0.4-0.6 μs → 0.15-0.2 μs (additional 2-3× faster)
- Only if memory safety or batch queries critical

**Recommendation**: **Start with Phase 1**, profile, and re-evaluate. You may not need Phase 2/3!

**Key Principle**: ALL optimizations are LOSSLESS - they preserve the full accumulated signal required for lens confidence trajectory analysis.

---

**Status**: Ready to implement
**Start**: Week 1 (2-bit packing)
**Timeline**: 2-4 weeks for Phase 1, re-evaluate after
**Expected Outcome**: 10× storage + 20× speed + 10% accuracy with ZERO information loss

**Last Updated**: November 21, 2025
**Version**: 1.0
