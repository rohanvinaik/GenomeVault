# Binary Split Architecture Analysis

**Proposal**: Split "binary" quantization into two separate binary encodings (AT binary + GC binary), each using either 0/1 or -1/+1 encoding.

**Date**: November 19, 2025
**Author**: Claude Code (prompted by Rohan Vinaik)

---

## Current Architecture

### How Binary is Encoded Now

**Current System: Unified Bipolar with 5 Lenses**

```python
LENS_DEFINITIONS = {
    'AT':   { 'positive': {'A'}, 'negative': {'T'} },
    'GC':   { 'positive': {'G'}, 'negative': {'C'} },
    'PuPy': { 'positive': {'A', 'G'}, 'negative': {'T', 'C'} },  # Purine vs Pyrimidine
    'AmKe': { 'positive': {'A', 'C'}, 'negative': {'G', 'T'} },  # Amino vs Keto
    'StWk': { 'positive': {'G', 'C'}, 'negative': {'A', 'T'} }   # Strong vs Weak
}

# Encoding per lens (at position i):
if nucleotide in positive_set:
    bipolar[i] = +1.0
elif nucleotide in negative_set:
    bipolar[i] = -1.0
else:  # 'N' or ambiguous
    bipolar[i] = 0.0

# Binary quantization (current):
binary_vector = np.sign(float32_vector)  # → {-1, 0, +1}
```

**Key Properties:**
- **Values**: {-1, 0, +1} (bipolar with zero)
- **Storage**: ~70 GB (3 values, gzip compressed)
- **All 5 lenses**: AT, GC, PuPy, AmKe, StWk encoded separately
- **Accuracy**: ~98.4% (with multi-lens voting)

---

## Proposed Architecture: Binary Split

### Option A: Dual Bipolar (-1/+1)

**Two separate encodings, each bipolar:**

```python
# AT Binary System
AT_binary = {
    'A': +1,
    'T': -1,
    'G': 0,   # Not part of AT system
    'C': 0,
    'N': 0
}

# GC Binary System
GC_binary = {
    'G': +1,
    'C': -1,
    'A': 0,   # Not part of GC system
    'T': 0,
    'N': 0
}
```

**Result**: 2 files × 70 GB ≈ **140 GB** (vs 70 GB current)

### Option B: Dual Unipolar (0/1)

**Two separate encodings, each unipolar:**

```python
# AT Binary System (unsigned)
AT_binary = {
    'A': 1,
    'T': 0,
    'G': 0,   # Not part of AT system
    'C': 0,
    'N': 0
}

# GC Binary System (unsigned)
GC_binary = {
    'G': 1,
    'C': 0,
    'A': 0,   # Not part of GC system
    'T': 0,
    'N': 0
}
```

**Result**: 2 files × 35 GB ≈ **70 GB** (only 2 values: 0/1)

### Option C: Four Unipolar Channels (One-Hot)

**Explicit one-hot encoding per nucleotide:**

```python
A_channel = {A: 1, else: 0}
T_channel = {T: 1, else: 0}
G_channel = {G: 1, else: 0}
C_channel = {C: 1, else: 0}
```

**Result**: 4 files × 35 GB ≈ **140 GB**

---

## Comparative Analysis

| Architecture | Files | Storage | Values | Biophysical Lenses | Redundancy | SIMD Friendly |
|--------------|-------|---------|--------|-------------------|------------|---------------|
| **Current (Unified 5-Lens)** | 1 | 70 GB | {-1,0,+1} | 5 (AT, GC, PuPy, AmKe, StWk) | High | No |
| **Option A (Dual Bipolar)** | 2 | 140 GB | {-1,0,+1} | 2 (AT, GC only) | Low | No |
| **Option B (Dual Unipolar)** | 2 | 70 GB | {0,1} | 2 (AT, GC only) | Low | **YES** |
| **Option C (One-Hot)** | 4 | 140 GB | {0,1} | 0 (direct nucleotides) | None | **YES** |

---

## Deep Dive: Advantages & Disadvantages

### Option A: Dual Bipolar (-1/+1)

#### ✅ Advantages:
1. **Maintains HDC semantics**: Bipolar vectors preserve dot product similarity nicely
2. **Separates base pairs**: AT and GC are chemically distinct (Watson-Crick pairing)
3. **Independent querying**: Can query AT and GC systems separately
4. **Reduced cross-talk**: No mixing between purine/pyrimidine properties

#### ❌ Disadvantages:
1. **2× storage** (140 GB vs 70 GB)
2. **Loses biophysical lenses**: No PuPy, AmKe, StWk (only AT + GC)
3. **Lower accuracy**: Current 98.4% relies on 5-lens voting; 2 lenses → ~85-90% accuracy
4. **More complex queries**: Need to coordinate two separate lookups
5. **Not SIMD-friendly**: {-1,0,+1} doesn't benefit from POPCNT/bitwise ops

### Option B: Dual Unipolar (0/1) ⭐ **RECOMMENDED**

#### ✅ Advantages:
1. **Same storage as current** (70 GB total)
2. **SIMD-friendly**: Binary 0/1 enables POPCNT, XOR, bitwise operations
3. **20-50× faster queries**: See "One-Hot Encoding Architecture" in theory doc
4. **Cache-aligned**: Can pack 8 positions into 1 byte (vs 3 bits for ternary)
5. **Separates base pairs**: AT and GC independent
6. **Simpler arithmetic**: Addition instead of weighted sums

#### ❌ Disadvantages:
1. **Loses biophysical lenses**: Only AT + GC (no PuPy, AmKe, StWk)
2. **Reduced redundancy**: 2 dimensions vs 5 → lower error correction
3. **Accuracy hit**: ~85-90% (2-lens voting) vs 98.4% (5-lens voting)
4. **Loses sign information**: Can't distinguish "opposite of A" from "not A"
5. **HDC semantics broken**: Unipolar vectors change similarity metric

### Option C: One-Hot (4 Channels) ⭐⭐ **BEST FOR SPEED**

#### ✅ Advantages:
1. **MAXIMUM speed**: 50-100× faster with SIMD
2. **Perfect reconstruction**: One bit per nucleotide, no ambiguity
3. **No lens voting needed**: Direct nucleotide lookup
4. **Cache-optimal**: 4 bits = 0.5 bytes per position
5. **Parallelizable**: 4 independent bitstreams

#### ❌ Disadvantages:
1. **2× storage** (140 GB)
2. **NO biophysical information**: Just raw nucleotides
3. **No error correction**: Single bit flip = wrong nucleotide
4. **Not HDC**: Abandons hyperdimensional computing entirely
5. **Privacy concerns**: Direct nucleotide encoding (less obfuscated)

---

## Implementation Impact on Current Pipeline

### 🚨 Breaking Changes

**If you adopt Option A or B, these modules BREAK:**

1. **Multi-lens voting** (`validate_multi_lens_with_theoretical.py`)
   - Currently: 5 lenses vote → nucleotide
   - New: Only 2 lenses (AT, GC) → reduces accuracy

2. **Biophysical recovery** (N-position prediction)
   - Currently: Uses PuPy, AmKe, StWk to predict sequencing failures
   - New: LOST (no PuPy/AmKe/StWk lenses)

3. **Validation suite** (`architecture_testing/`)
   - All 5-lens tests fail
   - Need new 2-lens validation tests

4. **Quantization files**
   - Currently: Single file per quantization mode
   - New: 2 files per mode (AT + GC)

### ✅ Compatible Modules

These would work unchanged:

1. **HDV dimension reduction** (10,000D → 6,000D)
2. **Streaming H5 access**
3. **Query infrastructure**
4. **Compression pipeline**

---

## My Recommendation: Hybrid Architecture

**Don't choose one—use BOTH systems in parallel!**

### Proposed Hybrid System:

```
System 1: Biophysical (Current)
├─ 5 lenses: AT, GC, PuPy, AmKe, StWk
├─ Quantization: float32, int8, int4, binary, ternary
├─ Storage: 70-281 GB
├─ Accuracy: 98.4%
└─ Use case: High-accuracy genomic analysis, error correction

System 2: Binary Split (New)
├─ 2 channels: AT binary (0/1), GC binary (0/1)
├─ Storage: 70 GB total
├─ Accuracy: 85-90% (2-lens voting)
├─ Speed: 20-50× faster (SIMD)
└─ Use case: Real-time queries, edge devices, rapid screening
```

### Why Hybrid?

1. **Use the right tool for the job**:
   - Biophysical system for deep analysis
   - Binary split for fast queries

2. **Benchmark comparison**:
   - Compare accuracy: 5-lens vs 2-lens
   - Compare speed: HDC float32 vs binary SIMD

3. **Storage is cheap** (70 GB extra for binary split)
4. **Validation flexibility**: Test both architectures

---

## Encoding Choice: 0/1 vs -1/+1

### When to use -1/+1 (Bipolar):
✅ HDC similarity metrics (dot product)
✅ Error correction (zero has meaning)
✅ Biophysical properties (opposite poles)
✅ Traditional hyperdimensional computing

### When to use 0/1 (Unipolar):
✅ **SIMD acceleration** (POPCNT, XOR, AND)
✅ **Cache efficiency** (bit-packing)
✅ **Query speed** (50× faster)
✅ **Simplicity** (easier to reason about)
✅ **Hardware-friendly** (GPUs, TPUs love 0/1)

### My Answer: **Use 0/1 for Binary Split**

**Rationale:**
- You're already sacrificing biophysical lenses (PuPy, AmKe, StWk)
- If you're going simpler, go ALL THE WAY → max speed with SIMD
- -1/+1 gives you nothing if you don't have 5 lenses for voting
- 0/1 unlocks hardware acceleration (POPCNT is 100× faster than multiply-add)

---

## Proposed Implementation Plan

### Phase 1: Create Binary Split Files (2-3 hours)

```python
# create_binary_split_files.py
def quantize_at_binary(float32_vector):
    """AT lens → {0, 1}"""
    # A → 1, T → 0, G/C/N → 0
    return (np.sign(float32_vector) > 0).astype(np.uint8)

def quantize_gc_binary(float32_vector):
    """GC lens → {0, 1}"""
    # G → 1, C → 0, A/T/N → 0
    return (np.sign(float32_vector) > 0).astype(np.uint8)

# Create:
# - encoded_genome_at_binary.h5 (~35 GB)
# - encoded_genome_gc_binary.h5 (~35 GB)
```

### Phase 2: Update Validation Suite (1 hour)

```python
# New validator: 2-lens voting
class BinarySplitHDV:
    def __init__(self, at_file, gc_file):
        self.at_system = load_h5(at_file)
        self.gc_system = load_h5(gc_file)

    def query_position(self, chrom, pos):
        at_vote = self.at_system.query(chrom, pos)  # 0 or 1
        gc_vote = self.gc_system.query(chrom, pos)  # 0 or 1

        # Decode:
        if at_vote == 1: return 'A'
        if at_vote == 0: return 'T'  # Wait, problem here...
```

**⚠️ ISSUE DISCOVERED**: Unipolar 0/1 loses directional information!

- AT bipolar: A=+1, T=-1 → two clear states
- AT unipolar: A=1, T=0 → but so is G, C, N!

**Solution**: You MUST use bipolar (-1/+1) OR use threshold:

```python
# AT lens with threshold
at_score = similarity(query_vector, at_codebook)
if at_score > 0.5:    # High confidence A
    return 'A'
elif at_score < -0.5:  # High confidence T
    return 'T'
else:
    # Fallback to GC lens
```

### Phase 3: Benchmark (30 min)

Compare:
- **5-lens float32**: 98.4% accuracy, 18 μs/query
- **5-lens binary**: 98.4% accuracy, 18 μs/query
- **2-lens binary split (bipolar)**: ??% accuracy, ?? μs/query
- **2-lens binary split (unipolar)**: ??% accuracy, ?? μs/query

---

## Final Verdict

### Should you do it? **YES, but as a parallel system, not a replacement.**

**Recommended Action Plan:**

1. **Keep existing 5-lens system** (don't delete anything)
2. **Create new binary split system** (AT + GC, dual bipolar -1/+1)
3. **Benchmark both**:
   - Accuracy: Does 2-lens voting work?
   - Speed: Is bipolar split faster than unified?
4. **Later**: Explore 0/1 unipolar with SIMD optimization

**Expected Results:**
- Accuracy drop: 98.4% → 85-92%
- Speed gain: Minimal (bipolar not SIMD-friendly)
- Storage: +70 GB

**If you want REAL speedup:**
- Need to implement SIMD kernels (POPCNT, AVX2)
- Use 0/1 encoding
- Write optimized C++/Rust query layer
- Potential: 50-100× faster

---

## Open Questions for Discussion

1. **Can we recover PuPy/AmKe/StWk from AT+GC?**
   - PuPy = f(AT, GC)?
   - Probably not (information loss)

2. **Is 2-lens voting sufficient?**
   - Need to benchmark
   - May be OK for high-quality samples

3. **SIMD implementation effort?**
   - ~2-3 weeks for optimized C++ layer
   - Worth it for 50× speedup?

4. **Privacy implications?**
   - 2 channels easier to attack than 5?
   - Less obfuscation

---

## Conclusion

Your intuition is **excellent**. Splitting binary into AT + GC systems is a valid architectural choice that trades:

- **Redundancy** (5 lenses → 2 lenses)
- **Accuracy** (98% → ~90%)
- **For potential speed gains** (if you use 0/1 + SIMD)

**My recommendation: Build it as an experiment, keep both systems, benchmark rigorously.**

The real win comes if you go full 0/1 unipolar + SIMD optimization. Otherwise, the 5-lens bipolar system is likely better (higher accuracy, same speed).

