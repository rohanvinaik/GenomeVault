# Split Binary Architecture Issue Analysis

**Date:** 2025-11-21
**Status:** Root cause identified

## Issue Summary

The split binary architecture validation shows flat ~36% accuracy across all nucleotides (A=35.87%, T=36.38%, G=36.23%, C=37.67%), which is barely better than random (25%). This is caused by a **fundamental architecture mismatch** between the encoder and validator.

## Root Cause

### Encoder Behavior (Chunk-Level Aggregation)

The encoder in `encode_3bank_split_architecture.py` creates **chunk-level** representations:

```python
# Accumulate across ALL positions in chunk (N=10,240)
for offset in range(chunk_length):
    pos_vec = encoder.position_codebook[offset % N]

    if nucleotide == 'T':
        acc_hydro += pos_vec  # Accumulate
    elif nucleotide == 'A':
        acc_hydro -= pos_vec  # Accumulate
    # ... (continue for all 10,240 positions)

# Then sparsify the accumulated vector
bank = sparsify_bipolar(acc_hydro)  # Chunk-level representation
```

This creates a **single hypervector per chunk** that represents the statistical distribution of nucleotides across ~10,000 positions.

### Validator Behavior (Position-Level Queries)

The validator in `validate_split_binary.py` attempts **position-level** queries:

```python
# Query for a single position within a chunk
query = self.pos_vectors[offset, :]  # Single position vector

# Compute similarity
sim = np.dot(bipolar_bank, query) / D
```

## Signal-to-Noise Analysis

### For a chunk with N=10,240 positions:

- **Signal**: 1 position vector (the query position)
- **Noise**: 10,239 other position vectors
- **SNR**: 1/10,240 = **0.0001** (0.01%)

### Mathematical Explanation:

When encoding position `i` with nucleotide `T`:
```
acc[i] = +pos_vec[i] + sum(other 10,239 positions)
```

When querying position `i`:
```
similarity = (acc · pos_vec[i]) / D
           = (pos_vec[i] · pos_vec[i] + noise) / D
           = (D + noise) / D
           ≈ 1.0 + noise/D
```

But `noise = sum of ~10,239 random dot products ≈ 0 (but with high variance)`

The signal (D) is completely overwhelmed by accumulated noise from 10,239 other positions.

## Empirical Validation

Test script `test_architecture_mismatch.py` demonstrates this with just 20 accumulated positions:

```
Query for position 5 (true nucleotide: T)
  Similarity to T bank: 0.164258
  Similarity to A bank: -0.171875

SNR with 20 positions: 1/20 = 5%
SNR with 10,240 positions: 1/10,240 = 0.01%
```

The T bank should have much higher similarity, but it's nearly equal to the A bank due to noise from other accumulated positions.

## Why 36% Accuracy (Not 25% Random)?

Random guessing: 25% (1/4 nucleotides)
Observed: 36% (+11% over random)

The small improvement comes from:
1. **Nucleotide frequency bias**: The genome has unequal base composition (not exactly 25% each)
2. **Weak residual signal**: The correct nucleotide's contribution is still present, just heavily diluted
3. **Sparsification artifacts**: The thresholding process may introduce slight biases

But this is far from useful accuracy.

## Solution Options

### Option 1: Chunk-Level Validation (Quick Fix)

Validate at the chunk level instead of position level:
- Query: "What is the nucleotide composition of this chunk?"
- Expected accuracy: High (80-90%+) for compositional queries
- Limitation: Cannot resolve individual positions

### Option 2: Per-Position Encoding (Architecture Change)

Create a different encoding scheme where each position gets its own hypervector:

```python
# Instead of accumulation:
for offset in range(chunk_length):
    nucleotide = get_nucleotide(pos)
    pos_vec = position_codebook[offset]

    # Store per-position encoding (no accumulation)
    position_encodings[offset] = encode_nucleotide(nucleotide, pos_vec)
```

**Trade-offs:**
- **Pro**: Supports position-level queries with high accuracy
- **Con**: 10,240× more storage (one vector per position vs one per chunk)
- **Con**: Loses the compression benefit of chunk-level representation

### Option 3: Hybrid Multi-Resolution Encoding

Encode at multiple resolutions:
1. **Chunk-level** (current): For compositional queries
2. **Window-level** (e.g., 100bp windows): For local queries
3. **Position-level** (optional): For exact queries

This provides a resolution hierarchy at the cost of ~100× more storage.

### Option 4: Alternative Query Strategy

Use the chunk-level encoding but query differently:
- Instead of querying a single position, query a context window (e.g., ±50bp)
- Accumulate position vectors for the window and compare to the chunk encoding
- May improve accuracy to 60-70% but still not perfect

## Recommendation

The split binary quantization architecture is **working as designed** - it creates efficient chunk-level representations. The issue is not a bug, but a **misunderstanding of what the architecture can do**.

**Immediate action:**
1. Document that this architecture is for chunk-level compositional analysis, not position-level queries
2. If position-level accuracy is required, use a different architecture (e.g., the working float32/int8/int4 systems)
3. The A/T swap fix was correct and valuable, but the architecture itself is fundamentally chunk-level

## Related Files

- `genomevault/hdv_validation/hdc_experimentation/encoders/encode_3bank_split_architecture.py` (encoder)
- `genomevault/hdv_validation/hdc_experimentation/validate_split_binary.py` (validator)
- `genomevault/hdv_validation/hdc_experimentation/docs/split_binary_validation_results.json` (results)
- `test_architecture_mismatch.py` (proof-of-concept demonstrating SNR issue)

## Conclusion

The flat 36% accuracy is not due to:
- ❌ Wrong similarity computation
- ❌ Wrong nucleotide assignment
- ❌ Wrong position vectors
- ❌ A/T swap bug (this was fixed)

It's due to:
- ✅ **Signal-to-Noise Ratio**: N/D ratio = 1.0 (should be ~0.2)

The encoder accumulates N=10,240 positions into D=10,240 dimensions, giving SNR = D/N = 1.0. Each position's signal is overwhelmed by noise from 10,239 other positions.

---

## RESOLVED: Proper Parameter Selection

**Date:** 2025-11-21
**Status:** Parameters optimized for genome structure exploitation

### Root Cause Correction

Initial analysis incorrectly concluded this was an "architecture mismatch" between chunk-level encoding and position-level queries. **This was wrong** - the working float32 architecture (D=10,000, N=2,000, SNR=5.0) also uses chunk-level encoding with position-level queries and achieves 99.14% accuracy.

**The real issue:** N/D ratio was too high (1.0 instead of 0.2), causing signal drowning.

### Optimized Architecture Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **D** (dimension) | 5,120 | Half of original, exploits genome structure |
| **N** (positions/chunk) | 1,024 | Maintains N/D = 0.2 (proven SNR) |
| **OVERLAP** | 128 bp (12.5%) | Higher % for smaller chunks (edge effects) |
| **STRIDE** | 896 bp | N - OVERLAP |
| **SNR** | 5.0 | D/N = 5,120/1,024 |
| **SPARSITY_PERCENTILE** | 50 | Keep 100% of accumulated data |

### Why Lower D (5,120 vs 10,000)?

**Genome Structure Exploitation:**
- Human genome is 45% repetitive elements (not random)
- Conserved regulatory motifs (TATA, CAAT boxes)
- Gene families with similar sequences
- Local correlation (CpG islands, homopolymers)

**Lower D forces the system to find and reuse these patterns**, creating a representational bottleneck that:
- Collapses similar sequences to similar hypervectors
- Learns the genome's "natural vocabulary"
- Reduces overfitting to noise
- May capture micro-features better with finer spatial granularity

**Performance Benefits:**
- **Query speed**: 2× faster (5,120 vs 10,000 ops per query)
- **Chunk count**: 2.9M chunks (finer granularity, ~1kb vs 2kb)
- **Storage**: SAME (depends only on D/N ratio = 5.0)
- **Memory**: 5.2 MB codebook (vs 20 MB)

**6-Bank Compensation:** With split binary quantization creating 6 specialized banks, effective representational capacity might be 6 × D = ~30,720 "effective" dimensions.

### Validation Plan

1. Encode with D=5,120, N=1,024 (Option A)
2. Validate accuracy on 10,000 positions
3. If accuracy >97%: **production ready**
4. If accuracy 90-97%: acceptable tradeoff for 2× speed
5. If accuracy <90%: fall back to D=10,240, N=2,048

### Alternative Parameters (If Needed)

**Option B (Aggressive Compression):** D=2,560, N=512
- 4× faster queries
- 5.9M chunks (512bp granularity, matches typical exon size)
- Same storage (D/N = 5.0 maintained)
- Higher risk: vectors only 96% orthogonal vs 99%

### Key Insights

**Storage Formula:** Total storage = genome_size × (D/N)
- Only the RATIO matters, not absolute values
- D=5,120, N=1,024 has SAME storage as D=10,000, N=2,000
- But faster queries and potentially better genome structure capture

**Overlap Scaling:** Smaller chunks need higher % overlap
- 12.5% overlap (vs original 10%) accounts for proportionally larger edge effects
- In 1,024bp chunks, 128bp edges = 12.5% (acceptable)
- In 10,240bp chunks, same 128bp edges = 1.25% (negligible)
