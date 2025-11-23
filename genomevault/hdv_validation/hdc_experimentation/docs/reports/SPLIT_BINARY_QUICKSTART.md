# Split Binary Architecture - Quick Start Guide

## Overview

The **Split Binary 6-Bank Architecture** converts ternary {-1, 0, +1} biophysical encodings into binary {0, 1} representations with √2 SNR improvement per complementary pair.

## Architecture

### 3-Bank Ternary (Input)
```
Bank 0: Hydrophobic {-1, 0, +1}  (A/T complementary pair)
Bank 1: MajorGroove {-1, 0, +1}  (G/C complementary pair)
Bank 2: Hinge {-1, 0, +1}        (Dinucleotide context)
```

### 6-Bank Split Binary (Output)
```
Bank 0: Hydrophobic_A {0, 1}     (A positions only)
Bank 1: Hydrophobic_T {0, 1}     (T positions only)
Bank 2: MajorGroove_G {0, 1}     (G positions only)
Bank 3: MajorGroove_C {0, 1}     (C positions only)
Bank 4: Hinge_pos {0, 1}         (Positive dinucleotide context)
Bank 5: Hinge_neg {0, 1}         (Negative dinucleotide context)
```

## Transformation Rules

```
Ternary → Binary Split
+1 → (1, 0)   positive bank active, negative bank inactive
-1 → (0, 1)   positive bank inactive, negative bank active
 0 → (0, 0)   both banks inactive (sparsity preserved!)
```

## Key Properties

### 1. Within-Lens Splitting
- Splits WITHIN complementary pairs (AT and GC)
- Preserves 4 orthogonal nucleotide dimensions {A, T, G, C}
- Maintains biophysical lens interpretability

### 2. √2 SNR Improvement
- Original: N positions active → noise variance = N
- Split: N/2 positions active per bank → noise variance = N/2
- SNR improvement: √2 per complementary pair

### 3. Complementary Sparsity
- In GC-rich regions (80% GC):
  - AT banks: 4,096 positions → sparse, clean signal
  - GC banks: 16,384 positions → dense, noisy signal
- AT signals anchor GC discrimination via cross-channel grounding

### 4. Binary Math Efficiency
- Evidence AGAINST one nucleotide = Evidence FOR complement
- No mathematical transformation needed (zero computational cost)
- Cross-channel grounding via β and γ coupling

## Files

| File | Description |
|------|-------------|
| `split_binary_quantizer.py` | Converter: 3-bank ternary → 6-bank binary |
| `validate_split_binary.py` | Architecture validator (structure, sparsity, metadata) |
| `split_binary_decoder.py` | Two-stage nucleotide decoder (pair selection + sign determination) |
| `SPLIT_BANK_ARCHITECTURE.md` | Full architectural documentation with mathematical proofs |

## Usage

### 1. Convert Ternary to Split Binary

```bash
python3 genomevault/hdv_validation/hdc_experimentation/quantization/split_binary_quantizer.py
```

**Input:**  `encoded_genome_3banks.h5` (932 MB, 327,655 chunks)
**Output:** `encoded_genome_6banks_split_binary.h5` (~450-500 MB compressed)

### 2. Validate Architecture

```bash
python3 genomevault/hdv_validation/hdc_experimentation/quantization/validate_split_binary.py
```

**Validates:**
- 6-bank structure with correct dimensions
- Bank naming and metadata
- Sparsity ratios (each binary bank should have ~half the sparsity of ternary)
- Within-lens splitting preserved

### 3. Decode Nucleotides

```bash
python3 genomevault/hdv_validation/hdc_experimentation/quantization/split_binary_decoder.py
```

**Decoding Algorithm (Two-Stage):**

**Stage 1: Pair Selection (Magnitude Comparison)**
```
|sim_AT| = max(sim_Hydro_A, sim_Hydro_T)
|sim_GC| = max(sim_Major_G, sim_Major_C)

Select pair with stronger signal
```

**Stage 2: Sign Determination**
```
If AT pair: argmax(sim_Hydro_A, sim_Hydro_T) → A or T
If GC pair: argmax(sim_Major_G, sim_Major_C) → G or C
```

## Expected Results

### Sparsity Analysis

**Ternary (3 banks):**
- Active bits per bank: ~7.44% (+1 or -1)
- Inactive bits per bank: ~92.56% (0)

**Split Binary (6 banks):**
- Active bits per bank: ~3.72% (1)
- Inactive bits per bank: ~96.28% (0)

**Sparsity Ratio:** Binary active / (Ternary active / 2) ≈ 1.0 ✓

### File Size

**Compression Efficiency:**
- Ternary: 3 banks × 327,655 chunks × 10,240 dimensions × 1 byte = 9.37 GB raw → 932 MB (10.1× compression)
- Binary: 6 banks × 327,655 chunks × 10,240 dimensions × 1 byte = 18.75 GB raw → ~450-500 MB (40-42× compression)

**Binary compresses better because:**
- Fewer unique values: {0,1} vs {-1,0,+1}
- Same 92% sparsity
- More regular run-length patterns
- Gzip loves binary data

## Why Within-Lens Splitting is Superior

### Information-Theoretic Requirement

To discriminate 4 states {A, T, G, C}, we need 4 orthogonal dimensions.

**Option A: Within-Lens Splitting (CHOSEN)**
```
6 banks → 4 orthogonal nucleotide dimensions
- Hydrophobic_A + Hydrophobic_T (AT pair isolation)
- MajorGroove_G + MajorGroove_C (GC pair isolation)
Result: Optimal for 4-way classification ✓
```

**Option B: Across-Lens Splitting (REJECTED)**
```
2 banks → Positive/Negative only
FAILS: Cannot tell if positive signal is from A or G!
Result: Collapses 4 states → 2 (loses discriminative power) ✗
```

### SNR Mathematics

**Option A: Within-Lens Splitting**
```
Ternary Hydrophobic (AT): 8,192 positions → Noise variance: 8,192
Split:
  Hydrophobic_A: 4,096 positions → Noise variance: 4,096 (HALF!)
  Hydrophobic_T: 4,096 positions → Noise variance: 4,096 (HALF!)

SNR improvement: √2 per vector ✓
```

**Option B: Across-Lens Splitting**
```
Still 8,192 positions in each bank
Noise variance unchanged
No SNR improvement ✗
```

### Complementary Sparsity

**Within-Lens:** In 80% GC regions, AT vectors are 2× cleaner (1,024 vs 4,096 positions)

**Across-Lens:** No sparsity benefit (all nucleotides contribute to both banks)

## Next Steps

1. **Accuracy Testing:** Compare decoding accuracy with ternary baseline
2. **Performance Benchmarking:** Measure query throughput (queries/second)
3. **Hardware Optimization:** Test binary operations on specialized hardware
4. **Integration:** Connect to main GenomeVault pipeline

## References

- Main architecture: `SPLIT_BANK_ARCHITECTURE.md`
- Information theory section: See "Binary Noise-to-Signal Conversion"
- Cross-channel grounding: See "Complementary Sparsity and Cross-Channel Grounding"

---

**Last Updated:** November 20, 2025
**Architecture Version:** Split Binary v1.0
**Status:** ✅ Implementation Complete, Validation Pending
