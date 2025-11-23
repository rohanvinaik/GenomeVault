# HDC Experimentation: Split Binary Genomic Encoding

**Research Area:** Hyperdimensional Computing for Genomic Compression
**Focus:** Split Binary Quantization + Structural Motif Lens Library
**Last Updated:** 2025-11-21

---

## Quick Start

### 1. Build Lens Library (One-Time Setup)

```bash
cd genomevault/hdv_validation/hdc_experimentation

python encoders/build_lens_library.py \
    --reference /path/to/consensus.fa \
    --output output/lens_library.h5 \
    --D 5120 --N 1024 --seed 42
```

**Output:** `lens_library.h5` (~5-10 MB)
**Reuse:** Same library works for ANY human genome

### 2. Encode Genome

```bash
python encoders/encode_3bank_split_architecture.py
```

**Parameters:**
- D = 5,120 (dimension)
- N = 1,024 (chunk size)
- SNR = D/N = 5.0
- Split binary quantization (6 banks)

**Output:** `output/encoded_genome_6banks_split_binary.h5` (~3 GB for chr22)

### 3. Query with Lens-Aware Decoder

```python
from decoders.lens_aware_decoder import LensLibrary, LensAwareDecoder
import numpy as np

# Load lens library
lens_library = LensLibrary.load('output/lens_library.h5')

# Initialize decoder
decoder = LensAwareDecoder(
    encoded_h5_path='output/encoded_genome_6banks_split_binary.h5',
    lens_library=lens_library,
    use_magnitude_weighting=True,
    lens_alpha=0.3
)

# Generate position codebook (must match encoder)
np.random.seed(42)
position_codebook = np.random.choice([-1, 1], size=(1024, 5120)).astype(np.int8)

# Query position
nucleotide, confidence, texture, lens = decoder.decode_position(
    chrom='chr1',
    pos=10000,
    position_codebook=position_codebook
)

print(f"Position chr1:10000 = {nucleotide} (confidence: {confidence:.2%})")
print(f"Texture: {texture}, Lens: {lens}")
```

---

## Directory Structure

```
hdc_experimentation/
├── README.md                           # This file
│
├── encoders/                           # Encoding & preprocessing
│   ├── encode_3bank_split_architecture.py  # Main genome encoder
│   └── build_lens_library.py               # Lens library builder
│
├── decoders/                           # Decoding & querying
│   └── lens_aware_decoder.py               # Lens-aware decoder with ZCR
│
├── quantization/                       # Quantization experiments
│   ├── split_binary_quantizer.py
│   └── test_chunked_rle.py
│
├── docs/                               # Documentation
│   ├── README.md                           # Docs navigation
│   ├── theory/                             # Research logs & theory
│   │   ├── EXPERIMENTAL_DATA_COLLECTION.md    # Primary research log ⭐
│   │   ├── STRUCTURAL_MOTIF_LENS_LIBRARY.md   # Lens library design ⭐
│   │   ├── SPLIT_BANK_ARCHITECTURE.md
│   │   └── SPLIT_BINARY_ARCHITECTURE_ISSUE_ANALYSIS.md
│   ├── reports/                            # Summaries & quick-start
│   └── results/                            # Raw experimental data (JSON)
│
├── output/                             # Generated files
│   ├── encoded_genome_6banks_split_binary.h5   # Encoded genome
│   ├── lens_library.h5                         # Lens library
│   └── encoding_optimized_D5120_N1024.log      # Encoding log
│
├── demo_lens_decoder.py                # Demo script
└── validate_split_binary.py            # Validation script
```

---

## Key Innovations

### 1. Split Binary Quantization

Ternary {-1, 0, +1} → 6-bank binary {0, 1}:
- **Bank 0:** A detector (Hydrophobic +)
- **Bank 1:** T detector (Hydrophobic -)
- **Bank 2:** G detector (Major groove +)
- **Bank 3:** C detector (Major groove -)
- **Bank 4:** Hinge flexibility (Y→R steps)
- **Bank 5:** Hinge flexibility (R→Y steps)

**SNR Gain:** √2 (41%) via noise variance reduction

### 2. Structural Motif Lens Library

**Texture Classification (ZCR-based):**
- HOMOPOLYMER: ZCR < 0.05 (Poly-A/T runs)
- ALTERNATING: ZCR > 0.8 (TATA boxes)
- CPG_LIKE: High magnitude + variance
- ALU_LIKE: Moderate (GC-rich + A-tail)
- COMPLEX_CODING: High variance, no pattern

**Lenses:**
- ALU_YI (11% prevalence)
- CPG_ISLAND (1%)
- TATA_BOX (0.1%)
- POLY_A (2%)
- L1_LINE (17%)
- TELOMERIC (<0.01%)
- CAG_REPEAT (<0.01%)

**Decoding Pipeline:**
1. Texture classification (Bank 2 ZCR + magnitude + variance)
2. Lens selection (match texture → best lens)
3. Lens overlay (0.3 alpha blending)
4. Similarity computation (dot products)
5. **LINEAR** magnitude weighting (compositional prior)
6. Final decoding (argmax)

### 3. Optimizations

**ZCR vs FFT:**
- O(N) instead of O(N log N)
- Perfect for binary Purine/Pyrimidine signals
- ~100× less computation

**Linear Magnitude Weighting:**
- Applies Bayesian compositional prior
- Preserves signal for rare nucleotides
- Not squared (0.2 vs 0.04 = 5× difference)

---

## Performance

| Metric | Value |
|--------|-------|
| **Encoding speed** | ~2-3 hours (chr22, 51 Mbp) |
| **Storage** | ~3 GB (chr22) = 60 bytes/bp |
| **Query overhead** | ~0.1% (lens + magnitude) |
| **Expected accuracy** | +5-10% overall, +10-15% uncertain positions |

**Storage Scaling:** `genome_size × (D/N)`
**SNR:** D/N = 5,120/1,024 = 5.0

---

## Usage Examples

### Build Lens Library

```bash
python encoders/build_lens_library.py \
    --reference data/consensus.fa \
    --output output/lens_library.h5 \
    --D 5120 --N 1024
```

### Encode Genome

```bash
python encoders/encode_3bank_split_architecture.py
```

### Demo

```bash
python demo_lens_decoder.py

# Ablation study
python demo_lens_decoder.py --compare
```

### Validate

```bash
python validate_split_binary.py --sample-size 10000 --seed 42
```

---

## Research Context

**See:** `docs/theory/EXPERIMENTAL_DATA_COLLECTION.md`

**Core Contributions:**
1. Split binary quantization for genomic HDC
2. ZCR-based texture classification
3. Structural motif lens library
4. Linear magnitude-based compositional weighting
5. Genomic Monty Hall framework
6. SNR amplification via dimensionality

**Publications:** TBD

---

## Contact

For questions:
- **Experimental:** See `EXPERIMENTAL_DATA_COLLECTION.md`
- **Implementation:** See this README
- **Architecture:** See `STRUCTURAL_MOTIF_LENS_LIBRARY.md`

---

**Last Updated:** 2025-11-21
**Maintained by:** GenomeVault research team
