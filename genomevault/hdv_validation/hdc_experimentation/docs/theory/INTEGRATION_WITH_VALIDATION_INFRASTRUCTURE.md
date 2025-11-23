# Integration with GenomeVault Validation Infrastructure

**Date**: November 21, 2025
**Status**: ✅ Complete

---

## Overview

The 3-ternary bank HDC architecture is now fully integrated with GenomeVault's existing validation infrastructure, reusing all the complex ground truth alignment, statistical reporting, and logging systems.

## Key Integration Points

### 1. Validation Infrastructure (`genomevault/hdv_validation/`)

**Existing Components (Reused)**:
- `validation_utils.py` - Shared utilities for all validation testing
  - `load_gdiff()` - Load differential encoding ground truth
  - `sample_test_positions()` - Sample genomic positions with stratification
  - `get_ground_truth()` - Align ground truth from GDiff + BAM
  - `save_results()` - Save validation results to JSON
  - `compute_confusion_matrix()` - Compute per-nucleotide confusion matrix

- `compare_quantizations.py` - Main comparison framework
  - Supports multiple quantization modes (float32, int8, int4, binary)
  - Handles 5-lens architecture (AT, GC, PuPy, AmKe, StWk)
  - Comprehensive statistical reporting
  - BED file generation for UCSC collision testing
  - Adaptive threshold correction with signatures

**New Component (3-Ternary)**:
- `query_engine_3ternary.py` - 3-ternary bank query engine
  - **Class**: `PreEncoded3TernaryHDV` - Compatible with existing validation workflow
  - **Function**: `run_3ternary_validation()` - Uses existing validation infrastructure
  - **Integration**: Imports from `validation_utils` for ground truth alignment

---

## Architecture Comparison

### 5-Lens Architecture (Existing)

**Storage Format**:
```python
# HDF5: encoded_genome_5lenses_3d.h5
shape = (chunks, 5, D)  # 5 lenses
lenses = ['AT', 'GC', 'PuPy', 'AmKe', 'StWk']
```

**Lens Definitions**:
- **AT**: A=+1, T=-1, GC=0
- **GC**: G=+1, C=-1, AT=0
- **PuPy**: Purine (+1) vs Pyrimidine (-1)
- **AmKe**: Amino (+1) vs Keto (-1)
- **StWk**: Strong (+1) vs Weak (-1)

**Decoding**: Multi-lens voting with optimal per-lens thresholds

**Validation Class**: `PreEncodedMultiLensHDV` in `query_engine.py`

### 3-Ternary Architecture (New)

**Storage Format**:
```python
# HDF5: encoded_genome_3banks.h5
shape = (chunks, 3, D)  # 3 ternary banks
dtype = np.int8  # {-1, 0, +1}
```

**Bank Definitions**:
- **Bank 1 (Hydrophobic)**: T=+1, A=-1, GC=0
- **Bank 2 (Major Groove)**: G=+1, C=-1, AT=0
- **Bank 3 (Hinge)**: YR=+1, RY=-1, neutral=0

**Decoding**:
1. ZCR-based texture classification (Bank 3)
2. Lens library selection (optional)
3. Lens overlay (0.3 alpha)
4. Similarity computation
5. LINEAR magnitude weighting
6. Genomic Monty Hall cross-validation

**Validation Class**: `PreEncoded3TernaryHDV` in `query_engine_3ternary.py`

---

## Usage: 3-Ternary Validation

### Standalone Validation

```bash
cd /Users/rohanvinaik/genomevault

python3 -m genomevault.hdv_validation.query_engine_3ternary \
    --encoded-h5 genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5 \
    --gdiff data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz \
    --lens-library genomevault/hdv_validation/hdc_experimentation/output/lens_library.h5 \
    --sample-size 1000 \
    --lens-alpha 0.3 \
    --seed 42
```

### Integration with Existing Validation Utils

```python
from genomevault.hdv_validation.query_engine_3ternary import (
    PreEncoded3TernaryHDV,
    run_3ternary_validation
)
from genomevault.hdv_validation.validation_utils import (
    load_gdiff,
    sample_test_positions,
    get_ground_truth,
    save_results
)

# Load ground truth
gdiff, variant_index = load_gdiff("data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz")

# Initialize 3-ternary query engine
query_engine = PreEncoded3TernaryHDV(
    hdf5_path="genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5",
    lens_library=lens_library,  # Optional
    use_magnitude_weighting=True,
    lens_alpha=0.3,
    D=5120,
    N=1024,
    seed=42
)

# Sample test positions (genome-wide stratified sampling)
test_positions, high_n_set = sample_test_positions(
    chunk_keys=query_engine.chunk_keys,
    validated_n_positions=[],
    sample_size=1000,
    seed=42
)

# Validate each position
for chrom, pos in test_positions:
    # Get ground truth using existing infrastructure
    ground_truth, guide_idx, has_n = get_ground_truth(
        chrom, pos, variant_index, exp_bam, region_map
    )

    # Query with 3-ternary architecture
    prediction, confidence, texture, lens_name = query_engine.query_position(chrom, pos)

    # Compare prediction vs ground_truth
    is_correct = (prediction == ground_truth)
```

---

## Shared Infrastructure Components

### Ground Truth Alignment (`validation_utils.get_ground_truth()`)

**Complexity Handled**:
1. Variant positions (GDiff differential_variants)
2. Non-variant positions (experimental BAM pileup)
3. N positions (no experimental coverage)
4. Guide index lookup (region_guide_map)

**Usage**:
```python
ground_truth, guide_idx, has_n = get_ground_truth(
    chrom='chr1',
    pos=10000,
    variant_index=variant_index,
    exp_bam=exp_bam,
    region_map=gdiff['region_guide_map']
)
```

**Returns**:
- `ground_truth`: 'A', 'T', 'G', 'C', or 'N' (no coverage)
- `guide_idx`: Which guide reference was used (0-11)
- `has_n`: Boolean, True if experimental data had no coverage

### Statistical Reporting (`validation_utils.compute_confusion_matrix()`)

**Per-Nucleotide Confusion Matrix**:
```python
confusion = compute_confusion_matrix(pred_list, truth_list)

# Result format:
{
    'A': {'A': 245, 'T': 3, 'G': 2, 'C': 0},
    'T': {'A': 1, 'T': 248, 'G': 0, 'C': 1},
    'G': {'A': 2, 'T': 0, 'G': 246, 'C': 2},
    'C': {'A': 0, 'T': 1, 'G': 1, 'C': 248}
}
```

### Results Persistence (`validation_utils.save_results()`)

**JSON Format**:
```json
{
    "architecture": "3-ternary banks",
    "overall": {
        "accuracy": 0.9842,
        "correct": 984,
        "total": 1000
    },
    "per_nucleotide": {
        "A": {"accuracy": 0.98, "correct": 245, "total": 250},
        "T": {"accuracy": 0.992, "correct": 248, "total": 250},
        "G": {"accuracy": 0.984, "correct": 246, "total": 250},
        "C": {"accuracy": 0.992, "correct": 248, "total": 250}
    },
    "texture_distribution": {
        "HOMOPOLYMER": 120,
        "ALTERNATING": 80,
        "CPG_LIKE": 45,
        "ALU_LIKE": 110,
        "COMPLEX_CODING": 645
    },
    "lens_usage": {
        "ALU_YI": 110,
        "CPG_ISLAND": 45,
        "TATA_BOX": 15,
        "POLY_A": 30
    },
    "confusion_matrix": {...}
}
```

---

## Comparison with 5-Lens Architecture

### Conceptual Differences

| Aspect | 5-Lens (Existing) | 3-Ternary (New) |
|--------|-------------------|-----------------|
| **Design Philosophy** | Complementary biophysical properties | Direct chemical bank encoding |
| **Lens Count** | 5 lenses | 3 banks (+ optional structural motif lenses) |
| **Quantization** | Binary positive/negative per lens | Ternary {-1, 0, +1} per bank |
| **Voting** | Multi-lens majority voting | Magnitude-weighted Monty Hall |
| **Texture** | Implicit in voting patterns | Explicit ZCR classification (Bank 3) |
| **Compositional Priors** | Via lens thresholds | Via LINEAR magnitude weighting |
| **Reconstruction Overhead** | None | None (direct ternary) |

### Performance Implications

**5-Lens Advantages**:
- ✅ Mature validation infrastructure
- ✅ Optimal per-lens thresholds empirically tuned
- ✅ Signature-based error correction (99.69% accuracy)
- ✅ Comprehensive statistical reporting
- ✅ BED file generation for UCSC collision testing

**3-Ternary Advantages**:
- ✅ 50% less encoding compute (3 ops vs 5-6 ops)
- ✅ Explicit texture classification (ZCR on Bank 3)
- ✅ Structural motif lens library (optional enhancement)
- ✅ Natural alignment with Genomic Monty Hall framework
- ✅ LINEAR magnitude weighting (more interpretable)

---

## Next Steps

### 1. Encoder Completion
- **Status**: Running (~119 minutes elapsed, ETA unknown)
- **Output**: `encoded_genome_3banks.h5` (~48.2 GB for full genome)
- **Monitor**: `tail -f genomevault/hdv_validation/hdc_experimentation/output/encoding_log.txt`

### 2. Lens Library Generation
```bash
python genomevault/hdv_validation/hdc_experimentation/encoders/build_lens_library.py \
    --reference data/consensus.fa \
    --output genomevault/hdv_validation/hdc_experimentation/output/lens_library.h5 \
    --D 5120 --N 1024 --seed 42
```

### 3. Run Validation
```bash
python3 -m genomevault.hdv_validation.query_engine_3ternary \
    --encoded-h5 genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5 \
    --gdiff data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz \
    --lens-library genomevault/hdv_validation/hdc_experimentation/output/lens_library.h5 \
    --sample-size 10000 \
    --seed 42
```

### 4. Ablation Study

Test configurations:
1. **Baseline**: No lens library, no magnitude weighting
2. **Lens-only**: Lens library, no magnitude weighting
3. **Magnitude-only**: No lens library, LINEAR magnitude weighting
4. **Full System**: Lens library + magnitude weighting

### 5. Compare with 5-Lens

Once 3-ternary validation completes:
- Compare accuracy at same sample size (10k positions)
- Analyze texture classification effectiveness
- Evaluate lens library contribution
- Measure query speed (2× faster expected due to D=5120 vs D=10000)

---

## File Reference

### 3-Ternary Architecture
- **Query Engine**: `genomevault/hdv_validation/query_engine_3ternary.py`
- **Decoder**: `genomevault/hdv_validation/hdc_experimentation/decoders/lens_aware_decoder_CORRECTED_3TERNARY.py`
- **Encoder**: `genomevault/hdv_validation/hdc_experimentation/encoders/encode_3bank_split_architecture.py`
- **Lens Builder**: `genomevault/hdv_validation/hdc_experimentation/encoders/build_lens_library.py`
- **Theory Docs**: `genomevault/hdv_validation/hdc_experimentation/docs/theory/STRUCTURAL_MOTIF_LENS_LIBRARY.md`
- **Alignment Summary**: `genomevault/hdv_validation/hdc_experimentation/docs/theory/LENS_DECODER_ALIGNMENT_SUMMARY.md`

### 5-Lens Architecture (Existing)
- **Query Engine**: `genomevault/hdv_validation/query_engine.py`
- **Comparison Framework**: `genomevault/hdv_validation/compare_quantizations.py`
- **Validation Utils**: `genomevault/hdv_validation/validation_utils.py`
- **Signature Correction**: `genomevault/hdv_validation/signature_correction.py`

---

## Summary

The 3-ternary architecture is now **fully integrated** with GenomeVault's validation infrastructure, reusing:
- ✅ Ground truth alignment (GDiff + BAM)
- ✅ Genome-wide stratified sampling
- ✅ Statistical reporting and confusion matrices
- ✅ JSON result persistence
- ✅ Logging and progress tracking

**No duplication** - all complex validation logic is shared between 5-lens and 3-ternary architectures.

**Ready for testing** - Once encoder completes, run standalone validation or integrate with `compare_quantizations.py`.

---

**Version**: 1.0 (Integration Complete)
**Last Updated**: November 21, 2025
