# Encoder Directory Status

**Last Updated:** November 23, 2025

## ⚠️ IMPORTANT: Production Encoder Location

The **current production encoder** is **NOT** in this directory. It is located at:

```
genomevault/hdv_validation/hdc_experimentation/encoders/encode_3bank_split_architecture.py
```

This encoder uses `ComplementaryPairEncoder` from:
```
genomevault/hypervector_transform/complementary_pair_encoder.py
```

## Files in This Directory

### REFERENCE (Keep for Historical Value)

These files are retained as reference implementations but are **not used in production**:

- **`biophysical_signature_encoder.py`** (18 KB, Nov 17, 2025)
  - Contains valuable biophysical logic that informed the 3-bank design
  - Multi-lens encoder with independent chemical property projections
  - Each lens has its own dimension, chunk size, and overlap parameters
  - Keep as reference for architectural decisions

- **`encode_genome_5lenses_CORRECT.py`** (6.2 KB, Nov 17, 2025)
  - Early multi-lens prototype
  - Demonstrates 5-lens encoding approach
  - Superseded by 3-bank split architecture

## Archived Files (Moved to archive/deprecated_encoders_20251123/)

### Lightning Variants (Experimental Quantizations)
Moved to: `archive/deprecated_encoders_20251123/lightning_variants/`

- `lightning_genome_hdc.py` - Base lightning encoder
- `int4_lightning_hdc.py` - Int4 quantization variant
- `int8_lightning_hdc.py` - Int8 quantization variant
- `binary_lightning_hdc.py` - Binary quantization variant

**Status:** All superseded by split_ternary_quantizer.py

### Error Profiling Scripts (Experimental Analysis)
Moved to: `archive/deprecated_encoders_20251123/error_profiling/`

- `binary_lightning_hdc_error_profiling.py`
- `bipolar_1bit_hdc_error_profiling.py`
- `float32_streaming_hdc_error_profiling.py`
- `int4_lightning_hdc_error_profiling.py`
- `int8_lightning_hdc_error_profiling.py`
- `onehot_hdc_error_profiling.py`

**Status:** Kept as diagnostic tools, not in production pipeline

### Old Whole-Genome Encoders
Moved to: `archive/deprecated_encoders_20251123/old_whole_genome/`

- `encode_whole_genome_hdv.py` (Nov 14, 2025)
- `encode_whole_genome_hdv_streaming.py` (Nov 14, 2025)

**Status:** Superseded by encode_3bank_split_architecture.py

### Old Multi-Lens Encoders
Moved to: `archive/deprecated_encoders_20251123/old_multi_lens/`

- `encode_genome_all_lenses.py` (Nov 17, 2025)
- `magnitude_direction_hdc.py` (Nov 17, 2025)

**Status:** Superseded by 3-bank split architecture

## Production Pipeline (Current, Nov 2025)

### Complete End-to-End Flow:

```
1. DATA GENERATION (k=12 Privacy Pipeline)
   Script: scripts/run_enhanced_privacy_pipeline_optimized.py
   Output: data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz (29 MB)

2. 3-BANK HDC ENCODING
   Script: genomevault/hdv_validation/hdc_experimentation/encoders/encode_3bank_split_architecture.py
   Core:   genomevault/hypervector_transform/complementary_pair_encoder.py
   Output: genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_3banks.h5 (5.3 GB)

   Architecture: 3 biophysical banks
   - Bank 1: Hydrophobic (T vs A, transparent to G/C)
   - Bank 2: Major Groove (G vs C, transparent to A/T)
   - Bank 3: Hinge (Y-R vs R-Y structural flexibility)

   Parameters: N=1024, D=5120, Overlap=128

3. SPLIT TERNARY QUANTIZATION
   Script: genomevault/hdv_validation/hdc_experimentation/quantization/split_ternary_quantizer.py
   Output: genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_ternary.h5 (6.1 GB)

   Architecture: 6 ternary banks (two orthogonal 3D vectors)
   - Vector 1 (GC-dominant): [AT=0, GC, Hinge]
   - Vector 2 (AT-dominant): [AT, GC=0, Hinge]

4. QUERY ENGINE
   Script: genomevault/hdv_validation/hdc_experimentation/query/lens_aware_simd_query_engine.py

   Features:
   - SIMD-optimized dot products (1.92 μs median)
   - Lens-aware decoding with texture classification
   - Smart binary search for optimal lens confidence
```

## Architecture Evolution Timeline

| Date | Milestone |
|------|-----------|
| Nov 14, 2025 | Old whole-genome encoders (deprecated) |
| Nov 15, 2025 | Lightning quantization experiments |
| Nov 17, 2025 | Biophysical and multi-lens prototypes |
| Nov 19, 2025 | Error profiling analysis |
| **Nov 21, 2025** | **✅ 3-bank split architecture established** |
| **Nov 22, 2025** | **✅ Split ternary quantization** |
| **Nov 23, 2025** | **✅ Lens-aware SIMD query engine active** |

## What Works (Production)

- ✅ 3-bank split architecture with biophysical transparency
- ✅ Split ternary quantization (6 banks, √2 SNR improvement)
- ✅ Sparse position codebook (1 dimension per position)
- ✅ N=1024, D=5120 (optimized for genome structure)
- ✅ Lens-aware SIMD queries (1.92 μs median)

## What's Deprecated

- ❌ Lightning variants (all quantization experiments)
- ❌ Old whole-genome encoders
- ⚠️ Error profiling scripts (kept as tools, not in pipeline)
- ⚠️ Multi-lens prototypes (superseded by 3-bank)

## Import Dependencies

The production encoder has minimal dependencies:

```python
# encode_3bank_split_architecture.py imports:
from genomevault.hypervector_transform.complementary_pair_encoder import ComplementaryPairEncoder

# complementary_pair_encoder.py imports:
# - None from genomevault (self-contained)
# - Uses: numpy, pysam, gzip
```

This isolation enables rapid experimentation in the `hdc_experimentation/` directory without breaking the main codebase.

## For More Information

See the complete pipeline documentation:
- `/Users/rohanvinaik/genomevault/CLAUDE.md` - Project overview
- `genomevault/hdv_validation/hdc_experimentation/README.md` - HDC experimentation guide
- `genomevault/hdv_validation/hdc_experimentation/docs/` - Detailed architecture docs
