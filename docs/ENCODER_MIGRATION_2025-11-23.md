# Encoder Migration & Cleanup - November 23, 2025

## Summary

Cleaned up deprecated encoder files and documented the production pipeline. Archived 13 deprecated encoder scripts while preserving the active 3-bank split architecture pipeline.

## What Was Archived

**Location:** `archive/deprecated_encoders_20251123/`

### Lightning Variants (4 files)
Experimental quantization approaches superseded by split_ternary_quantizer.py:
- `lightning_genome_hdc.py`
- `int4_lightning_hdc.py`
- `int8_lightning_hdc.py`
- `binary_lightning_hdc.py`

### Error Profiling Scripts (6 files)
Diagnostic tools kept for reference:
- `binary_lightning_hdc_error_profiling.py`
- `bipolar_1bit_hdc_error_profiling.py`
- `float32_streaming_hdc_error_profiling.py`
- `int4_lightning_hdc_error_profiling.py`
- `int8_lightning_hdc_error_profiling.py`
- `onehot_hdc_error_profiling.py`

### Old Encoders (3 files)
- `encode_whole_genome_hdv.py`
- `encode_whole_genome_hdv_streaming.py`
- `encode_genome_all_lenses.py`
- `magnitude_direction_hdc.py`

## What Was Kept

**In `genomevault/hypervector_transform/encoders/`:**

### Reference Implementations (2 files)
- `biophysical_signature_encoder.py` - Contains valuable biophysical logic
- `encode_genome_5lenses_CORRECT.py` - Early multi-lens prototype

### Documentation
- `README_ENCODERS.md` - Complete encoder directory documentation

## Production Pipeline (Current)

### Data Flow

```
1. GDiff Generation (k=12 Privacy Pipeline)
   → scripts/run_enhanced_privacy_pipeline_optimized.py
   → Output: data/experimental_strands/ERR3239334/encoding/experimental.gdiff.gz (29 MB)

2. 3-Bank HDC Encoding
   → genomevault/hdv_validation/hdc_experimentation/encoders/encode_3bank_split_architecture.py
   → Uses: genomevault/hypervector_transform/complementary_pair_encoder.py
   → Output: encoded_genome_3banks.h5 (5.3 GB)

   Architecture:
   - Bank 1: Hydrophobic (T vs A, transparent to G/C)
   - Bank 2: Major Groove (G vs C, transparent to A/T)
   - Bank 3: Hinge (Y-R vs R-Y structural flexibility)
   - Parameters: N=1024, D=5120, Overlap=128

3. Split Ternary Quantization
   → genomevault/hdv_validation/hdc_experimentation/quantization/split_ternary_quantizer.py
   → Output: encoded_genome_6banks_split_ternary.h5 (6.1 GB)

   Architecture:
   - Vector 1 (GC-dominant): [AT=0, GC, Hinge]
   - Vector 2 (AT-dominant): [AT, GC=0, Hinge]
   - √2 SNR improvement per vector

4. Query Engine
   → genomevault/hdv_validation/hdc_experimentation/query/lens_aware_simd_query_engine.py
   → SIMD-optimized dot products (1.92 μs median)
```

## Key Files in Production

| File | Purpose | Status |
|------|---------|--------|
| `complementary_pair_encoder.py` | Core encoder class | ✅ Production |
| `encode_3bank_split_architecture.py` | 3-bank HDC encoding | ✅ Production |
| `split_ternary_quantizer.py` | Ternary quantization | ✅ Production |
| `lens_aware_simd_query_engine.py` | Query engine | ✅ Production |
| `threshold_grid_search.py` | Optimization tool | ✅ Active development |
| `experiment_0_biophysical_context_validation.py` | Validation | ✅ Production |

## Architecture Evolution

| Date | Milestone |
|------|-----------|
| Nov 14, 2025 | Old whole-genome encoders |
| Nov 15, 2025 | Lightning quantization experiments |
| Nov 17, 2025 | Biophysical and multi-lens prototypes |
| Nov 19, 2025 | Error profiling analysis |
| **Nov 21, 2025** | **✅ 3-bank split architecture established** |
| **Nov 22, 2025** | **✅ Split ternary quantization** |
| **Nov 23, 2025** | **✅ Codebase cleanup & documentation** |

## What Works (Production-Ready)

- ✅ 3-bank split architecture with biophysical transparency
- ✅ Split ternary quantization (6 banks, √2 SNR improvement)
- ✅ Sparse position codebook (1 dimension per position)
- ✅ N=1024, D=5120 (optimized for genome structure)
- ✅ Lens-aware SIMD queries (1.92 μs median)
- ✅ k=12 privacy pipeline with guide FASTAs

## Import Dependencies

The production pipeline is intentionally isolated:

```python
# encode_3bank_split_architecture.py
from genomevault.hypervector_transform.complementary_pair_encoder import ComplementaryPairEncoder

# complementary_pair_encoder.py
# NO imports from genomevault (self-contained)
# Uses: numpy, pysam, gzip

# All other hdc_experimentation files
# NO imports from main genomevault codebase
# Uses: numpy, h5py, numba (local imports only)
```

This isolation enables rapid experimentation without breaking the main codebase.

## Commits

1. **Repository cleanup** (65d719b1)
   - Archived 95 temporary files to `archive/cleanup_20251123/`
   - Removed broken files and old logs

2. **Encoder cleanup** (510f49bf)
   - Archived 13 deprecated encoder files
   - Created comprehensive README_ENCODERS.md
   - Documented production pipeline

## Recovery

All archived files are preserved in git history and can be recovered:

```bash
# View archived encoders
ls -la archive/deprecated_encoders_20251123/

# Restore a specific file if needed
git checkout 65d719b1 -- genomevault/hypervector_transform/encoders/lightning_genome_hdc.py
```

## References

- **Encoder Documentation:** `genomevault/hypervector_transform/encoders/README_ENCODERS.md`
- **Project Overview:** `CLAUDE.md`
- **HDC Architecture:** `genomevault/hdv_validation/hdc_experimentation/docs/`
- **Archive:** `archive/deprecated_encoders_20251123/`

---

**Migration Completed:** November 23, 2025
**Status:** Production pipeline clean and documented
**Codebase Size Reduction:** 13 deprecated files removed from active directory
