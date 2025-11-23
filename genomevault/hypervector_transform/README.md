# Hypervector Transform Module

This module contains all HDV (Hyperdimensional Vector) and HDC (Hyperdimensional Computing) related code for GenomeVault.

## Directory Structure

```
genomevault/hypervector_transform/
├── __init__.py                     # Core HDV transform functionality
├── README.md                       # This file
│
├── tests/                          # Testing & validation suite ⭐ NEW LOCATION
│   ├── validation/                 # Core validation scripts (moved from validation/)
│   │   ├── validate_multi_lens_with_theoretical.py
│   │   ├── validate_hdv_*.py
│   │   └── ... (19 validation scripts)
│   │
│   ├── architecture/               # Architecture testing (moved from validation/architecture_testing/)
│   │   ├── compare_quantizations.py
│   │   ├── generate_report.py
│   │   ├── signature_correction.py
│   │   └── ... (32 test scripts)
│   │
│   ├── benchmarks/                 # Performance benchmarks
│   └── utils/                      # Shared test utilities
│
├── encoders/                       # Encoder implementations
│   ├── encode_genome_5lenses_CORRECT.py
│   ├── encode_whole_genome_hdv.py
│   ├── biophysical_signature_encoder.py
│   ├── *_lightning_hdc.py         # Lightning-fast HDC variants
│   └── ...
│
├── quantization/                   # Quantization tools
│   ├── generate_int8_quantized_h5.py
│   ├── generate_int4_quantized_h5.py
│   └── generate_binary_quantized_h5.py
│
├── benchmarks/                     # HDC-specific benchmarks
└── debug/                          # Debugging utilities
```

## Key Files

### Core Modules
- `complementary_pair_encoder.py` - Complementary pair HDV encoding
- `error_profiler.py` - Error profiling and analysis
- `nucleotide_hdv_explicit.py` - Explicit nucleotide HDV encoding
- `privacy_hdv_single_encoding.py` - Privacy-preserving single encoding
- `privacy_preserving_genome_hdv.py` - Privacy-preserving genome HDV

### Testing & Validation (NEW LOCATION: `tests/`)
Comprehensive validation scripts for testing accuracy, performance, and correctness of HDV encoding systems.

**Main validation scripts:**
- `tests/validation/validate_multi_lens_with_theoretical.py` - Multi-lens HDV query system (PRODUCTION)
- `tests/validation/validate_hdv_against_gdiff.py` - GDiff comparison validation
- `tests/validation/validate_whole_genome_hdv.py` - Whole genome validation
- `tests/architecture/compare_quantizations.py` - Quantization comparison (ACTIVELY USED)

**See `tests/README.md` for detailed documentation**

### Encoders
Various encoder implementations for different use cases:

**Production encoders:**
- `encode_genome_5lenses_CORRECT.py` - 5-lens multi-lens encoder (PRODUCTION)
- `encode_whole_genome_hdv.py` - Whole genome HDV encoder

**Experimental encoders:**
- `binary_lightning_hdc.py` - Binary quantized lightning HDC
- `int8_lightning_hdc.py` - Int8 quantized lightning HDC
- `int4_lightning_hdc.py` - Int4 quantized lightning HDC
- `lightning_genome_hdc.py` - Lightning-fast genome HDC

### Quantization
Tools for converting HDV encodings to reduced precision formats:

- `generate_int8_quantized_h5.py` - Int8 quantization (127 levels, 3.7× compression)
- `generate_int4_quantized_h5.py` - Int4 quantization (15 levels, 7.4× compression)
- `generate_binary_quantized_h5.py` - Binary quantization (3 levels, 3.7× compression)

**Usage:**
```bash
# Generate int8 quantized version
python genomevault/hypervector_transform/quantization/generate_int8_quantized_h5.py

# Generate binary quantized version
python genomevault/hypervector_transform/quantization/generate_binary_quantized_h5.py
```

### Benchmarks
Performance benchmarking tools for measuring HDV operation speed and efficiency.

### Debug
Debugging and analysis tools for troubleshooting HDV encoding issues.

## Validation Results

Validation results are stored in `HDV_VALIDATION_PACKAGE/`:

```
HDV_VALIDATION_PACKAGE/
├── main_findings/              # Primary validation results
│   ├── comprehensive_float32_test.log
│   ├── binary_quantization_test.log
│   └── ... (results from major tests)
│
├── multi_lens_tests/          # Multi-lens specific tests
│   └── ...
│
├── int8_quantization.log      # Quantization logs
├── binary_quantization.log
└── int4_quantization.log
```

## Running Validations

```bash
# Quantization comparison (1000 queries, quick test)
python -m genomevault.hypervector_transform.tests.architecture.compare_quantizations \
    --quantizations float32 int8 int4 binary \
    --sample-size 1000 \
    --seed 42

# Full validation with report (10,000 queries)
python -m genomevault.hypervector_transform.tests.architecture.compare_quantizations \
    --quantizations float32 int8 int4 binary \
    --sample-size 10000 \
    --seed 42 \
    --generate-report

# Generate report from existing results
python -m genomevault.hypervector_transform.tests.architecture.generate_report
```

## Key Findings (Latest: Nov 2025, 10,000 queries)

**Multi-lens accuracy with corrective lens system:**
- Float32: 99.32% (baseline, +0.11% from correction)
- Int8: 99.27% (production-ready, +0.08% from correction)
- Int4: 99.02% (+0.01% from correction)
- Binary: 97.11% (+0.54% from correction)

**Query performance (optimized H5 I/O):**
- Float32: 0.622 ms/query (~1,600 queries/sec)
- Int8: 2.486 ms/query (~400 queries/sec)
- Int4: 2.125 ms/query (~470 queries/sec)
- Binary: 0.667 ms/query (~1,500 queries/sec)

**Speedup vs BAM pileup (40 ms/query):**
- Float32: 64× faster
- Int8: 16× faster
- Int4: 19× faster
- Binary: 60× faster

**Compression:**
- Int8: 5.2× smaller (54 GB vs 281 GB)
- Int4: 11.7× smaller (24 GB vs 281 GB)
- Binary: 31× smaller (9 GB vs 281 GB)

## Unit Tests

Unit tests are located in `tests/hypervector/`:

```bash
# Run all hypervector tests
pytest tests/hypervector/

# Run specific test
pytest tests/hypervector/test_int8_accuracy.py
```

## Integration with GenomeVault

This module integrates with the broader GenomeVault system:

- **HDV Encoding:** `genomevault/hypervector_transform/`
- **GDiff Encoding:** `genomevault/differential_encoding/gdiff/`
- **Reference Pool:** `genomevault/differential_encoding/align_to_reference_pool.py`
- **Privacy Stack:** `genomevault/cli/privacy_query.py`

## Migration Notes

### Latest Migration (Nov 20, 2025)

**Testing code reorganized** to proper structure:

- **OLD:** `genomevault/hypervector_transform/validation/` → **NEW:** `genomevault/hypervector_transform/tests/validation/`
- **OLD:** `genomevault/hypervector_transform/validation/architecture_testing/` → **NEW:** `genomevault/hypervector_transform/tests/architecture/`

**Import changes:**
```python
# OLD (deprecated)
from genomevault.hypervector_transform.validation.validate_multi_lens_with_theoretical import PreEncodedMultiLensHDV

# NEW (current)
from genomevault.hypervector_transform.tests.validation.validate_multi_lens_with_theoretical import PreEncodedMultiLensHDV
```

**Backup:** Old structure backed up in `validation_OLD_BACKUP/` - safe to remove after verification.

### Previous Migrations

- Root directory validation scripts → `genomevault/hypervector_transform/validation/` (now → `tests/validation/`)
- Root directory encoders → `genomevault/hypervector_transform/encoders/`
- Root directory tests → `tests/hypervector/`
- HDV_VALIDATION_PACKAGE generators → `genomevault/hypervector_transform/quantization/`

All import paths have been updated to reflect the new structure.

## Documentation

- **HDV Architecture:** `docs/HDV_ENCODING_ARCHITECTURE_EXPLAINED.md`
- **Privacy Stack:** `docs/guides/PROBABILISTIC_ALIGNMENT_PRIVACY_STACK.md`
- **Validation Reports:** `HDV_VALIDATION_PACKAGE/*.md`

---

**Last Updated:** November 20, 2025
**Version:** 1.1.0 (Testing code reorganized into `tests/` directory)
