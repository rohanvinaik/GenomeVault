# HDV Validation Suite

**Location:** `/Users/rohanvinaik/genomevault/genomevault/hdv_validation/`

Clean, flat structure for HDV (Hyperdimensional Vector) validation and testing.

## Quick Start

```bash
# Run quantization comparison (10,000 queries with report)
python3 -m genomevault.hdv_validation.compare_quantizations \
    --quantizations float32 int8 int4 binary \
    --sample-size 10000 \
    --seed 42 \
    --generate-report

# Generate BED files for UCSC collision testing
python3 -m genomevault.hdv_validation.compare_quantizations \
    --quantizations float32 int8 int4 binary \
    --sample-size 10000 \
    --seed 42 \
    --generate-beds

# Standalone BED generation from existing error files
python3 -m genomevault.hdv_validation.generate_collision_beds \
    --input genomevault/hdv_validation/results/comparison_results/high_precision_errors.json \
    --output-dir genomevault/hdv_validation/results/bed_files

# T2T Validation Workflow (4 steps)
# 1. Clean BED files for UCSC upload (raw/ → ucsc_cleaned/)
python3 -m genomevault.hdv_validation.clean_bed_for_ucsc \
    --input-dir genomevault/hdv_validation/results/bed_files/liftover_bed

# 2. Upload to UCSC liftOver (manual step)
#    - Go to: https://genome.ucsc.edu/cgi-bin/hgLiftOver
#    - Upload files from ucsc_cleaned/
#    - Download results to liftover/

# 3. Clean liftOver output (liftover/ gets extra columns from UCSC)
python3 -m genomevault.hdv_validation.clean_bed_for_ucsc \
    --input-dir genomevault/hdv_validation/results/bed_files/liftover_bed/liftover \
    --no-subdirs

# 4. Validate against T2T-CHM13v2.0 (reads from liftover/)
python3 -m genomevault.hdv_validation.validate_against_t2t \
    --bed-file genomevault/hdv_validation/results/bed_files/liftover_bed/liftover/common_genomevault_liftover.bed
```

## Key Files

### Core Query Engine
- **`query_engine.py`** - Multi-lens HDV query system (PreEncodedMultiLensHDV class)
  - Reads from 3D H5 files (optimized batch I/O)
  - 5-lens biophysical encoding (AT, GC, PuPy, AmKe, StWk)
  - Supports float32, int8, int4, binary quantization

### Main Testing Scripts
- **`compare_quantizations.py`** - Main comparison script across all quantization levels
- **`generate_report.py`** - Generates comprehensive markdown reports
- **`generate_collision_beds.py`** - Generates BED files for UCSC collision testing
- **`clean_bed_for_ucsc.py`** - Cleans BED files for UCSC Genome Browser (4-column format)
- **`validate_against_t2t.py`** - Validates predictions against T2T-CHM13v2.0 via UCSC API
- **`validation_utils.py`** - Shared utilities (load_gdiff, sample positions, etc.)
- **`signature_correction.py`** - Error correction system using safe signatures

### Supporting Scripts
- Discovery scripts: `discover_safe_signatures*.py`
- Tuning scripts: `tune_lens_thresholds.py`, `tune_correction_profiles*.py`
- Analysis scripts: `analyze_*.py`, `error_profile_analysis.py`

## Data Files (NOT in this directory)

HDV data files remain in their original locations:
- **H5 files:** `data/experimental_strands/ERR3239334/hdv_encoding/`
  - `encoded_genome_5lenses_3d.h5` (float32, 281 GB)
  - `encoded_genome_5lenses_3d_int8.h5` (int8, 54 GB)
  - `encoded_genome_5lenses_3d_int4.h5` (int4, 24 GB)
  - `encoded_genome_5lenses_3d_binary.h5` (binary, 9 GB)

- **Guide FASTAs:** `/Volumes/1TBStorage/guide_strands/`
  - ref1.fa.gz through ref11.fa.gz

- **GDiff file:** `data/experimental_strands/ERR3239334/experimental.gdiff.gz`

## Results Output

All results are saved to: **`genomevault/hdv_validation/`**

### Comparison Results
`genomevault/hdv_validation/results/comparison_results/`
- `{quantization}_predictions_detailed.json` - All predictions with lens votes
- `{quantization}_predictions_corrected.json` - Corrected predictions
- `{quantization}_correction_stats.json` - Corrective lens statistics
- `quantization_comparison_same_queries.json` - Summary comparison across all quantizations
- `common_errors.json` - Errors across all quantizations
- `high_precision_errors.json` - Float32/int8 specific errors
- `low_precision_errors.json` - Int4/binary specific errors

### Reports
`genomevault/hdv_validation/reports/`
- `quantization_validation_report.md` - Comprehensive validation report

### Logs
`genomevault/hdv_validation/logs/`
- Timestamped execution logs

### Signature Corrections
`genomevault/hdv_validation/results/signature_corrections/`
- `exhaustive_ALL_CORRECT/` - Safe signatures (0 breaks on training set)
- `conservative_union/` - Conservative signature set
- `retrained_signatures/` - Retrained signature corrections

### BED Files (Organized Workflow)
`genomevault/hdv_validation/results/bed_files/`
- `genomevault_predictions.bed` - HDV biophysical predictions (GenomeVault calls)
- `bam_ground_truth.bed` - BAM reference calls (standard pipeline)
- `high_precision_*.bed` - High-precision error cohorts (float32 ∩ int8)
- `low_precision_*.bed` - Low-precision error cohorts (int4 ∩ binary)
- `common_*.bed` - Common errors across all quantizations

`genomevault/hdv_validation/results/bed_files/liftover_bed/` - **T2T Validation Pipeline**
```
liftover_bed/
├── raw/              ← Step 1: Raw BED files (track headers removed)
├── ucsc_cleaned/     ← Step 2: Ready for UCSC liftOver upload
├── liftover/         ← Step 3: liftOver results from UCSC (manually placed)
└── [cleaned files]   ← Step 4: Final cleaned liftOver output (with --no-subdirs)
```

**Workflow:**
1. `raw/` - Original error BED files (4-column, no track headers)
2. `ucsc_cleaned/` - Cleaned for UCSC upload (remove any extra formatting)
3. Upload to UCSC liftOver → download to `liftover/`
4. Clean liftOver output (removes extra columns UCSC adds)
5. Validate against T2T-CHM13v2.0

### T2T-CHM13v2.0 Validation Results
`genomevault/hdv_validation/results/BAM_vs_pipeline_accuracy/`
- `t2t_validation_*.md` - Validation reports vs. T2T reference genome
- `t2t_validation_*.json` - Detailed validation data (per-position results)
- Accuracy metrics against gold-standard T2T-CHM13v2.0 via UCSC API

## Import Structure

All scripts use the `genomevault.hdv_validation` package:

```python
from genomevault.hdv_validation.query_engine import PreEncodedMultiLensHDV
from genomevault.hdv_validation.validation_utils import load_gdiff, sample_test_positions
from genomevault.hdv_validation.signature_correction import analyze_with_signatures
from genomevault.hdv_validation.generate_report import generate_markdown_report
from genomevault.hdv_validation.generate_collision_beds import generate_collision_beds
from genomevault.hdv_validation.clean_bed_for_ucsc import clean_bed_directory
from genomevault.hdv_validation.validate_against_t2t import T2TReferenceValidator
```

## Performance (Latest: Nov 20, 2025, 10,000 queries)

### Accuracy (with corrective lens)
- **Float32:** 99.37% (+0.22% from correction)
- **Int8:** 99.39% (+0.17% from correction)
- **Int4:** 99.07% (+0.21% from correction)
- **Binary:** 97.25% (+0.72% from correction)

### Query Speed (optimized H5 I/O)
- **Float32:** 0.292 ms/query (~3,425 queries/sec)
- **Int8:** 0.540 ms/query (~1,852 queries/sec)
- **Int4:** 0.531 ms/query (~1,883 queries/sec)
- **Binary:** 0.479 ms/query (~2,088 queries/sec)

### Compression
- **Int8:** 5.2× smaller (54 GB vs 281 GB)
- **Int4:** 11.7× smaller (24 GB vs 281 GB)
- **Binary:** 31× smaller (9 GB vs 281 GB)

## Architecture

### Multi-Lens Biophysical Encoding

Five complementary lenses capture different biophysical properties:

| Lens | Property | Accuracy |
|------|----------|----------|
| **AT** | Hydrogen bonding (weak) | High for A/T detection |
| **GC** | Hydrogen bonding (strong) | High for G/C detection |
| **PuPy** | Ring structure (purine/pyrimidine) | Moderate |
| **AmKe** | Functional groups (amino/keto) | Moderate |
| **StWk** | Thermodynamic stability | Moderate |

### Query Process

1. **Load position from H5** - Batch read all 5 lenses (88 μs)
2. **Multi-lens voting** - Each lens votes for A, T, G, or C (2 μs)
3. **Corrective lens** (optional) - Signature-based error correction (35 μs)
4. **Total:** ~125 μs per query

### Error Correction

Post-query signature-based corrections:
- **Conservative signatures:** 0 breaks on training set
- **Relaxed signatures:** ≥5:1 fixes-to-breaks ratio
- **Impact:** +0.01% to +0.54% accuracy improvement

## Migration History

**Nov 20, 2025:** Consolidated from scattered locations into single `hdv_validation/` directory

**Previous locations:**
- `genomevault/hypervector_transform/validation/architecture_testing/`
- `genomevault/hypervector_transform/validation/validate_multi_lens_with_theoretical.py`
- `genomevault/hypervector_transform/tests/` (short-lived)

**Benefits:**
- Flat, easy-to-navigate structure
- Clear separation of code vs. data
- Simple import paths
- All validation tools in one place

---

**Last Updated:** November 20, 2025
**Version:** 2.0.0 (Consolidated structure)
