# JSON File Creation Tracking

This document tracks which methods/files create each JSON output in the `results/comparison_results/` directory.

## Primary Workflow (UPDATED ✅)

### 1. `compare_quantizations.py` (MAIN SCRIPT)

**Output Directory:** `genomevault/hdv_validation/results/comparison_results/`

**Line 71:** Sets default output_dir
```python
output_dir = Path("genomevault/hdv_validation/results/comparison_results")
```

#### Files Created:

| JSON File | Line | Method | Status |
|-----------|------|--------|--------|
| `quantization_comparison_same_queries.json` | 338-339 | `save_results(report, report_file)` | ✅ NEW PATH |
| `{quant}_predictions_detailed.json` (×4) | 343-346 | `json.dump(predictions_by_quant[quant], f)` | ✅ NEW PATH |
| `high_precision_errors.json` | 364-372 | `json.dump(error_data, f)` | ✅ NEW PATH |
| `low_precision_errors.json` | 364-372 | `json.dump(error_data, f)` | ✅ NEW PATH |
| `common_errors.json` | 364-372 | `json.dump(error_data, f)` | ✅ NEW PATH |
| `{quant}_predictions_corrected.json` (×4) | 402-404 | `json.dump(correction_analysis['corrected_predictions'], f)` | ✅ NEW PATH |
| `{quant}_correction_stats.json` (×4) | 408-410 | `json.dump(correction_analysis['statistics'], f)` | ✅ NEW PATH |

**Total:** 16 JSON files created by `compare_quantizations.py`

---

### 2. `signature_correction.py` (SUPPORTING MODULE)

**Called by:** `compare_quantizations.py` (line 392)

**Method:** `analyze_with_signatures(predictions, signatures_path, quantization)`

**Reads from:** `genomevault/hdv_validation/results/signature_corrections/exhaustive_ALL_CORRECT/`

**Returns:** Dictionary with:
- `corrected_predictions` (saved to `{quant}_predictions_corrected.json`)
- `statistics` (saved to `{quant}_correction_stats.json`)

**Status:** ✅ MIGRATED, reads from correct signature path (line 388)

---

### 3. `generate_report.py` (REPORT GENERATOR)

**Output Directory:** `genomevault/hdv_validation/reports/`

**Line 680:** Sets base_dir
```python
base_dir = Path("/Users/rohanvinaik/genomevault/genomevault/hdv_validation")
```

**Line 450:** Sets report output in `compare_quantizations.py`
```python
report_output_dir = output_dir.parent.parent / "reports"
```

#### Files Created:

| File | Line | Method | Status |
|------|------|--------|--------|
| `quantization_validation_report.md` | 450-453 | `generate_markdown_report()` | ✅ NEW PATH |

**Status:** ✅ MIGRATED, outputs to correct report path

---

## Summary of JSON File Sources

### Comparison Results (16 files)

1. **Float32** (4 files)
   - `float32_predictions_detailed.json` ← `compare_quantizations.py:343`
   - `float32_predictions_corrected.json` ← `compare_quantizations.py:402`
   - `float32_correction_stats.json` ← `compare_quantizations.py:408`
   - Included in error files below

2. **Int8** (4 files)
   - `int8_predictions_detailed.json` ← `compare_quantizations.py:343`
   - `int8_predictions_corrected.json` ← `compare_quantizations.py:402`
   - `int8_correction_stats.json` ← `compare_quantizations.py:408`
   - Included in error files below

3. **Int4** (4 files)
   - `int4_predictions_detailed.json` ← `compare_quantizations.py:343`
   - `int4_predictions_corrected.json` ← `compare_quantizations.py:402`
   - `int4_correction_stats.json` ← `compare_quantizations.py:408`
   - Included in error files below

4. **Binary** (4 files)
   - `binary_predictions_detailed.json` ← `compare_quantizations.py:343`
   - `binary_predictions_corrected.json` ← `compare_quantizations.py:402`
   - `binary_correction_stats.json` ← `compare_quantizations.py:408`
   - Included in error files below

5. **Error Analysis** (3 files)
   - `common_errors.json` ← `compare_quantizations.py:364`
   - `high_precision_errors.json` ← `compare_quantizations.py:364`
   - `low_precision_errors.json` ← `compare_quantizations.py:364`

6. **Summary** (1 file)
   - `quantization_comparison_same_queries.json` ← `compare_quantizations.py:338`

### Legacy Files (not regenerated in latest test)
- `query_speed_breakdown.json` (from older analysis scripts)
- `theoretical_predictions_float32.json` (from older analysis scripts)

---

## Supporting Scripts (NOT UPDATED ⚠️)

The following scripts still reference old paths but are NOT part of the main workflow:

1. **Discovery Scripts** (signature finding)
   - `discover_safe_signatures_binary.py` (lines 286-287)
   - `discover_safe_signatures_int4.py` (line 327)
   - `discover_thermodynamic_corrections_advanced.py` (line 279)

2. **Tuning Scripts** (parameter optimization)
   - `tune_from_teachable_moments.py` (line 186)
   - `tune_correction_profiles_v2.py` (line 177)

3. **Analysis Scripts** (older one-off analyses)
   - `signature_based_correction.py` (line 219)

**Note:** These scripts are for research/development and do not affect the main validation workflow.

---

## Migration Verification

### ✅ VERIFIED - Latest Test (Nov 20, 12:59)

All 16 primary JSON files created with correct new paths:

```bash
ls -lh genomevault/hdv_validation/results/comparison_results/*.json | grep "Nov 20 12:59"
```

**Result:** 16 files confirmed at correct location

### ✅ VERIFIED - Report Generation

```bash
ls -lh genomevault/hdv_validation/reports/quantization_validation_report.md
```

**Result:** Report created at correct location (626 lines, 18 KB)

---

## Quick Reference: File Creation Flow

```
compare_quantizations.py
  ↓
  1. Runs queries on all quantizations
  2. Saves detailed predictions (×4)
  3. Categorizes errors (×3 error files)
  4. Calls analyze_with_signatures() (from signature_correction.py)
     ↓
     - Loads signatures from: results/signature_corrections/exhaustive_ALL_CORRECT/
     - Returns corrected predictions + statistics
  5. Saves corrected predictions (×4)
  6. Saves correction statistics (×4)
  7. Saves comparison summary (×1)
  8. Calls generate_markdown_report()
     ↓
     - Generates comprehensive markdown report
     - Saves to: reports/quantization_validation_report.md
```

---

**Last Updated:** November 20, 2025
**Status:** ✅ All primary workflow scripts migrated and verified
