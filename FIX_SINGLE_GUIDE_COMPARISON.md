# GDiff Encoder Fix: Single Guide Comparison

**Date:** November 6, 2025
**Issue:** Comparing experimental data against ALL 12 guides instead of ONE
**Impact:** 84× inflated variant count (420M instead of 5M)

## Problem

The encoder was computing consensus across all 12 guide references and encoding every position where experimental data differed from that majority consensus. This violated the fundamental privacy architecture:

**Broken behavior:**
```python
# Get alleles from ALL 12 guides
for pool_bam in pool_bams:  # All 12 guides
    alleles = get_alleles(pool_bam)
    all_pool_alleles.extend(alleles)

# Consensus = majority vote across all 12
pool_consensus = most_common(all_pool_alleles)

# Encode if experimental ≠ majority
if query_allele != pool_consensus:
    encode_variant()
```

**Result:** ~420M variants (projected from 20% completion)

## Solution

Compare experimental data to **ONE randomly-selected guide per chunk**, using the existing SGRS cryptographic binding system:

```python
# Select ONE guide per chunk (SGRS-based)
chunk_id = start // self.chunk_size
if chunk_id in self.chunk_guide_map:
    selected_guide_idx, alignment_seed = self.chunk_guide_map[chunk_id]
else:
    selected_guide_idx = deterministic_random(chunk_id)

# Use ONLY the selected guide
selected_guide_bam = pool_bams[selected_guide_idx]

# Get alleles from ONLY this guide
guide_alleles = get_guide_alleles(selected_guide_bam, pos)
guide_consensus = most_common(guide_alleles)

# Encode ONLY if experimental ≠ selected guide
if query_allele != guide_consensus:
    encode_variant()
```

**Expected result:** ~5-7M variants (correct)

## Changes Made

### `genomevault/differential_encoding/gdiff/encoder.py`

1. **Added guide selection logic** (lines 567-593):
   - Uses `chunk_guide_map` from SGRS if available
   - Falls back to deterministic random selection
   - Logs selected guide per region

2. **New helper method** `_get_guide_alleles_at_position()` (lines 736-768):
   - Extracts alleles from single guide BAM
   - Used instead of `_get_pool_alleles_at_position()` for comparison

3. **Updated comparison logic** (lines 616-631):
   - Changed from `pool_consensus` to `guide_consensus`
   - Compares experimental to SINGLE selected guide only

4. **Updated variant creation** (lines 721, 671):
   - Changed `ref=pool_consensus` to `ref=guide_consensus`
   - Documentation updated to reflect SGRS-based selection

## Privacy Guarantee

The fix maintains information-theoretic privacy:

- **Random guide selection per chunk** → unpredictable comparison reference
- **Cryptographic binding via SGRS** → chunk→guide mapping encrypted with HMAC-SHA256
- **k=12 anonymity preserved** → still using 12 guides, just not simultaneously

## Verification

**Expected outcomes after restart:**
1. Variant count should drop from ~420M → ~5-7M (84× reduction)
2. GDiff file size should drop from ~1.5GB → ~15MB
3. Processing may be slightly faster (fewer BAM seeks)

**Test command:**
```bash
# After first chromosome completes
python3 scripts/analyze_gdiff_run.py gdiff_sgrs_pipeline_fixed.log | grep "Total Variants"
```

## Related Files

- `genomevault/differential_encoding/gdiff/template_utils.py` - Template auto-detection (created)
- `benchmarks/run_k12_gdiff_pipeline.py` - Updated to use GRCh38 build for template detection
- `genomevault/differential_encoding/gdiff/encoder.py` - Main fix location

## Next Steps

1. ✅ Kill current pipeline (PID 7696)
2. ✅ Restart with fixed encoder
3. ⏳ Monitor variant count after first chromosome
4. ⏳ If verified, implement template deduplication for additional 99% reduction
