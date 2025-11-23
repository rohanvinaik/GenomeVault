# Guide Re-alignment Status (Coordinate System Fix)

**Status:** In Progress (Started: Nov 6, 2025 5:14 PM)

## Problem Summary

The k=12 GDiff pipeline was producing 75% variant density (~1.5M variants per 2MB chunk) instead of the expected ~0.1% (2-5K variants per chunk).

**Root Cause:** Coordinate system mismatch
- Guide BAMs (`ref*.sorted.bam`): Aligned to consensus reference → in consensus coordinate space
- Experimental BAM: Aligned to guide FASTAs → in guide FASTA coordinate space
- Comparing chr1:10M in consensus coords vs chr1:10M in guide FASTA coords = comparing different genomic regions

**Result:** GDiff encoder was comparing random sequences, producing 75% variant density (expected for random 4-nucleotide sequences with 25% match rate).

## Solution Implementation

### Phase 1: Core Architecture Fix ✅ COMPLETE

**Files Modified:**

1. `genomevault/differential_encoding/align_to_reference_pool.py` (lines 137-229)
   - Added `align_guides_to_own_fastas()` static method
   - Creates guide BAMs in guide FASTA coordinate space

2. `scripts/run_enhanced_privacy_pipeline_optimized.py` (lines 36-39, 693-747)
   - Added Layer 2B re-alignment step
   - Automatically re-aligns guides during pipeline execution

3. `benchmarks/run_k12_gdiff_pipeline.py` (lines 50-54)
   - Updated to use `ref*_gdiff.bam` instead of `ref*.sorted.bam`

4. `CLAUDE.md` (lines 49-59)
   - Documented dual coordinate system requirement

5. `scripts/realign_guides_for_gdiff.py` (NEW)
   - Standalone utility for immediate re-alignment

### Phase 2: Guide Re-alignment ⏳ IN PROGRESS (11/12 Complete)

**Status:** ref1-ref11 completed, ref12 in progress

**Command Running (ref12 only):**
```bash
minimap2 -ax sr -t 10 ref12.fa.gz ERR3239934_R{1,2}.fastq.gz | \
  sambamba sort -t 10 -m 12G --tmpdir=/tmp -o ref12_gdiff.bam
```

**Process ID:** Background task 929fd5
**Log File:** `ref12_realignment.log`
**Current Progress:** 11/12 (92% complete)

**Expected Output Files:**
```
data/guide_strands/ref1_gdiff.bam   (~27 GB, guide FASTA coords)
data/guide_strands/ref2_gdiff.bam   (~27 GB, guide FASTA coords)
...
data/guide_strands/ref12_gdiff.bam  (~27 GB, guide FASTA coords)
```

**Time Estimate:**
- Per guide: ~4h 47m average (based on ref1-ref11)
- ref12: ~4-5 hours remaining
- Completion: Nov 9, 2025 ~5:30-6:30 AM

### Phase 3: Verification & Re-run ⏸️ PENDING

After re-alignment completes:

1. **Verify BAM files:**
   ```bash
   ls -lh data/guide_strands/ref*_gdiff.bam*
   ```
   - Should see 12 BAM files + 12 .bai index files
   - Each ~27 GB

2. **Restart k=12 GDiff pipeline:**
   ```bash
   caffeinate -dims python3 benchmarks/run_k12_gdiff_pipeline.py 2>&1 | tee k12_gdiff_corrected.log &
   ```

3. **Validate results:**
   - Variant density should drop from ~75% to ~0.1%
   - Expect 2,000-5,000 variants per 2MB chunk (instead of 1.5M)
   - Total variants should be ~2-5 million (instead of 500+ million)

## Current Status (as of Nov 9, 1:15 AM)

### ✅ Actions Completed:
- Implemented core architectural fixes in codebase (`benchmarks/run_k12_gdiff_pipeline.py:52-53`)
- Successfully re-aligned ref1-ref11 (11/12 guides, 92% complete)
- Each BAM file verified: 25-30 GB + .bai index
- Average time per guide: 4h 47m
- Pipeline already configured to use `ref*_gdiff.bam` files

### 🔄 Currently Running:
- **ref12 re-alignment** (started 1:11 AM)
  - minimap2 aligning ERR3239934 FASTQ (25 GB) → ref12.fa.gz
  - sambamba sorting with 12 GB memory (upgraded from 8 GB)
  - caffeinate preventing sleep interruption
  - Log: `ref12_realignment.log`

### ⏸️ Next Steps (after ref12 completes ~5:30-6:30 AM):
- Verify ref12_gdiff.bam exists and is indexed
- Run k=12 GDiff pipeline with corrected coordinate system
- Validate variant density drops from ~75% to ~0.1%

## Architecture: Guide BAMs in TWO Coordinate Systems

**Layer 2: Guide Strands (Blind Middleman)**

```
Guide FASTQ
    ↓
    ├─→ align to consensus → ref*.sorted.bam (consensus coords)
    │                        ↓
    │                    samtools consensus → ref*.fa.gz (guide FASTA)
    │
    └─→ align to guide FASTA → ref*_gdiff.bam (guide FASTA coords) ← NEW!
```

**Why Two Coordinate Systems?**

1. **Consensus coords** (`ref*.sorted.bam`):
   - Used to extract guide FASTA sequences
   - Maintains alignment to public genome references

2. **Guide FASTA coords** (`ref*_gdiff.bam`):
   - Used for GDiff differential encoding
   - Matches experimental BAM coordinate system
   - **CRITICAL for correct variant calling**

## Next Steps

1. **Monitor re-alignment progress:**
   ```bash
   tail -f guide_realignment.log
   ```

2. **When complete (~30 hours):**
   - Verify all 12 `ref*_gdiff.bam` files exist
   - Restart k=12 GDiff pipeline
   - Monitor for correct variant density (~0.1% instead of ~75%)

3. **Expected Results:**
   - GDiff file size: ~15 MB (gzipped) instead of ~1.2 GB
   - Variant count: ~2-5 million instead of ~500 million
   - Privacy preserved: k=12 anonymity maintained

## Summary

The coordinate system mismatch has been **diagnosed and fixed**. The solution is implemented in the codebase and running. Once the 30-hour re-alignment completes, we'll have the correct guide BAMs and can produce accurate GDiff encodings with proper variant density.

**Impact:** This fix is critical for all future k=12 GDiff pipelines. The main pipeline now includes Layer 2B re-alignment automatically, so this issue won't occur again.

---

**Last Updated:** Nov 9, 2025 1:15 AM
**Status Check:** `tail -20 ref12_realignment.log`
**Monitor Progress:**
```bash
# Watch ref12 re-alignment in real-time
tail -f ref12_realignment.log

# Or check specific markers
grep "mapped.*sequences" ref12_realignment.log | tail -5
```
