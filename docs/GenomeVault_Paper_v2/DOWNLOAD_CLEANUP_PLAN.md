# Genomic Download Cleanup Plan

**Date:** October 24, 2025
**Issue:** Downloaded 3 European samples instead of intended 2
**Disk Impact:** 68 GB (can save 23 GB by removing 1 sample)

---

## Current Downloads

| Accession | Status | Size | Files |
|-----------|--------|------|-------|
| ERR3239548 | ✅ Complete | 23 GB | ERR3239548_1.fastq.gz (11G), ERR3239548_2.fastq.gz (12G) |
| ERR3239590 | ✅ Complete | 22 GB | ERR3239590_1.fastq.gz, ERR3239590_2.fastq.gz |
| ERR3239620 | ✅ Complete | 23 GB | ERR3239620_1.fastq.gz, ERR3239620_2.fastq.gz |
| ERR3239520 | ❌ Failed | 0 GB | Lock file error |
| ERR3239790 | ❌ Failed | 0 GB | Transfer interrupted |
| ERR3239812 | ❌ Failed | 0 GB | Prefetch failed |
| ERR3239920 | ⏸️ Stopped | 0 GB | Incomplete |

**Total:** 68 GB downloaded, 23 GB can be removed

---

## Recommended Actions

### 1. Keep 2 Best Quality Samples

**Keep:**
- ✅ ERR3239548 (23 GB) - First successful download
- ✅ ERR3239590 (22 GB) - Second successful download

**Remove:**
- ❌ ERR3239620 (23 GB) - Third sample (exceeds requirement)
- ❌ ERR3239520, ERR3239790, ERR3239812, ERR3239920 - Failed/incomplete

**Commands:**
```bash
# Remove third complete sample
rm -rf /Users/rohanvinaik/genomevault/data/downloaded/fastq/european/ERR3239620

# Remove failed attempts
rm -rf /Users/rohanvinaik/genomevault/data/downloaded/fastq/european/ERR3239520
rm -rf /Users/rohanvinaik/genomevault/data/downloaded/fastq/european/ERR3239790
rm -rf /Users/rohanvinaik/genomevault/data/downloaded/fastq/european/ERR3239812
rm -rf /Users/rohanvinaik/genomevault/data/downloaded/fastq/european/ERR3239920

# Verify cleanup
ls -lh /Users/rohanvinaik/genomevault/data/downloaded/fastq/european/
```

### 2. Update Download State JSON

Edit `/Users/rohanvinaik/genomevault/data/download_state.json`:
- Remove entries for ERR3239620, ERR3239520, ERR3239790, ERR3239812, ERR3239920
- Update `total_downloaded_gb` from 73.09 to ~45.5 GB

### 3. Prevent Future Over-Downloads

**Option A: Update SAMPLE_POOLS configuration**

Edit `scripts/download_genomic_data_automated.py` line 33-36:
```python
'european': {
    'reference': [
        'ERR3239548', 'ERR3239590'  # Limit to 2 samples
    ],
    'query': ['ERR3239276', 'ERR3239334'],
    'description': 'European ancestry (UK/Europe) - 2 samples per cohort'
},
```

**Option B: Always use --samples 2 flag**
```bash
python scripts/download_genomic_data_automated.py --pool european --samples 2 --type reference
```

---

## Verification After Cleanup

**Expected state:**
```
data/downloaded/fastq/european/
├── ERR3239548/
│   ├── ERR3239548_1.fastq.gz (11 GB)
│   └── ERR3239548_2.fastq.gz (12 GB)
└── ERR3239590/
    ├── ERR3239590_1.fastq.gz (~11 GB)
    └── ERR3239590_2.fastq.gz (~11 GB)

Total: ~45 GB (2 samples, 4 files)
```

**Verify with:**
```bash
# Count samples
ls -1d data/downloaded/fastq/european/ERR* | wc -l
# Should output: 2

# Total size
du -sh data/downloaded/fastq/european/
# Should show: ~45G
```

---

## Other Cohorts Status

Check if other cohorts also have this issue:

```bash
# East Asian
ls -1d data/downloaded/fastq/east_asian/ERR* 2>/dev/null | wc -l

# African
ls -1d data/downloaded/fastq/african/ERR* 2>/dev/null | wc -l

# South Asian
ls -1d data/downloaded/fastq/south_asian/ERR* 2>/dev/null | wc -l
```

Each should show **2 samples** or **0** (if not downloaded yet).

---

## Disk Space Saved

- Before cleanup: 68 GB
- After cleanup: 45 GB
- **Savings: 23 GB** (33% reduction)

---

## Next Steps

1. Execute cleanup commands above
2. Update download_state.json
3. Verify only 2 European samples remain
4. Check other cohorts
5. Document final download configuration (2 samples per cohort)
