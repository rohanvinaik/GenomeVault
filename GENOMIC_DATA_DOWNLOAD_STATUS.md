# Genomic Data Download Status

**Last Updated:** October 24, 2025, 9:43 PM PST

---

## Current Downloads in Progress

### ✅ European (Complete)
- **ERR3239548**: ✅ Complete (23 GB) - 2 FASTQ files
- **ERR3239590**: ✅ Complete (22 GB) - 2 FASTQ files
- **Status**: 2/2 samples downloaded, 45 GB total

### 🔄 East Asian (In Progress)
- **ERR3239578**: 🔄 Downloading (started 9:42 PM)
- **ERR3239612**: ⏳ Queued (will start after ERR3239578)
- **Status**: 0/2 samples complete

### ⏳ African (Queued)
- **ERR3239756**: ⏳ Not started
- **ERR3239778**: ⏳ Not started
- **Status**: 0/2 samples

### ⏳ South Asian (Queued)
- **ERR3239912**: ⏳ Not started
- **ERR3239934**: ⏳ Not started
- **Status**: 0/2 samples

---

## Overall Progress

| Cohort | Samples | Status | Size | ETA |
|--------|---------|--------|------|-----|
| European | 2/2 | ✅ Complete | 45 GB | Done |
| East Asian | 0/2 | 🔄 In Progress | 0 GB | 2-3 hrs |
| African | 0/2 | ⏳ Queued | 0 GB | 4-6 hrs |
| South Asian | 0/2 | ⏳ Queued | 0 GB | 6-9 hrs |
| **TOTAL** | **2/8** | **25% Complete** | **45/180 GB** | **~6-9 hrs** |

---

## Monitoring Commands

```bash
# Check overall download status
cat data/download_state.json | python3 -m json.tool | grep -A 5 "status"

# Monitor East Asian download
tail -f logs/download_east_asian_*.log

# Check disk usage
du -sh data/downloaded/fastq/*/

# Count completed samples
find data/downloaded/fastq -type d -name "ERR*" -exec ls -d {} \; | wc -l

# Check running processes
ps aux | grep "download_genomic_data" | grep -v grep
```

---

## Next Steps

The downloads are running **sequentially** in this order:

1. **East Asian** (PID 83158) - Currently downloading ERR3239578
2. **African** - Will start automatically after East Asian completes
3. **South Asian** - Will start automatically after African completes

**To monitor in real-time:**
```bash
# Watch download progress (updates every 30 seconds)
watch -n 30 'cat data/download_state.json | python3 -m json.tool | grep -B 2 -A 8 "\"status\": \"downloading\""'
```

---

## Expected Final State

After all downloads complete (~6-9 hours):

```
data/downloaded/fastq/
├── european/
│   ├── ERR3239548/ (23 GB)
│   └── ERR3239590/ (22 GB)
├── east_asian/
│   ├── ERR3239578/ (~22 GB)
│   └── ERR3239612/ (~22 GB)
├── african/
│   ├── ERR3239756/ (~22 GB)
│   └── ERR3239778/ (~22 GB)
└── south_asian/
    ├── ERR3239912/ (~22 GB)
    └── ERR3239934/ (~22 GB)

Total: 8 samples, ~176 GB
```

---

## Notes

- All downloads are **sequential** (one at a time)
- Each sample takes 1-1.5 hours on average
- Downloads use SRA Toolkit: prefetch → fasterq-dump → pigz
- Progress is tracked in `data/download_state.json`
- Logs are in `logs/download_*.log`

---

## Troubleshooting

**If a download fails:**
```bash
# Check the error in download_state.json
cat data/download_state.json | python3 -m json.tool | grep -A 10 "failed"

# Retry a specific sample
python scripts/download_genomic_data_automated.py --accession ERR3239XXX
```

**If downloads are stuck:**
```bash
# Kill stuck processes
pkill -9 -f prefetch
pkill -9 -f fasterq-dump

# Restart from where it left off (script auto-skips completed samples)
python scripts/download_genomic_data_automated.py --pool east_asian --samples 2
```
