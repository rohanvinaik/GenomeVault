# GenomeVault Genomic Data Downloader

## 🎯 Overview

Automated pipeline for downloading genomic data from ENA/SRA with real-time graphical monitoring.

## ✨ Features

- ✅ Automated download from European Nucleotide Archive (ENA)
- ✅ Parallel download support with configurable sample count
- ✅ Real-time graphical progress tracker
- ✅ Automatic FASTQ compression with pigz
- ✅ Resumable downloads with JSON state tracking
- ✅ Disk space monitoring and warnings
- ✅ Process resource usage tracking (CPU, Memory)
- ✅ Comprehensive error handling and logging

## 📦 Components

### 1. Download Script
**File**: `scripts/download_genomic_data_automated.py`

Automated downloader with features:
- Downloads from 4 ancestry pools (European, East Asian, African, South Asian)
- Tracks download state in JSON file
- Handles prefetch → FASTQ extraction → compression pipeline
- Resumable: Can restart failed downloads
- Parallel sample processing

### 2. Graphical Monitor
**File**: `scripts/monitor_genomic_downloads.py`

Real-time dashboard showing:
- Overall download progress with progress bars
- Per-sample status (completed, downloading, failed, queued)
- Active download processes (fasterq-dump, prefetch, pigz)
- Disk space usage with warnings
- Download speeds and ETAs
- CPU/Memory usage per process

### 3. Quick Start Launcher
**File**: `start_genomic_downloads.sh`

One-command launcher that:
- Checks dependencies automatically
- Verifies disk space
- Starts download in background
- Launches graphical monitor
- Provides easy controls

## 🚀 Quick Start

### Option 1: Use the Launcher (Easiest)

```bash
# Download 3 European samples (default)
./start_genomic_downloads.sh

# Download 7 European samples
./start_genomic_downloads.sh european 7

# Download from all pools (5 samples each)
./start_genomic_downloads.sh all 5
```

### Option 2: Manual Usage

```bash
# Start download in background
python scripts/download_genomic_data_automated.py \
    --pool european \
    --samples 5 \
    > logs/download.log 2>&1 &

# Launch monitor
python scripts/monitor_genomic_downloads.py --watch
```

### Option 3: Single Sample Download

```bash
# Download a specific accession
python scripts/download_genomic_data_automated.py \
    --accession ERR3239363
```

## 📊 Sample Pools Available

### European Ancestry (UK/Europe)
- **Reference**: 7 samples
  - ERR3239363, ERR3239372, ERR3239401, ERR3239428
  - ERR3239445, ERR3239512, ERR3239567
- **Query**: 3 samples
  - ERR3239276, ERR3239334, ERR3239454

### East Asian Ancestry (China/Japan/Korea)
- **Reference**: 6 samples
  - ERR3239578, ERR3239612, ERR3239634
  - ERR3239689, ERR3239701, ERR3239723
- **Query**: 1 sample
  - ERR3239745

### African Ancestry (Sub-Saharan Africa)
- **Reference**: 6 samples
  - ERR3239756, ERR3239778, ERR3239801
  - ERR3239823, ERR3239845, ERR3239867
- **Query**: 1 sample
  - ERR3239889

### South Asian Ancestry (India/Pakistan/Bangladesh)
- **Reference**: 6 samples
  - ERR3239912, ERR3239934, ERR3239956
  - ERR3239978, ERR3240001, ERR3240023
- **Query**: 1 sample
  - ERR3240045

## 💾 Storage Requirements

| Samples | Estimated Size | Notes |
|---------|----------------|-------|
| 1 sample | ~40 GB | Whole genome, 30× coverage |
| 3 samples | ~120 GB | Minimum for k=3 anonymity |
| 7 samples | ~280 GB | Recommended for k=7+ |
| 10 samples | ~400 GB | Good diversity |
| All pools (30 samples) | ~1.2 TB | Complete dataset |

**Note**: Sizes are after compression with pigz

## ⏱️ Download Times

Approximate times (depends on connection speed):

| Samples | Time (100 Mbps) | Time (1 Gbps) |
|---------|-----------------|---------------|
| 1 sample | 1-2 hours | 10-15 minutes |
| 3 samples | 3-6 hours | 30-45 minutes |
| 7 samples | 7-14 hours | 1-2 hours |
| 10 samples | 10-20 hours | 2-3 hours |

## 📁 Output Structure

```
data/
├── download_state.json          # Download progress tracking
└── downloaded/
    └── fastq/
        ├── european/
        │   ├── ERR3239363/
        │   │   ├── ERR3239363_1.fastq.gz  (~20 GB)
        │   │   └── ERR3239363_2.fastq.gz  (~20 GB)
        │   ├── ERR3239372/
        │   │   └── ...
        │   └── ...
        ├── east_asian/
        │   └── ...
        ├── african/
        │   └── ...
        └── south_asian/
            └── ...
```

## 🔧 Prerequisites

### Required Tools

```bash
# Install with conda
conda install -c bioconda sra-tools pigz

# Or with brew (macOS)
brew install sratoolkit pigz
```

### Check Installation

```bash
python scripts/download_genomic_data_automated.py --check-deps
```

Should output: `✅ All dependencies installed`

## 📈 Graphical Monitor

The monitor displays:

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                         🧬 GENOMEVAULT DATA DOWNLOAD MONITOR 🧬                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

┌─ 📊 OVERALL STATUS ─────────────────────────────────────────────────────────────────────────────┐
│  Pipeline Status: DOWNLOADING         │ Elapsed Time: 2.5h              │
│  Total Samples:   7                   │ Downloaded:   120.4 GB          │
│
│  [████████████████████████░░░░░░░░░░░░░░░░░░] 57.1%
│
│  ✅ Completed: 4    │  ⏳ Downloading: 1    │  ❌ Failed: 0    │  ⏸️  Queued: 2
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─ 💾 DISK SPACE ─────────────────────────────────────────────────────────────────────────────────┐
│  [██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░] 45.2%
│  Available: 234.5 GB    │ Used: 189.3 GB    │ Total: 423.8 GB
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─ 🔄 ACTIVE DOWNLOAD PROCESSES ─────────────────────────────────────────────────────────────────┐
│  📥 fasterq-dump   │ PID: 12345   │ CPU:  145.3% │ MEM:   8.2%  │
│     ERR3239445 --outdir data/downloaded/fastq/european/ERR3239445 --split-files...
│  🗜️  pigz           │ PID: 12389   │ CPU:  312.1% │ MEM:   1.5%  │
│     -p 8 data/downloaded/fastq/european/ERR3239428/ERR3239428_2.fastq
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─ 📦 SAMPLE DOWNLOAD STATUS ────────────────────────────────────────────────────────────────────┐
│  ✅ ERR3239363     │ european   │ ref   │ Completed    │ 38.4 GB    │ 1.2h     │
│  ✅ ERR3239372     │ european   │ ref   │ Completed    │ 41.2 GB    │ 1.3h     │
│  ✅ ERR3239401     │ european   │ ref   │ Completed    │ 39.8 GB    │ 1.2h     │
│  ⏳ ERR3239428     │ european   │ ref   │ Downloading  │ 21.0 GB    │ 0.6h     │
│  ⏸️  ERR3239445     │ european   │ ref   │ Queued       │ 0 GB       │          │
│  ⏸️  ERR3239512     │ european   │ ref   │ Queued       │ 0 GB       │          │
│  ⏸️  ERR3239567     │ european   │ ref   │ Queued       │ 0 GB       │          │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

⏱️  Auto-refreshing every 5 seconds... (Press Ctrl+C to stop)
📄 State file: data/download_state.json
```

## 🔍 Monitoring Options

```bash
# One-time status check
python scripts/monitor_genomic_downloads.py

# Auto-refresh every 5 seconds
python scripts/monitor_genomic_downloads.py --watch

# Custom refresh interval (10 seconds)
python scripts/monitor_genomic_downloads.py --watch --interval 10

# Monitor different state file
python scripts/monitor_genomic_downloads.py --state /path/to/state.json --watch
```

## 🛠️ Advanced Usage

### Resume Failed Downloads

The downloader automatically tracks state. If a download fails, simply run it again:

```bash
# Will skip completed samples and retry failed ones
python scripts/download_genomic_data_automated.py --pool european --samples 7
```

### Download Multiple Pools

```bash
# Download from all pools (5 samples each, 20 total)
python scripts/download_genomic_data_automated.py --pool all --samples 5
```

### Query Samples

```bash
# Download query samples instead of reference samples
python scripts/download_genomic_data_automated.py \
    --pool european \
    --type query
```

## 📋 State File Format

The `data/download_state.json` file tracks progress:

```json
{
  "start_time": "2025-10-24T18:30:00.123456",
  "status": "downloading",
  "total_downloaded_gb": 120.4,
  "samples": {
    "ERR3239363": {
      "accession": "ERR3239363",
      "sample_type": "reference",
      "pool": "european",
      "status": "completed",
      "start_time": "2025-10-24T18:30:15.456789",
      "end_time": "2025-10-24T19:42:33.123456",
      "size_gb": 38.4,
      "files": [
        "ERR3239363_1.fastq.gz",
        "ERR3239363_2.fastq.gz"
      ],
      "error": null
    }
  }
}
```

## 🐛 Troubleshooting

### "Missing required tools"
```bash
conda install -c bioconda sra-tools pigz
```

### "Insufficient disk space"
Free up space or use `--output-dir` to specify a different location:
```bash
python scripts/download_genomic_data_automated.py \
    --pool european \
    --samples 3 \
    --output-dir /mnt/external/genomic_data
```

### Download stuck or slow
- Check internet connection
- Check ENA/SRA server status
- Try downloading a single sample to test:
  ```bash
  python scripts/download_genomic_data_automated.py --accession ERR3239363
  ```

### Monitor shows "No download state file found"
Start a download first. The state file is created when downloads begin.

## 📚 Integration with GenomeVault Pipeline

After downloading, use the data with GenomeVault:

```bash
# Run complete privacy-preserving pipeline
python benchmarks/run_complete_privacy_pipeline.py \
    --reference-pool-fastq \
        data/downloaded/fastq/european/ERR3239363/ERR3239363_1.fastq.gz \
        data/downloaded/fastq/european/ERR3239363/ERR3239363_2.fastq.gz \
        data/downloaded/fastq/european/ERR3239372/ERR3239372_1.fastq.gz \
        data/downloaded/fastq/european/ERR3239372/ERR3239372_2.fastq.gz \
        data/downloaded/fastq/european/ERR3239401/ERR3239401_1.fastq.gz \
        data/downloaded/fastq/european/ERR3239401/ERR3239401_2.fastq.gz \
    --query-fastq \
        data/downloaded/fastq/european/ERR3239276/ERR3239276_1.fastq.gz \
        data/downloaded/fastq/european/ERR3239276/ERR3239276_2.fastq.gz
```

## 🎯 Recommended Workflow

1. **Start with small test** (1-3 samples):
   ```bash
   ./start_genomic_downloads.sh european 3
   ```

2. **Monitor progress**:
   - Watch the graphical dashboard
   - Check `data/download_state.json` for detailed status

3. **Scale up** after successful test:
   ```bash
   ./start_genomic_downloads.sh european 7
   ```

4. **Download diverse pools** for better k-anonymity:
   ```bash
   ./start_genomic_downloads.sh all 5
   ```

## 📞 Support

- Check logs in `logs/download_*.log`
- Review state file: `data/download_state.json`
- See main documentation: `CLAUDE.md`
- Data acquisition guides: `data/acquisition_plan/`

---

**Last Updated**: October 2025
**Status**: ✅ Tested and Working
