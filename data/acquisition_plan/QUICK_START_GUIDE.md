# GenomeVault Data Acquisition - Quick Start Guide

## 🎯 Objective
Scale GenomeVault testing from k=3 to k=10+ with diverse, well-organized genomic datasets.

---

## 📋 Prerequisites

### 1. Install Required Tools

```bash
# SRA Toolkit (for downloading data)
conda install -c bioconda sra-tools

# Quality control tools
conda install -c bioconda fastqc pigz

# Check installation
prefetch --version
fasterq-dump --version
fastqc --version
```

### 2. Check Disk Space

```bash
# Phase 1 only: ~276 GB
# Phases 1+2: ~1.2 TB
# All phases: ~1.8 TB

df -h data/
```

### 3. Make Scripts Executable

```bash
chmod +x scripts/create_data_structure.sh
chmod +x scripts/download_genomic_data.py
chmod +x scripts/generate_sample_metadata.py
chmod +x scripts/generate_pool_manifest.py
```

---

## 🚀 Phase 1: Scale to k=10 (European Ancestry)

**Goal**: Expand from 3 to 10 reference samples  
**Size**: ~276 GB  
**Time**: 1-2 days

### Step 1: Create Directory Structure

```bash
bash scripts/create_data_structure.sh
```

**Expected output**:
```
✓ Created european_ancestry reference pool directory
✓ Created east_asian_ancestry reference pool directory
✓ Created african_ancestry reference pool directory
✓ Created south_asian_ancestry reference pool directory
✓ Created query sample directories
✓ Created metadata directory
✓ Created processed data directories
```

### Step 2: Download Additional Reference Samples (7 samples)

```bash
# Reference pool samples
SAMPLES=(
  "ERR3239363"
  "ERR3239372"
  "ERR3239401"
  "ERR3239428"
  "ERR3239445"
  "ERR3239512"
  "ERR3239567"
)

OUTPUT_DIR="data/raw_fastq/reference_pools/european_ancestry/k10_pool_v1"

for accession in "${SAMPLES[@]}"; do
  echo "Downloading $accession..."
  
  # Download
  fasterq-dump "$accession" \
    --outdir "${OUTPUT_DIR}/${accession}" \
    --split-files \
    --threads 8 \
    --progress \
    --mem 8G
  
  # Compress
  pigz -p 8 "${OUTPUT_DIR}/${accession}"/*.fastq
  
  # Generate metadata
  python scripts/generate_sample_metadata.py \
    --accession "$accession" \
    --sample-dir "${OUTPUT_DIR}/${accession}" \
    --output "${OUTPUT_DIR}/${accession}/metadata.json" \
    --pool-assignment "european_ancestry/k10_pool_v1" \
    --role "reference"
  
  echo "✓ Completed $accession"
  echo ""
done
```

### Step 3: Download Query Samples (5 samples)

```bash
QUERIES=(
  "ERR3239285"
  "ERR3239398"
  "ERR3239421"
  "ERR3239489"
  "ERR3239534"
)

QUERY_DIR="data/raw_fastq/query_samples/baseline/european"

for accession in "${QUERIES[@]}"; do
  echo "Downloading query: $accession..."
  
  fasterq-dump "$accession" \
    --outdir "${QUERY_DIR}/${accession}" \
    --split-files \
    --threads 8 \
    --progress \
    --mem 8G
  
  pigz -p 8 "${QUERY_DIR}/${accession}"/*.fastq
  
  python scripts/generate_sample_metadata.py \
    --accession "$accession" \
    --sample-dir "${QUERY_DIR}/${accession}" \
    --output "${QUERY_DIR}/${accession}/metadata.json" \
    --test-scenario "baseline" \
    --role "query"
  
  echo "✓ Completed query $accession"
  echo ""
done
```

### Step 4: Copy Existing Samples (3 samples)

```bash
# Your existing samples are already downloaded
# Just move/copy them to the new structure

EXISTING_REFS=(
  "ERR3239276"
  "ERR3239454"
  "ERR3239475"
)

EXISTING_QUERY="ERR3239334"

for accession in "${EXISTING_REFS[@]}"; do
  # If they're in the root directory
  if [ -d "$accession" ]; then
    echo "Moving existing reference: $accession"
    cp -r "$accession" "${OUTPUT_DIR}/"
    
    # Generate metadata if it doesn't exist
    if [ ! -f "${OUTPUT_DIR}/${accession}/metadata.json" ]; then
      python scripts/generate_sample_metadata.py \
        --accession "$accession" \
        --sample-dir "${OUTPUT_DIR}/${accession}" \
        --output "${OUTPUT_DIR}/${accession}/metadata.json" \
        --pool-assignment "european_ancestry/k10_pool_v1" \
        --role "reference"
    fi
  fi
done

# Move existing query
if [ -d "$EXISTING_QUERY" ]; then
  echo "Moving existing query: $EXISTING_QUERY"
  cp -r "$EXISTING_QUERY" "${QUERY_DIR}/"
  
  if [ ! -f "${QUERY_DIR}/${EXISTING_QUERY}/metadata.json" ]; then
    python scripts/generate_sample_metadata.py \
      --accession "$EXISTING_QUERY" \
      --sample-dir "${QUERY_DIR}/${EXISTING_QUERY}" \
      --output "${QUERY_DIR}/${EXISTING_QUERY}/metadata.json" \
      --test-scenario "baseline" \
      --role "query"
  fi
fi
```

### Step 5: Generate Pool Manifest

```bash
python scripts/generate_pool_manifest.py \
  --pool-id "european_ancestry_k10_pool_v1" \
  --pool-dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/pool_manifest.json" \
  --k 10
```

### Step 6: Validate

```bash
# Check all samples are present
ls -la ${OUTPUT_DIR}/

# Should show 10 directories (one per sample)

# Check pool manifest
cat ${OUTPUT_DIR}/pool_manifest.json | jq '.pool_config'

# Should show:
# {
#   "k_anonymity": 10,
#   "ancestry_group": "European",
#   "actual_samples": 10
# }
```

---

## 🎯 What You Get from Phase 1

After completing Phase 1, you will have:

✅ **k=10 European ancestry reference pool**
- 10 high-quality whole-genome samples (~230 GB)
- Consistent with your existing data (same study)
- Organized in standardized directory structure
- Complete metadata for each sample
- Pool manifest for easy management

✅ **6 European query samples**
- 5 new + 1 existing (~138 GB)
- Diverse complexity for testing
- Tagged by test scenario

✅ **Ready for production benchmarks**
- Test k=3 vs k=5 vs k=7 vs k=10
- Measure entropy scaling
- Validate cryptographic security claims

---

## 📊 Verification Checklist

Run these checks after Phase 1:

```bash
# 1. Count reference pool samples
find data/raw_fastq/reference_pools/european_ancestry/k10_pool_v1 -mindepth 1 -maxdepth 1 -type d | wc -l
# Expected: 10

# 2. Count query samples
find data/raw_fastq/query_samples/baseline/european -mindepth 1 -maxdepth 1 -type d | wc -l
# Expected: 6

# 3. Check total size
du -sh data/raw_fastq/
# Expected: ~370-400 GB (including existing data)

# 4. Verify all metadata files exist
find data/raw_fastq -name "metadata.json" | wc -l
# Expected: 16 (10 refs + 6 queries)

# 5. Verify FASTQ integrity
for dir in data/raw_fastq/reference_pools/european_ancestry/k10_pool_v1/*/; do
  accession=$(basename "$dir")
  echo "Checking $accession..."
  gzip -t "${dir}/${accession}_1.fastq.gz" && echo "  R1: OK" || echo "  R1: FAILED"
  gzip -t "${dir}/${accession}_2.fastq.gz" && echo "  R2: OK" || echo "  R2: FAILED"
done
```

---

## 🔄 Next Steps After Phase 1

### Option A: Run Benchmarks Immediately
```bash
# Process Layer 2 for all 10 references
# (This will take ~30 hours for 10 samples @ 3h each)

# Then run comprehensive k-scaling benchmark
genomevault test scenario k_scaling_validation \
  --pool european_ancestry_k10_pool_v1 \
  --k-values 3,5,7,10 \
  --output-report reports/k_scaling_benchmark_$(date +%Y%m%d).pdf
```

### Option B: Continue to Phase 2
```bash
# Add East Asian, African, and South Asian pools
# Follow similar process for each ancestry group
# See DATA_ACQUISITION_PLAN.md for details
```

---

## 📚 Key Files Reference

| File | Location | Purpose |
|------|----------|---------|
| **Master Plan** | `data/acquisition_plan/DATA_ACQUISITION_PLAN.md` | Complete documentation |
| **Directory Setup** | `scripts/create_data_structure.sh` | Initialize directories |
| **Sample Metadata** | `scripts/generate_sample_metadata.py` | Per-sample metadata |
| **Pool Manifest** | `scripts/generate_pool_manifest.py` | Pool-level aggregation |
| **Data Config** | `data_config.yaml` | API/CLI interface |

---

## 🆘 Troubleshooting

### Download Failures

**Problem**: `fasterq-dump` fails with connection error

**Solution**:
```bash
# Configure SRA toolkit cache
vdb-config --interactive

# Or use prefetch first (more robust)
prefetch ERR3239363
fasterq-dump ERR3239363 --split-files
```

### Insufficient Disk Space

**Problem**: Running out of space

**Solution**:
```bash
# Download samples incrementally
# Process and compress immediately after each download

# Or download to external drive
OUTPUT_DIR="/Volumes/ExternalDrive/genomevault_data"
```

### Slow Downloads

**Problem**: Downloads taking too long

**Solution**:
```bash
# Use Aspera for faster downloads (if available)
prefetch ERR3239363 --ascp-path $(which ascp)

# Or download multiple samples in parallel
# (but watch network bandwidth!)
```

---

## 💰 Cost Estimate for Phase 1

### Storage
- **276 GB new data** @ $0.023/GB/month = $6.35/month
- **Annual cost**: $76

### Compute (if using cloud)
- **12 samples** × 5 hours × $0.34/hour = $20.40 one-time

### Total Phase 1 Cost: ~$96/year

---

## 🎓 Best Practices

1. **Download samples sequentially** - easier to debug failures
2. **Generate metadata immediately** - don't wait until all downloads complete
3. **Validate checksums** - ensure data integrity from the start
4. **Back up original FASTQ** - before processing, make a backup copy
5. **Document everything** - keep notes in progress_tracking/

---

## ✅ Success Criteria

Phase 1 is complete when you can say YES to all of these:

- [ ] All 10 reference samples downloaded (ERR3239276, ERR3239454, ERR3239475, ERR3239363, ERR3239372, ERR3239401, ERR3239428, ERR3239445, ERR3239512, ERR3239567)
- [ ] All 6 query samples downloaded (ERR3239334, ERR3239285, ERR3239398, ERR3239421, ERR3239489, ERR3239534)
- [ ] Every sample has metadata.json
- [ ] Pool manifest generated successfully
- [ ] All FASTQ files pass gzip integrity check
- [ ] Total size ~370-400 GB
- [ ] Ready to process Layer 2 for k=10 pool

---

**Questions or issues?** Check the full plan: `data/acquisition_plan/DATA_ACQUISITION_PLAN.md`

**Ready to begin?** Start with: `bash scripts/create_data_structure.sh`
