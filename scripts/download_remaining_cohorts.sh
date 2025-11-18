#!/bin/bash
# Download remaining ethnic cohorts (2 samples each)
# This script runs downloads sequentially to avoid overwhelming the network

set -e

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "==================================================================="
echo "GenomeVault: Download Remaining Ethnic Cohorts"
echo "==================================================================="
echo "This will download 2 samples from each of:"
echo "  - East Asian (ERR3239578, ERR3239612)"
echo "  - African (ERR3239756, ERR3239778)"
echo "  - South Asian (ERR3239912, ERR3239934)"
echo ""
echo "Total: 6 samples (~120-150 GB, 6-9 hours)"
echo "==================================================================="
echo ""

# East Asian
echo "[1/3] Starting East Asian downloads..."
python scripts/download_genomic_data_automated.py \
    --pool east_asian \
    --samples 2 \
    --type reference \
    2>&1 | tee "$LOG_DIR/download_east_asian_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ East Asian complete"
echo ""

# African
echo "[2/3] Starting African downloads..."
python scripts/download_genomic_data_automated.py \
    --pool african \
    --samples 2 \
    --type reference \
    2>&1 | tee "$LOG_DIR/download_african_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ African complete"
echo ""

# South Asian
echo "[3/3] Starting South Asian downloads..."
python scripts/download_genomic_data_automated.py \
    --pool south_asian \
    --samples 2 \
    --type reference \
    2>&1 | tee "$LOG_DIR/download_south_asian_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ South Asian complete"
echo ""

# Final summary
echo "==================================================================="
echo "All Downloads Complete!"
echo "==================================================================="
echo ""
echo "Final cohort summary:"
ls -lh data/downloaded/fastq/*/
echo ""
du -sh data/downloaded/fastq/*/
echo ""
echo "Total samples downloaded:"
find data/downloaded/fastq -type d -name "ERR*" | wc -l
echo ""
echo "==================================================================="
