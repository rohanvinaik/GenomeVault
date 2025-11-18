#!/bin/bash
# Download 2 samples from each ethnic group in parallel
# This script runs multiple download processes simultaneously

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================================"
echo "Multi-Ethnic Genomic Data Download"
echo "Starting: $(date)"
echo "======================================================================"
echo ""
echo "Target: 2 samples from each of 4 ethnic groups (8 total samples)"
echo "  - European: Already complete (3 samples)"
echo "  - East Asian: 1 complete, 1 in progress, will download 0 more"
echo "  - African: Downloading 2 samples"
echo "  - South Asian: Will download 2 samples"
echo ""
echo "Downloads will run sequentially within each ethnic group"
echo "But different ethnic groups will download in parallel"
echo "======================================================================"
echo ""

# African samples (already started - PID 81903)
echo "[1/2] African cohort download already running (PID 81903)"
echo "  → ERR3239756, ERR3239778"
echo ""

# Wait a few seconds to let African download get established
sleep 10

# South Asian samples (start now in parallel)
echo "[2/2] Starting South Asian cohort download..."
nohup python3 "$SCRIPT_DIR/download_genomic_data_automated.py" \
    --pool south_asian \
    --samples 2 \
    --type reference \
    > "$LOG_DIR/download_south_asian_${TIMESTAMP}.log" 2>&1 &
SOUTH_ASIAN_PID=$!
echo "  → South Asian download started (PID: $SOUTH_ASIAN_PID)"
echo "  → ERR3239912, ERR3239934"
echo ""

echo "======================================================================"
echo "Download processes started!"
echo "======================================================================"
echo ""
echo "Active download PIDs:"
echo "  - African:     81903"
echo "  - South Asian: $SOUTH_ASIAN_PID"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/download_african_*.log"
echo "  tail -f logs/download_south_asian_${TIMESTAMP}.log"
echo ""
echo "Or check all downloads:"
echo "  ps aux | grep download_genomic_data_automated"
echo "======================================================================"
