#!/bin/bash
#
# GenomeVault Genomic Data Downloader - Quick Start
#
# This script starts the automated genomic data downloader
# and launches a real-time graphical monitor.
#
# Usage:
#   ./start_genomic_downloads.sh                    # Download 3 European samples
#   ./start_genomic_downloads.sh european 7         # Download 7 European samples
#   ./start_genomic_downloads.sh all 5              # Download 5 from each pool
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
POOL=${1:-european}
SAMPLES=${2:-3}

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}   🧬 GenomeVault Genomic Data Acquisition Pipeline 🧬        ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo

# Check dependencies
echo -e "${YELLOW}[1/3]${NC} Checking dependencies..."
if ! python scripts/download_genomic_data_automated.py --check-deps 2>&1 | grep -q "✅"; then
    echo -e "${RED}❌ Missing dependencies. Please install:${NC}"
    echo "  conda install -c bioconda sra-tools pigz"
    exit 1
fi

echo -e "${GREEN}✅${NC} Dependencies OK"
echo

# Check disk space
echo -e "${YELLOW}[2/3]${NC} Checking disk space..."
REQUIRED_GB=$((SAMPLES * 40))
AVAILABLE_GB=$(df -BG data 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//' || echo "1000")

echo "  Required: ${REQUIRED_GB} GB"
echo "  Available: ${AVAILABLE_GB} GB"

if [ "$AVAILABLE_GB" -lt "$REQUIRED_GB" ]; then
    echo -e "${YELLOW}⚠️  Warning: May run out of disk space!${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
fi

echo -e "${GREEN}✅${NC} Disk space OK"
echo

# Start download in background
echo -e "${YELLOW}[3/3]${NC} Starting download pipeline..."
echo "  Pool: ${POOL}"
echo "  Samples: ${SAMPLES}"
echo

# Create log directory
mkdir -p logs

# Start downloader in background
python scripts/download_genomic_data_automated.py \
    --pool "$POOL" \
    --samples "$SAMPLES" \
    --type reference \
    > logs/download_$(date +%Y%m%d_%H%M%S).log 2>&1 &

DOWNLOAD_PID=$!
echo -e "${GREEN}✅${NC} Download started (PID: $DOWNLOAD_PID)"
echo "  Log: logs/download_$(date +%Y%m%d_%H%M%S).log"
echo

# Wait a moment for state file to be created
sleep 2

# Check if download is still running
if ! ps -p $DOWNLOAD_PID > /dev/null; then
    echo -e "${RED}❌ Download failed to start. Check logs/${NC}"
    exit 1
fi

# Launch monitor
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Download pipeline is running!${NC}"
echo
echo "Starting graphical monitor..."
echo "Press Ctrl+C to stop the monitor (download will continue in background)"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo

# Give user a moment to read
sleep 2

# Start monitor (this will run until Ctrl+C)
python scripts/monitor_genomic_downloads.py --watch --interval 5

# After monitor exits
echo
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Monitor stopped.${NC}"
echo
echo "Download is still running in background (PID: $DOWNLOAD_PID)"
echo
echo "To check status again:"
echo "  python scripts/monitor_genomic_downloads.py --watch"
echo
echo "To stop the download:"
echo "  kill $DOWNLOAD_PID"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
