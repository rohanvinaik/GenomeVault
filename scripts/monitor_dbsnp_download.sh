#!/bin/bash
# Monitor dbSNP Download Progress
# Tracks wget download of dbSNP b156 for GRCh38

TARGET_FILE="data/public_genomics/dbsnp_b156_GRCh38.vcf.gz"
TARGET_SIZE_GB=8.5  # Expected final size: ~8.5 GB
LOG_FILE="dbsnp_download.log"

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

clear
echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         dbSNP b156 Download Monitor (GRCh38)                  ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to get file size in GB
get_size_gb() {
    if [ -f "$1" ]; then
        size_bytes=$(stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null)
        echo "scale=2; $size_bytes / 1024 / 1024 / 1024" | bc
    else
        echo "0"
    fi
}

# Function to calculate download speed
last_size=0
last_time=$(date +%s)

while true; do
    current_time=$(date +%s)

    # Check if wget is still running
    if ps aux | grep -E "wget.*dbsnp.*GCF_000001405" | grep -v grep > /dev/null; then
        status="${GREEN}●${NC} DOWNLOADING"
    else
        if [ -f "$TARGET_FILE" ]; then
            status="${GREEN}✓${NC} COMPLETE"
        else
            status="${RED}✗${NC} NOT RUNNING"
        fi
    fi

    # Get current file size
    if [ -f "$TARGET_FILE" ]; then
        current_size_gb=$(get_size_gb "$TARGET_FILE")
        percent=$(echo "scale=1; $current_size_gb * 100 / $TARGET_SIZE_GB" | bc)

        # Calculate speed
        current_size_bytes=$(stat -f%z "$TARGET_FILE" 2>/dev/null || stat -c%s "$TARGET_FILE" 2>/dev/null)
        time_diff=$((current_time - last_time))

        if [ $time_diff -gt 0 ] && [ $last_size -gt 0 ]; then
            bytes_diff=$((current_size_bytes - last_size))
            speed_mb=$(echo "scale=2; $bytes_diff / 1024 / 1024 / $time_diff" | bc)

            # Calculate ETA
            remaining_bytes=$(echo "($TARGET_SIZE_GB * 1024 * 1024 * 1024) - $current_size_bytes" | bc)
            if (( $(echo "$speed_mb > 0" | bc -l) )); then
                eta_seconds=$(echo "$remaining_bytes / ($speed_mb * 1024 * 1024)" | bc)
                eta_hours=$(echo "scale=0; $eta_seconds / 3600" | bc)
                eta_mins=$(echo "scale=0; ($eta_seconds % 3600) / 60" | bc)
                eta="${eta_hours}h ${eta_mins}m"
            else
                eta="calculating..."
            fi
        else
            speed_mb="0.00"
            eta="calculating..."
        fi

        last_size=$current_size_bytes
        last_time=$current_time
    else
        current_size_gb="0.00"
        percent="0.0"
        speed_mb="0.00"
        eta="N/A"
    fi

    # Display status
    clear
    echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║         dbSNP b156 Download Monitor (GRCh38)                  ║${NC}"
    echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Download Information:${NC}"
    echo -e "  Database:  ${CYAN}dbSNP build 156${NC}"
    echo -e "  Genome:    ${CYAN}GRCh38${NC}"
    echo -e "  Source:    ${CYAN}NCBI FTP${NC}"
    echo -e "  Target:    ${CYAN}~70M common variants${NC}"
    echo ""
    echo -e "${BOLD}Progress:${NC}"
    echo -e "  Status:    $status"
    echo -e "  Size:      ${BLUE}${current_size_gb} GB${NC} / ${TARGET_SIZE_GB} GB"
    echo -e "  Progress:  ${YELLOW}${percent}%${NC}"

    # Progress bar
    bar_width=50
    filled=$(echo "scale=0; $percent * $bar_width / 100" | bc)
    bar=$(printf "%-${bar_width}s" "$(printf '█%.0s' $(seq 1 $filled))")
    echo -e "  ${GREEN}[${bar// /░}]${NC}"

    echo ""
    echo -e "${BOLD}Performance:${NC}"
    echo -e "  Speed:     ${CYAN}${speed_mb} MB/s${NC}"
    echo -e "  ETA:       ${YELLOW}${eta}${NC}"
    echo ""
    echo -e "${BOLD}File:${NC}"
    echo -e "  Path:      ${CYAN}${TARGET_FILE}${NC}"

    # Check disk space
    disk_avail=$(df -h data/public_genomics 2>/dev/null | tail -1 | awk '{print $4}')
    if [ ! -z "$disk_avail" ]; then
        echo -e "  Disk Free: ${CYAN}${disk_avail}${NC}"
    fi

    echo ""
    echo -e "${BOLD}Processes:${NC}"

    # Check wget process
    wget_proc=$(ps aux | grep -E "wget.*dbsnp.*GCF_000001405" | grep -v grep | head -1)
    if [ ! -z "$wget_proc" ]; then
        cpu=$(echo "$wget_proc" | awk '{print $3}')
        mem=$(echo "$wget_proc" | awk '{print $4}')
        echo -e "  wget:      ${GREEN}Running${NC} (CPU: ${cpu}%, MEM: ${mem}%)"
    else
        echo -e "  wget:      ${RED}Not running${NC}"
    fi

    echo ""
    echo -e "${BOLD}Next Steps:${NC}"
    if ps aux | grep -E "wget.*dbsnp" | grep -v grep > /dev/null; then
        echo -e "  ${CYAN}→${NC} Download in progress..."
        echo -e "  ${CYAN}→${NC} Once complete, rebuild template with:"
        echo -e "    ${YELLOW}python3 scripts/build_gdiff_template.py${NC}"
    else
        if [ -f "$TARGET_FILE" ]; then
            file_size=$(get_size_gb "$TARGET_FILE")
            if (( $(echo "$file_size > 7.0" | bc -l) )); then
                echo -e "  ${GREEN}✓${NC} Download complete!"
                echo -e "  ${CYAN}→${NC} Run: ${YELLOW}python3 scripts/build_gdiff_template.py${NC}"
                echo -e "  ${CYAN}→${NC} Expected: ~70M common variants"
                echo -e "  ${CYAN}→${NC} Build time: ~15-30 minutes"
            else
                echo -e "  ${RED}⚠${NC}  File exists but may be incomplete (${file_size} GB < 7 GB)"
            fi
        else
            echo -e "  ${RED}✗${NC} Download not started or failed"
            echo -e "  ${CYAN}→${NC} Check download process"
        fi
    fi

    echo ""
    echo -e "${CYAN}[Auto-refresh: 5s] [Ctrl+C to exit]${NC}"

    # Wait 5 seconds before next update
    sleep 5
done
