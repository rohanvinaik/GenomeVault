#!/bin/bash
# Simple GDiff Pipeline Progress Monitor

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

while true; do
    clear
    echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║             GDiff Encoding Progress Monitor                    ║${NC}"
    echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Detect total chunks from latest run initialization
    total_chunks_line=$(grep "Total chunks:" k12_gdiff_pipeline.log 2>/dev/null | tail -1)
    detected_total=$(echo "$total_chunks_line" | grep -oE "[0-9]+ \(will" | awk '{print $1}')

    if [ -z "$detected_total" ]; then
        detected_total="1301"  # Default to latest known total
    fi

    # Get latest progress from CURRENT run only
    latest_line=$(grep -E "✓.*\[[0-9]+/${detected_total}\]" k12_gdiff_pipeline.log 2>/dev/null | tail -1)

    if [ ! -z "$latest_line" ]; then
        # Extract chunk numbers
        current=$(echo "$latest_line" | grep -oE "\[[0-9]+/${detected_total}\]" | sed 's/\[//' | sed 's/\]//' | cut -d'/' -f1)
        total=$detected_total

        # Calculate progress percentage
        progress_pct=$(echo "scale=1; $current * 100 / $total" | bc 2>/dev/null)

        # Extract chromosome and variant count
        chrom=$(echo "$latest_line" | grep -oE "chr[0-9XY]+_consensus" | head -1)
        variant_count=$(echo "$latest_line" | grep -oE "[0-9]+ variants" | grep -oE "[0-9]+")

        # Get timestamp from log
        timestamp=$(echo "$latest_line" | awk '{print $1, $2}')

        # Calculate elapsed time (auto-detect start from first chunk of CURRENT run)
        start_line=$(grep -E "✓.*\[[0-9]+/${detected_total}\]" k12_gdiff_pipeline.log 2>/dev/null | head -1)
        start_time=$(echo "$start_line" | awk '{print $1, $2}')
        if [ ! -z "$start_time" ]; then
            start_epoch=$(date -j -f "%Y-%m-%d %H:%M:%S" "$start_time" +%s 2>/dev/null)
        else
            # Fallback to "Total chunks" initialization line from current run
            init_line=$(grep "Total chunks:" k12_gdiff_pipeline.log 2>/dev/null | tail -1)
            start_time=$(echo "$init_line" | awk '{print $1, $2}')
            start_epoch=$(date -j -f "%Y-%m-%d %H:%M:%S" "$start_time" +%s 2>/dev/null)
        fi
        current_epoch=$(date +%s)
        elapsed_secs=$((current_epoch - start_epoch))
        elapsed_hours=$((elapsed_secs / 3600))
        elapsed_mins=$(((elapsed_secs % 3600) / 60))

        # Calculate ETA
        if [ "$current" -gt 0 ]; then
            secs_per_chunk=$(echo "scale=2; $elapsed_secs / $current" | bc 2>/dev/null)
            remaining_chunks=$((total - current))
            eta_secs=$(echo "scale=0; $remaining_chunks * $secs_per_chunk" | bc 2>/dev/null)
            eta_secs=${eta_secs%.*}  # Remove decimal if present
            if [ ! -z "$eta_secs" ]; then
                eta_hours=$((eta_secs / 3600))
                eta_mins=$(((eta_secs % 3600) / 60))
            fi
        fi

        # Render progress bar
        bar_width=50
        filled=$(echo "scale=0; $progress_pct * $bar_width / 100" | bc 2>/dev/null)
        filled=${filled%.*}  # Remove decimal if present
        if [ ! -z "$filled" ] && [ $filled -gt 0 ]; then
            bar=$(printf "%-${bar_width}s" "$(printf '█%.0s' $(seq 1 $filled 2>/dev/null) 2>/dev/null)")
            bar_display=$(echo "$bar" | sed 's/ /░/g')
        else
            bar_display=$(printf '░%.0s' $(seq 1 $bar_width 2>/dev/null) 2>/dev/null)
        fi

        echo -e "${BOLD}Progress:${NC}"
        echo -e "  ${GREEN}[${bar_display}]${NC} ${CYAN}${progress_pct}%${NC}"
        echo ""
        echo -e "${BOLD}Status:${NC}"
        echo -e "  Chunks:       ${CYAN}${current}/${total}${NC}"
        echo -e "  Chromosome:   ${CYAN}${chrom}${NC}"
        echo -e "  Last variants: ${CYAN}${variant_count}${NC}"
        echo ""
        echo -e "${BOLD}Timing:${NC}"
        echo -e "  Elapsed:      ${CYAN}${elapsed_hours}h ${elapsed_mins}m${NC}"
        if [ ! -z "$eta_hours" ]; then
            echo -e "  ETA:          ${YELLOW}${eta_hours}h ${eta_mins}m${NC}"
        fi
        echo -e "  Last update:  ${CYAN}${timestamp}${NC}"
        echo ""
        echo -e "${BOLD}Recent Activity:${NC}"
        tail -5 k12_gdiff_pipeline.log | grep -E "✓|Processing" | sed 's/^/  /'
    else
        echo -e "${YELLOW}Waiting for pipeline to start...${NC}"
        echo ""
        echo "Log file: k12_gdiff_pipeline.log"
    fi

    echo ""
    echo -e "${CYAN}[Auto-refresh: 5s] [Ctrl+C to exit]${NC}"
    sleep 5
done
