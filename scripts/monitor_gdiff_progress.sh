#!/bin/bash
# k=11 GDiff Pipeline Progress Monitor
# Updated for multi-stage pipeline with SD card guide strands

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# Use argument or default to latest restart log
LOG_FILE="${1:-k11_pipeline_restart_$(ls -t k11_pipeline_restart_*.log 2>/dev/null | head -1 | sed 's/k11_pipeline_restart_//')}"
if [ ! -f "$LOG_FILE" ]; then
    # Try to find the latest log file
    LOG_FILE=$(ls -t k11_pipeline_restart_*.log 2>/dev/null | head -1)
    if [ -z "$LOG_FILE" ]; then
        LOG_FILE=$(ls -t k11_REAL_FIX.log 2>/dev/null | head -1)
    fi
fi

while true; do
    clear
    echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║        k=11 Privacy-Preserving Pipeline Monitor                ║${NC}"
    echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Log file: $LOG_FILE${NC}"
    echo -e "${CYAN}Guide strands: /Volumes/1TBStorage/guide_strands${NC}"
    echo ""

    # Detect current stage
    if grep -q "STAGE 1: Privacy-Preserving Alignment" "$LOG_FILE" 2>/dev/null && ! grep -q "STAGE 2: GDiff Differential Encoding" "$LOG_FILE" 2>/dev/null; then
        STAGE="STAGE 1: ALIGNMENT"

        # Check if building index or aligning
        if grep -q "Building minimap2 index" "$LOG_FILE" 2>/dev/null && ! grep -q "Aligning experimental reads" "$LOG_FILE" 2>/dev/null; then
            SUBSTAGE="Building minimap2 index"

            # Count guide pool assembly progress
            guides_added=$(grep "Adding guide" "$LOG_FILE" 2>/dev/null | wc -l | tr -d ' ')
            total_guides=11

            # Count minimap2 index chunks built
            index_chunks=$(grep -E "\[M::main::" "$LOG_FILE" 2>/dev/null | wc -l | tr -d ' ')

            echo -e "${BOLD}${GREEN}Stage: ALIGNMENT (Building Index)${NC}"
            echo -e "  Guides assembled: ${CYAN}${guides_added}/${total_guides}${NC}"
            echo -e "  Index chunks built: ${CYAN}${index_chunks}${NC}"

            # Get latest minimap2 stats
            latest_stats=$(grep -E "\[M::mm_idx_stat\]" "$LOG_FILE" 2>/dev/null | tail -1)
            if [ ! -z "$latest_stats" ]; then
                seqs=$(echo "$latest_stats" | grep -oE "#seq: [0-9]+" | grep -oE "[0-9]+")
                minimizers=$(echo "$latest_stats" | grep -oE "distinct minimizers: [0-9]+" | grep -oE "[0-9]+")
                length=$(echo "$latest_stats" | grep -oE "total length: [0-9]+" | grep -oE "[0-9]+")

                echo ""
                echo -e "${BOLD}Index Statistics:${NC}"
                echo -e "  Sequences: ${CYAN}${seqs}${NC}"
                echo -e "  Distinct k-mers: ${CYAN}${minimizers}${NC}"
                echo -e "  Total length: ${CYAN}$((length / 1000000000)) GB${NC}"
            fi
        elif grep -q "Aligning experimental reads" "$LOG_FILE" 2>/dev/null; then
            SUBSTAGE="Aligning reads"

            echo -e "${BOLD}${GREEN}Stage: ALIGNMENT (Aligning Reads)${NC}"

            # Check for minimap2 alignment progress (mapped/unmapped)
            mapped=$(grep -E "\[M::main\] mapped" "$LOG_FILE" 2>/dev/null | tail -1)
            if [ ! -z "$mapped" ]; then
                echo -e "  ${CYAN}${mapped}${NC}"
            else
                echo -e "  ${YELLOW}Processing...${NC}"
            fi
        fi

    elif grep -q "STAGE 2: GDiff Differential Encoding" "$LOG_FILE" 2>/dev/null; then
        STAGE="STAGE 2: GDIFF ENCODING"

        # Detect total chunks from latest run initialization
        total_chunks_line=$(grep "Total chunks:" "$LOG_FILE" 2>/dev/null | tail -1)
        detected_total=$(echo "$total_chunks_line" | grep -oE "[0-9]+ \(will" | awk '{print $1}')

        if [ -z "$detected_total" ]; then
            detected_total="790"  # Default estimate
        fi

        # Get latest progress from CURRENT run only
        latest_line=$(grep -E "✓.*\[[0-9]+/${detected_total}\]" "$LOG_FILE" 2>/dev/null | tail -1)

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

            # Calculate elapsed time
            start_line=$(grep -E "✓.*\[[0-9]+/${detected_total}\]" $LOG_FILE 2>/dev/null | head -1)
            start_time=$(echo "$start_line" | awk '{print $1, $2}')
            if [ ! -z "$start_time" ]; then
                start_epoch=$(date -j -f "%Y-%m-%d %H:%M:%S" "$start_time" +%s 2>/dev/null)
            else
                init_line=$(grep "Total chunks:" $LOG_FILE 2>/dev/null | tail -1)
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
                eta_secs=${eta_secs%.*}
                if [ ! -z "$eta_secs" ]; then
                    eta_hours=$((eta_secs / 3600))
                    eta_mins=$(((eta_secs % 3600) / 60))
                fi
            fi

            # Render progress bar
            bar_width=50
            filled=$(echo "scale=0; $progress_pct * $bar_width / 100" | bc 2>/dev/null)
            filled=${filled%.*}
            if [ ! -z "$filled" ] && [ $filled -gt 0 ]; then
                bar=$(printf "%-${bar_width}s" "$(printf '█%.0s' $(seq 1 $filled 2>/dev/null) 2>/dev/null)")
                bar_display=$(echo "$bar" | sed 's/ /░/g')
            else
                bar_display=$(printf '░%.0s' $(seq 1 $bar_width 2>/dev/null) 2>/dev/null)
            fi

            echo -e "${BOLD}${GREEN}Stage: GDIFF ENCODING${NC}"
            echo ""
            echo -e "${BOLD}Progress:${NC}"
            echo -e "  ${GREEN}[${bar_display}]${NC} ${CYAN}${progress_pct}%${NC}"
            echo ""
            echo -e "${BOLD}Status:${NC}"
            echo -e "  Chunks:        ${CYAN}${current}/${total}${NC}"
            echo -e "  Chromosome:    ${CYAN}${chrom}${NC}"
            echo -e "  Last variants: ${CYAN}${variant_count}${NC}"
            echo ""
            echo -e "${BOLD}Timing:${NC}"
            echo -e "  Elapsed:       ${CYAN}${elapsed_hours}h ${elapsed_mins}m${NC}"
            if [ ! -z "$eta_hours" ]; then
                echo -e "  ETA:           ${YELLOW}${eta_hours}h ${eta_mins}m${NC}"
            fi
            echo -e "  Last update:   ${CYAN}${timestamp}${NC}"
        else
            echo -e "${BOLD}${GREEN}Stage: GDIFF ENCODING${NC}"
            echo -e "${YELLOW}Waiting for chunk processing to start...${NC}"
        fi

    elif grep -q "STAGE 3: HDC Encoding" "$LOG_FILE" 2>/dev/null; then
        echo -e "${BOLD}${GREEN}Stage: HDC ENCODING${NC}"

        # Extract HDC statistics
        total_variants=$(grep "Total variants:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE "[0-9,]+" | tr -d ',')
        hdc_dim=$(grep "Hypervector dimension:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE "[0-9,]+")
        backend=$(grep "Backend:" "$LOG_FILE" 2>/dev/null | tail -1 | awk -F': ' '{print $2}')

        echo ""
        echo -e "${BOLD}Status:${NC}"
        echo -e "  Total variants: ${CYAN}${total_variants}${NC}"
        echo -e "  HDV dimension:  ${CYAN}${hdc_dim}${NC}"
        echo -e "  Backend:        ${CYAN}${backend}${NC}"

    elif grep -q "PIPELINE COMPLETE" "$LOG_FILE" 2>/dev/null; then
        echo -e "${BOLD}${GREEN}✓ PIPELINE COMPLETE!${NC}"
        echo ""

        # Show final statistics
        results_file="data/experimental_strands/ERR3239334/encoding/k12_pipeline_results.json"
        if [ -f "$results_file" ]; then
            echo -e "${BOLD}Final Results:${NC}"
            cat "$results_file" | grep -E '"total_variants"|"hdc_dimension"|"hdc_backend"|"hdc_size_kb"' | sed 's/^/  /'
        fi
    else
        echo -e "${YELLOW}Waiting for pipeline to start...${NC}"
    fi

    echo ""
    echo -e "${BOLD}Recent Activity:${NC}"
    tail -8 "$LOG_FILE" 2>/dev/null | grep -E "INFO|✓|Processing|\[M::" | tail -5 | sed 's/^/  /' | cut -c 1-120

    echo ""
    echo -e "${CYAN}[Auto-refresh: 5s] [Ctrl+C to exit]${NC}"
    sleep 5
done
