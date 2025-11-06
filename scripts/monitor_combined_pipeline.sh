#!/bin/bash
# Combined Monitor: k=12 Privacy Pipeline + Template Builder
# Tracks both processes running in parallel

# ANSI colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

clear

while true; do
    current_time=$(date +"%H:%M:%S")
    current_epoch=$(date +%s)

    # ========================================================================
    # k=12 PRIVACY PIPELINE (GDiff Encoding)
    # ========================================================================
    if ps aux | grep -E "run_k12_(privacy|gdiff)_pipeline.py" | grep -v grep > /dev/null; then
        k12_status="${GREEN}●${NC} RUNNING"
        k12_pid=$(ps aux | grep -E "run_k12_(privacy|gdiff)_pipeline.py" | grep -v grep | awk '{print $2}' | head -1)
        k12_cpu=$(ps aux | grep -E "run_k12_(privacy|gdiff)_pipeline.py" | grep -v grep | awk '{print $3}' | head -1)
        k12_mem=$(ps aux | grep -E "run_k12_(privacy|gdiff)_pipeline.py" | grep -v grep | awk '{print $4}' | head -1)
    else
        k12_status="${RED}✗${NC} NOT RUNNING"
        k12_pid="N/A"
        k12_cpu="0.0"
        k12_mem="0.0"
    fi

    # Check k=12 pipeline stage and parse GDiff progress
    minimap2_index=$(ps aux | grep "minimap2.*guide.*mmi" | grep -v grep | head -1)
    if [ ! -z "$minimap2_index" ]; then
        k12_stage="${GREEN}●${NC} Building indexes"
        guide_num=$(echo "$minimap2_index" | grep -o "guide[0-9]*" | grep -o "[0-9]*")
        k12_detail="  Index ${guide_num}/12"
        k12_cpu_active=$(echo "$minimap2_index" | awk '{print $3}')
        k12_eta_str="N/A"
        k12_progress_pct="0"
    elif ps aux | grep "minimap2.*-ax sr" | grep -v grep > /dev/null; then
        k12_stage="${GREEN}●${NC} Aligning chunks"
        k12_detail="  Random guide cycling"
        k12_cpu_active=$(ps aux | grep "minimap2.*-ax sr" | grep -v grep | awk '{print $3}' | head -1)
        k12_eta_str="N/A"
        k12_progress_pct="0"
    elif ps aux | grep "samtools" | grep -v grep > /dev/null; then
        k12_stage="${GREEN}●${NC} Processing BAMs"
        k12_detail="  Sorting/merging"
        k12_cpu_active=$(ps aux | grep "samtools" | grep -v grep | awk '{print $3}' | head -1)
        k12_eta_str="N/A"
        k12_progress_pct="0"
    else
        # Check for GDiff encoding progress
        if [ "$k12_status" != "${RED}✗${NC} NOT RUNNING" ]; then
            k12_stage="${GREEN}●${NC} GDiff encoding"

            # Parse chunk progress from k12_gdiff_pipeline.log
            progress_line=$(grep -E "✓.*\[[0-9]+/[0-9]+\]" k12_gdiff_pipeline.log 2>/dev/null | tail -1)
            latest_chunk=$(echo "$progress_line" | grep -oE "\[[0-9]+/[0-9]+\]" | sed 's/\[//' | sed 's/\]//' | cut -d'/' -f1)
            total_chunks=$(echo "$progress_line" | grep -oE "\[[0-9]+/[0-9]+\]" | sed 's/\[//' | sed 's/\]//' | cut -d'/' -f2)

            if [ -z "$latest_chunk" ] || [ -z "$total_chunks" ]; then
                # No progress yet - check if process is starting
                if ps -p $k12_pid > /dev/null 2>&1; then
                    k12_detail="  Processing chunks..."
                    k12_eta_str="calculating..."
                    k12_progress_pct="0"
                else
                    k12_detail="  Starting..."
                    k12_eta_str="N/A"
                    k12_progress_pct="0"
                fi
            else
                # Calculate progress
                k12_progress_pct=$(echo "scale=1; $latest_chunk * 100 / $total_chunks" | bc 2>/dev/null)

                # Get elapsed time from process
                elapsed_secs=$(ps -p $k12_pid -o etime= 2>/dev/null | awk -F: '{if (NF==3) print $1*3600+$2*60+$3; else if (NF==2) print $1*60+$2; else print $1}')

                if [ ! -z "$elapsed_secs" ] && [ "$elapsed_secs" -gt 0 ] && [ "$latest_chunk" -gt 0 ]; then
                    # Calculate chunks per second
                    chunks_per_sec=$(echo "scale=4; $latest_chunk / $elapsed_secs" | bc 2>/dev/null)
                    remaining_chunks=$((total_chunks - latest_chunk))

                    if [ ! -z "$chunks_per_sec" ] && (( $(echo "$chunks_per_sec > 0" | bc -l 2>/dev/null) )); then
                        eta_secs=$(echo "scale=0; $remaining_chunks / $chunks_per_sec" | bc 2>/dev/null)
                        eta_mins=$((eta_secs / 60))
                        eta_hours=$((eta_mins / 60))
                        eta_mins_display=$((eta_mins % 60))

                        if [ $eta_hours -gt 0 ]; then
                            k12_eta_str="${eta_hours}h ${eta_mins_display}m"
                        else
                            k12_eta_str="${eta_mins}m"
                        fi
                    else
                        k12_eta_str="calculating..."
                    fi
                else
                    k12_eta_str="calculating..."
                fi

                k12_detail="  ${latest_chunk}/${total_chunks} chunks (${k12_progress_pct}%)"
            fi

            k12_cpu_active=$(ps aux | grep -E "run_k12_(privacy|gdiff)_pipeline" | grep -v grep | awk '{print $3}' | head -1)
            if [ -z "$k12_cpu_active" ]; then
                k12_cpu_active="0.0"
            fi
        else
            k12_stage="${BLUE}●${NC} GDiff/HDC encoding"
            k12_detail="  Not started"
            k12_cpu_active="0.0"
            k12_eta_str="N/A"
            k12_progress_pct="0"
        fi
    fi

    # k=12 output files
    output_dir="benchmark_results/enhanced_privacy_k13_phase123_optimized"
    if [ -f "$output_dir/experimental.sorted.bam" ]; then
        k12_bam_size=$(du -h "$output_dir/experimental.sorted.bam" 2>/dev/null | awk '{print $1}')
        k12_bam_status="${GREEN}✓${NC} $k12_bam_size"
    else
        k12_bam_status="${YELLOW}⏳${NC} Building..."
    fi

    # ========================================================================
    # TEMPLATE BUILDER
    # ========================================================================
    if ps aux | grep "build_gdiff_template" | grep -v grep > /dev/null; then
        template_status="${GREEN}●${NC} RUNNING"
        template_pid=$(ps aux | grep "build_gdiff_template" | grep -v grep | awk '{print $2}')
        template_cpu=$(ps aux | grep "build_gdiff_template" | grep -v grep | awk '{print $3}')
        template_mem=$(ps aux | grep "build_gdiff_template" | grep -v grep | awk '{print $4}')

        # Get start time from log (first "Loading dbSNP" line)
        log_start_time=$(grep "Loading dbSNP (common_only=True)" template_build.log 2>/dev/null | head -1 | awk '{print $1, $2}')
        if [ ! -z "$log_start_time" ]; then
            # Convert log timestamp to epoch
            start_epoch=$(date -j -f "%Y-%m-%d %H:%M:%S" "$log_start_time" +%s 2>/dev/null)
            if [ ! -z "$start_epoch" ]; then
                elapsed_seconds=$((current_epoch - start_epoch))
            else
                elapsed_seconds=0
            fi
        else
            elapsed_seconds=0
        fi

        elapsed_minutes=$((elapsed_seconds / 60))
        elapsed_hours=$((elapsed_minutes / 60))
        elapsed_mins_display=$((elapsed_minutes % 60))

        # Check which file is open and calculate real ETA from progress
        open_file=$(lsof -p $template_pid 2>/dev/null | grep -E "\.vcf\.gz" | awk '{print $NF}' | tail -1)
        if echo "$open_file" | grep -q "dbsnp"; then
            template_stage="${GREEN}●${NC} Loading dbSNP"

            # Parse actual progress from log
            latest_count=$(grep "Loaded.*variants" template_build.log 2>/dev/null | tail -1 | sed 's/.*Loaded //' | grep -oE "[0-9,]+" | head -1 | tr -d ',')
            if [ ! -z "$latest_count" ] && [ "$latest_count" -gt 0 ] && [ "$elapsed_seconds" -gt 0 ]; then
                # Calculate rate: variants per second
                variants_per_sec=$(echo "scale=2; $latest_count / $elapsed_seconds" | bc 2>/dev/null)

                # Target: 70M variants
                target_variants=70000000
                remaining_variants=$((target_variants - latest_count))

                if [ ! -z "$variants_per_sec" ] && (( $(echo "$variants_per_sec > 0" | bc -l 2>/dev/null) )); then
                    eta_seconds=$(echo "scale=0; $remaining_variants / $variants_per_sec" | bc 2>/dev/null)
                    eta_minutes=$((eta_seconds / 60))
                    eta_hours=$((eta_minutes / 60))
                    eta_mins_display=$((eta_minutes % 60))

                    # Format progress
                    progress_pct=$(echo "scale=1; $latest_count * 100 / $target_variants" | bc 2>/dev/null)
                    variants_millions=$(echo "scale=1; $latest_count / 1000000" | bc 2>/dev/null)

                    template_detail="  ${variants_millions}M / 70M (${progress_pct}%)"

                    if [ $eta_hours -gt 0 ]; then
                        template_eta_str="${eta_hours}h ${eta_mins_display}m remaining"
                    else
                        template_eta_str="${eta_minutes}m remaining"
                    fi
                else
                    template_detail="  ~70M common variants"
                    template_eta_str="calculating..."
                fi
            else
                template_detail="  ~70M common variants (starting...)"
                template_eta_str="calculating..."
            fi
        elif echo "$open_file" | grep -q "clinvar"; then
            template_stage="${GREEN}●${NC} Loading ClinVar"
            template_detail="  ~4M clinical variants"
            template_eta_str="~2-3m remaining"
        elif echo "$open_file" | grep -q "gnomad"; then
            template_stage="${GREEN}●${NC} Loading gnomAD"
            template_detail="  Population frequencies"
            template_eta_str="~30-40m remaining"
        else
            template_stage="${GREEN}●${NC} Processing"
            template_detail="  Building template"
            template_eta_str="~1-2m remaining"
        fi

        template_elapsed="${elapsed_hours}h ${elapsed_mins_display}m"
    else
        template_status="${RED}✗${NC} NOT RUNNING"
        template_pid="N/A"
        template_cpu="0.0"
        template_mem="0.0"
        template_stage="${BLUE}●${NC} Idle"
        template_detail="  Not started"
        template_elapsed="N/A"
        template_eta_str="N/A"
    fi

    # Check template output
    if [ -f "data/templates/gdiff_template_GRCh38.json.gz" ]; then
        template_size=$(du -h "data/templates/gdiff_template_GRCh38.json.gz" 2>/dev/null | awk '{print $1}')
        template_file_status="${GREEN}✓${NC} $template_size"
    else
        template_file_status="${YELLOW}⏳${NC} Building..."
    fi

    # ========================================================================
    # DISPLAY
    # ========================================================================
    clear
    echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║          Combined Pipeline Monitor (Parallel Tasks)           ║${NC}"
    echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}═══ k=12 Privacy-Preserving Pipeline ═══${NC}"
    echo -e "  Status:    $k12_status (PID: ${CYAN}$k12_pid${NC})"
    echo -e "  Stage:     $k12_stage"
    echo -e "  Detail:    ${CYAN}$k12_detail${NC}"

    # Show ETA and progress bar if encoding
    if [ "$k12_eta_str" != "N/A" ] && [ "$k12_eta_str" != "calculating..." ] && [ ! -z "$k12_eta_str" ]; then
        echo -e "  ETA:       ${YELLOW}${k12_eta_str}${NC}"
    fi

    # Render progress bar
    if [ ! -z "$k12_progress_pct" ] && (( $(echo "$k12_progress_pct > 0" | bc -l 2>/dev/null || echo 0) )); then
        bar_width=50
        filled=$(echo "scale=0; $k12_progress_pct * $bar_width / 100" | bc 2>/dev/null || echo 0)
        if [ $filled -gt 0 ]; then
            bar=$(printf "%-${bar_width}s" "$(printf '█%.0s' $(seq 1 $filled 2>/dev/null) 2>/dev/null)")
            bar_display=$(echo "$bar" | sed 's/ /░/g')
            echo -e "  Progress:  ${GREEN}[${bar_display}]${NC} ${k12_progress_pct}%"
        fi
    fi

    if [ "$k12_cpu_active" != "0.0" ] && [ ! -z "$k12_cpu_active" ]; then
        echo -e "  CPU:       ${CYAN}${k12_cpu_active}%${NC}"
    fi
    echo -e "  Output:    $k12_bam_status"
    echo -e "  Privacy:   ${GREEN}Random guide cycling (k=12)${NC}"
    echo ""
    echo -e "${BOLD}═══ GDiff Template Builder ═══${NC}"
    echo -e "  Status:    $template_status (PID: ${CYAN}$template_pid${NC})"
    echo -e "  Stage:     $template_stage"
    echo -e "  Detail:    ${CYAN}$template_detail${NC}"
    if [ "$template_cpu" != "0.0" ] && [ "$template_cpu" != "N/A" ]; then
        echo -e "  CPU:       ${CYAN}${template_cpu}%${NC}"
        echo -e "  Memory:    ${CYAN}${template_mem}%${NC}"
    fi
    if [ "$template_elapsed" != "N/A" ]; then
        echo -e "  Elapsed:   ${CYAN}${template_elapsed}${NC}"
        echo -e "  ETA:       ${YELLOW}${template_eta_str}${NC}"
    fi
    echo -e "  Output:    $template_file_status"
    echo ""
    echo -e "${BOLD}═══ System Summary ═══${NC}"
    total_cpu=$(echo "$k12_cpu + $template_cpu" | bc 2>/dev/null || echo "N/A")
    if [ "$total_cpu" != "N/A" ]; then
        echo -e "  Total CPU: ${CYAN}${total_cpu}%${NC}"
    fi
    disk_free=$(df -h . | tail -1 | awk '{print $4}')
    echo -e "  Disk Free: ${CYAN}${disk_free}${NC}"
    echo ""
    echo -e "${BOLD}═══ Latest Logs ═══${NC}"
    echo -e "  ${CYAN}k=12 pipeline:${NC}"
    tail -2 k12_gdiff_pipeline.log 2>/dev/null | sed 's/^/    /' || tail -2 k12_privacy_pipeline.log 2>/dev/null | sed 's/^/    /' || echo "    (waiting...)"
    echo ""
    echo -e "${CYAN}[Time: $current_time] [Auto-refresh: 5s] [Ctrl+C to exit]${NC}"

    sleep 5
done
