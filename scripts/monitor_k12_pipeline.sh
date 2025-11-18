#!/bin/bash
# Monitor k=12 Privacy-Preserving Pipeline
# Tracks index building, alignment, and encoding stages

# ANSI colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

clear
echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     k=12 Privacy-Preserving Pipeline Monitor                  ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

while true; do
    current_time=$(date +"%H:%M:%S")

    # Check if pipeline is running
    if ps aux | grep "run_k12_privacy_pipeline.py" | grep -v grep > /dev/null; then
        pipeline_status="${GREEN}●${NC} RUNNING"
        pid=$(ps aux | grep "run_k12_privacy_pipeline.py" | grep -v grep | awk '{print $2}')
        cpu=$(ps aux | grep "run_k12_privacy_pipeline.py" | grep -v grep | awk '{print $3}')
        mem=$(ps aux | grep "run_k12_privacy_pipeline.py" | grep -v grep | awk '{print $4}')
    else
        pipeline_status="${RED}✗${NC} NOT RUNNING"
        pid="N/A"
        cpu="0.0"
        mem="0.0"
    fi

    # Check for minimap2 index building
    minimap2_proc=$(ps aux | grep "minimap2.*guide.*mmi" | grep -v grep | head -1)
    if [ ! -z "$minimap2_proc" ]; then
        current_stage="${GREEN}●${NC} Building minimap2 indexes"
        minimap2_cpu=$(echo "$minimap2_proc" | awk '{print $3}')
        minimap2_mem=$(echo "$minimap2_proc" | awk '{print $4}')

        # Extract which guide is being indexed
        guide_num=$(echo "$minimap2_proc" | grep -o "guide[0-9]*" | grep -o "[0-9]*")
        if [ -z "$guide_num" ]; then
            guide_num="?"
        fi
        stage_detail="  Index ${guide_num}/12"
    else
        # Check for alignment stage
        if ps aux | grep "minimap2.*-ax sr" | grep -v grep > /dev/null; then
            current_stage="${GREEN}●${NC} Aligning chunks to random guides"
            stage_detail="  Random guide cycling active"
            minimap2_cpu=$(ps aux | grep "minimap2.*-ax sr" | grep -v grep | awk '{print $3}' | head -1)
            minimap2_mem=$(ps aux | grep "minimap2.*-ax sr" | grep -v grep | awk '{print $4}' | head -1)
        elif ps aux | grep "samtools" | grep -v grep > /dev/null; then
            current_stage="${GREEN}●${NC} Post-processing BAMs"
            stage_detail="  Sorting/merging chunk BAMs"
            minimap2_cpu="N/A"
            minimap2_mem="N/A"
        else
            current_stage="${BLUE}●${NC} Waiting/Encoding"
            stage_detail="  Check log for details"
            minimap2_cpu="0.0"
            minimap2_mem="0.0"
        fi
    fi

    # Count temp guide indexes
    temp_dir=$(find /var/folders -name "tmp*" -type d 2>/dev/null | grep "$(whoami)" | head -1)
    if [ ! -z "$temp_dir" ]; then
        guide_indexes=$(find "$temp_dir" -name "guide*.mmi" 2>/dev/null | wc -l | tr -d ' ')
    else
        guide_indexes="0"
    fi

    # Check output files
    output_dir="benchmark_results/enhanced_privacy_k13_phase123_optimized"
    if [ -f "$output_dir/experimental.sorted.bam" ]; then
        exp_bam_size=$(du -h "$output_dir/experimental.sorted.bam" 2>/dev/null | awk '{print $1}')
        exp_bam_status="${GREEN}✓${NC} $exp_bam_size"
    else
        exp_bam_status="${YELLOW}⏳${NC} Building..."
    fi

    if [ -f "$output_dir/experimental.gdiff.gz" ]; then
        gdiff_size=$(du -h "$output_dir/experimental.gdiff.gz" 2>/dev/null | awk '{print $1}')
        gdiff_status="${GREEN}✓${NC} $gdiff_size"
    else
        gdiff_status="${YELLOW}⏳${NC} Pending..."
    fi

    if [ -f "$output_dir/experimental_hypervector.npy" ]; then
        hv_size=$(du -h "$output_dir/experimental_hypervector.npy" 2>/dev/null | awk '{print $1}')
        hv_status="${GREEN}✓${NC} $hv_size"
    else
        hv_status="${YELLOW}⏳${NC} Pending..."
    fi

    # Display status
    clear
    echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║     k=12 Privacy-Preserving Pipeline Monitor                  ║${NC}"
    echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Pipeline Status:${NC}"
    echo -e "  Status:    $pipeline_status"
    echo -e "  PID:       ${CYAN}$pid${NC}"
    echo -e "  CPU:       ${CYAN}${cpu}%${NC}"
    echo -e "  Memory:    ${CYAN}${mem}%${NC}"
    echo ""
    echo -e "${BOLD}Current Stage:${NC}"
    echo -e "  Stage:     $current_stage"
    echo -e "  Detail:    ${CYAN}$stage_detail${NC}"
    if [ "$minimap2_cpu" != "N/A" ] && [ "$minimap2_cpu" != "0.0" ]; then
        echo -e "  CPU:       ${CYAN}${minimap2_cpu}%${NC}"
        echo -e "  Memory:    ${CYAN}${minimap2_mem}%${NC}"
    fi
    echo ""
    echo -e "${BOLD}Privacy Architecture:${NC}"
    echo -e "  Method:    ${CYAN}Random guide cycling${NC}"
    echo -e "  Guides:    ${CYAN}12 separate indexes${NC}"
    echo -e "  Indexes:   ${CYAN}${guide_indexes}/12 built${NC}"
    echo -e "  Anonymity: ${GREEN}k=12${NC}"
    echo ""
    echo -e "${BOLD}Output Files:${NC}"
    echo -e "  BAM:       $exp_bam_status"
    echo -e "  GDiff:     $gdiff_status"
    echo -e "  HDV:       $hv_status"
    echo ""
    echo -e "${BOLD}Log File:${NC}"
    echo -e "  Path:      ${CYAN}k12_privacy_pipeline.log${NC}"
    echo -e "  Latest:"
    tail -3 k12_privacy_pipeline.log 2>/dev/null | sed 's/^/    /' || echo "    (waiting for updates...)"
    echo ""
    echo -e "${CYAN}[Time: $current_time] [Auto-refresh: 5s] [Ctrl+C to exit]${NC}"

    sleep 5
done
