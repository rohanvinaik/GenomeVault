#!/bin/bash
# GDiff Template Builder Monitor
# Tracks template generation from dbSNP + ClinVar

# ANSI colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

TARGET_SIZE_MB=1000  # Expected final size: ~1 GB uncompressed
DBSNP_SIZE_GB=28

clear
echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         GDiff Template Builder Monitor                         ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Track start time
start_time=$(date +%s)
last_size=0
last_time=$(date +%s)

while true; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    # Format elapsed time
    elapsed_hours=$((elapsed_time / 3600))
    elapsed_mins=$(((elapsed_time % 3600) / 60))
    elapsed_secs=$((elapsed_time % 60))
    elapsed_str=$(printf "%02d:%02d:%02d" $elapsed_hours $elapsed_mins $elapsed_secs)

    # Check if template builder is running
    if ps aux | grep "build_gdiff_template" | grep -v grep > /dev/null; then
        process_status="${GREEN}●${NC} RUNNING"
        pid=$(ps aux | grep "build_gdiff_template" | grep -v grep | awk '{print $2}')
        cpu=$(ps aux | grep "build_gdiff_template" | grep -v grep | awk '{print $3}')
        mem=$(ps aux | grep "build_gdiff_template" | grep -v grep | awk '{print $4}')

        # Check which file is being read
        open_file=$(lsof -p $pid 2>/dev/null | grep -E "\.vcf\.gz" | awk '{print $NF}' | tail -1)

        if echo "$open_file" | grep -q "dbsnp"; then
            current_stage="${GREEN}●${NC} Loading dbSNP"
            stage_detail="Reading 28 GB VCF file"
            stage_desc="Filtering COMMON=1 variants (AF > 0.01)"
        elif echo "$open_file" | grep -q "clinvar"; then
            current_stage="${GREEN}●${NC} Loading ClinVar"
            stage_detail="Reading clinical variants"
            stage_desc="Processing pathogenic/benign annotations"
        elif echo "$open_file" | grep -q "gnomad"; then
            current_stage="${GREEN}●${NC} Loading gnomAD"
            stage_detail="Reading population frequencies"
            stage_desc="Processing allele frequencies"
        else
            # Check if writing output
            if lsof -p $pid 2>/dev/null | grep -q "gdiff_template"; then
                current_stage="${GREEN}●${NC} Writing template"
                stage_detail="Saving compressed JSON"
                stage_desc="Building hash index for O(1) lookup"
            else
                current_stage="${BLUE}●${NC} Processing"
                stage_detail="Building template structure"
                stage_desc="Creating sparse coordinate map"
            fi
        fi
    else
        if [ -f "data/templates/gdiff_template_GRCh38.json.gz" ]; then
            process_status="${GREEN}✓${NC} COMPLETE"
            current_stage="${GREEN}✓${NC} Finished"
            stage_detail="Template ready for use"
            stage_desc="See usage example below"
        else
            process_status="${RED}✗${NC} NOT RUNNING"
            current_stage="${RED}✗${NC} Stopped"
            stage_detail="Process terminated or failed"
            stage_desc="Check template_build.log for errors"
        fi
        pid="N/A"
        cpu="0.0"
        mem="0.0"
    fi

    # Check template file
    if [ -f "data/templates/gdiff_template_GRCh38.json.gz" ]; then
        template_size_bytes=$(stat -f%z "data/templates/gdiff_template_GRCh38.json.gz" 2>/dev/null || stat -c%s "data/templates/gdiff_template_GRCh38.json.gz" 2>/dev/null)
        template_size_mb=$(echo "scale=1; $template_size_bytes / 1024 / 1024" | bc)

        # Calculate growth rate
        time_diff=$((current_time - last_time))
        if [ $time_diff -gt 0 ] && [ $last_size -gt 0 ]; then
            bytes_diff=$((template_size_bytes - last_size))
            growth_mb_per_sec=$(echo "scale=2; $bytes_diff / 1024 / 1024 / $time_diff" | bc)

            # Calculate ETA
            remaining_mb=$(echo "$TARGET_SIZE_MB - $template_size_mb" | bc)
            if (( $(echo "$growth_mb_per_sec > 0" | bc -l) )); then
                eta_seconds=$(echo "$remaining_mb / $growth_mb_per_sec" | bc)
                eta_mins=$(echo "scale=0; $eta_seconds / 60" | bc)
                eta_str="${eta_mins}m"
            else
                eta_str="calculating..."
            fi
        else
            growth_mb_per_sec="0.00"
            eta_str="calculating..."
        fi

        last_size=$template_size_bytes
        last_time=$current_time

        template_status="${GREEN}✓${NC} ${template_size_mb} MB"

        # Progress percentage (rough estimate)
        if (( $(echo "$template_size_mb > 0" | bc -l) )); then
            progress_pct=$(echo "scale=1; $template_size_mb * 100 / $TARGET_SIZE_MB" | bc)
            if (( $(echo "$progress_pct > 100" | bc -l) )); then
                progress_pct="100.0"
            fi
        else
            progress_pct="0.0"
        fi
    else
        template_status="${YELLOW}⏳${NC} Pending..."
        template_size_mb="0.0"
        progress_pct="0.0"
        growth_mb_per_sec="0.00"
        eta_str="N/A"
    fi

    # Display status
    clear
    echo -e "${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║         GDiff Template Builder Monitor                         ║${NC}"
    echo -e "${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Process Status:${NC}"
    echo -e "  Status:    $process_status"
    echo -e "  PID:       ${CYAN}$pid${NC}"
    echo -e "  CPU:       ${CYAN}${cpu}%${NC}"
    echo -e "  Memory:    ${CYAN}${mem}%${NC}"
    echo -e "  Runtime:   ${CYAN}${elapsed_str}${NC}"
    echo ""
    echo -e "${BOLD}Current Stage:${NC}"
    echo -e "  Stage:     $current_stage"
    echo -e "  Detail:    ${CYAN}$stage_detail${NC}"
    echo -e "  Progress:  ${CYAN}$stage_desc${NC}"
    echo ""
    echo -e "${BOLD}Template File:${NC}"
    echo -e "  Path:      ${CYAN}data/templates/gdiff_template_GRCh38.json.gz${NC}"
    echo -e "  Size:      $template_status"
    echo -e "  Progress:  ${YELLOW}${progress_pct}%${NC}"

    # Progress bar
    if (( $(echo "$progress_pct > 0" | bc -l) )); then
        bar_width=50
        filled=$(echo "scale=0; $progress_pct * $bar_width / 100" | bc)
        bar=$(printf "%-${bar_width}s" "$(printf '█%.0s' $(seq 1 $filled 2>/dev/null))")
        echo -e "  ${GREEN}[${bar// /░}]${NC}"
    fi

    if [ "$growth_mb_per_sec" != "0.00" ]; then
        echo -e "  Growth:    ${CYAN}${growth_mb_per_sec} MB/s${NC}"
        echo -e "  ETA:       ${YELLOW}${eta_str}${NC}"
    fi
    echo ""
    echo -e "${BOLD}Input Data:${NC}"
    echo -e "  dbSNP:     ${CYAN}28 GB (build 156/157)${NC}"
    echo -e "  ClinVar:   ${CYAN}173 MB (3.88M variants)${NC}"
    echo -e "  Filter:    ${CYAN}COMMON=1 (AF > 0.01)${NC}"
    echo ""
    echo -e "${BOLD}Expected Output:${NC}"
    echo -e "  Variants:  ${CYAN}~70M common variants${NC}"
    echo -e "  Size:      ${CYAN}~1 GB compressed${NC}"
    echo -e "  Format:    ${CYAN}Sparse hash map (O(1) lookup)${NC}"
    echo -e "  Purpose:   ${CYAN}Template-based differential encoding${NC}"
    echo ""
    echo -e "${BOLD}Latest Log:${NC}"
    tail -3 template_build.log 2>/dev/null | sed 's/^/  /' || echo "  (waiting for log updates...)"
    echo ""

    if [ "$process_status" == "${GREEN}✓${NC} COMPLETE" ]; then
        echo -e "${BOLD}✓ BUILD COMPLETE!${NC}"
        echo ""
        echo -e "${BOLD}Usage:${NC}"
        echo -e "  ${YELLOW}encoder = GDiffEncoder(${NC}"
        echo -e "      ${YELLOW}query_bam='experimental.bam',${NC}"
        echo -e "      ${YELLOW}pool_bams=['ref1.bam', 'ref2.bam', ...],${NC}"
        echo -e "      ${YELLOW}template_path='data/templates/gdiff_template_GRCh38.json.gz',${NC}"
        echo -e "      ${YELLOW}enable_quality_check=True${NC}"
        echo -e "  ${YELLOW})${NC}"
        echo ""
        echo -e "${GREEN}Press Ctrl+C to exit${NC}"
        sleep 10
        break
    fi

    echo -e "${CYAN}[Time: $(date +%H:%M:%S)] [Auto-refresh: 5s] [Ctrl+C to exit]${NC}"

    # Wait 5 seconds before next update
    sleep 5
done
