#!/bin/bash
#
# GenomeVault Download Tracker - Auto-Updating Graphical Version
# Monitors ongoing whole genome FASTQ downloads with real-time updates
#
# Usage: ./scripts/track_downloads.sh
# Press Ctrl+C to exit

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
FASTQ_DIR="data/downloaded/fastq"
SRA_CACHE_DIR="${HOME}/ncbi/public/sra"
SRA_DOWNLOAD_DIR="."  # prefetch downloads to current directory
TEMP_DIR="data/downloaded/tmp/fasterq_temp"
REFRESH_INTERVAL=3  # seconds

# Actual expected sizes based on ENA API query (queried once, hardcoded for speed)
# get_expected_sra_size function returns actual size per accession
get_expected_sra_size() {
    local accession=$1
    case $accession in
        ERR3239334) echo "10395586560" ;;  # 9.69 GB in bytes
        ERR3239276) echo "11540983808" ;;  # 10.75 GB in bytes
        ERR3239454) echo "10116562944" ;;  # 9.42 GB in bytes
        ERR3239475) echo "10169548800" ;;  # 9.47 GB in bytes
        *) echo "10737418240" ;; # Default 10 GB
    esac
}

get_expected_fastq_size() {
    local accession=$1
    case $accession in
        ERR3239334) echo "27373864140" ;;  # 25.50 GB in bytes
        ERR3239276) echo "30360166195" ;;  # 28.29 GB in bytes
        ERR3239454) echo "26622631321" ;;  # 24.80 GB in bytes
        ERR3239475) echo "26740072038" ;;  # 24.91 GB in bytes
        *) echo "26843545600" ;; # Default 25 GB
    esac
}

EXPECTED_TEMP_SIZE_GB=18  # Expected temp files size during extraction (conservative estimate)

# Create temp file for storing previous sizes
SPEED_CACHE="/tmp/genomevault_download_speeds.txt"
touch "$SPEED_CACHE"

# Function to format bytes to human readable
format_bytes() {
    local bytes=$1
    if [ $bytes -lt 1024 ]; then
        echo "${bytes}B"
    elif [ $bytes -lt 1048576 ]; then
        printf "%.1fKB" "$(echo "scale=1; $bytes / 1024" | bc)"
    elif [ $bytes -lt 1073741824 ]; then
        printf "%.1fMB" "$(echo "scale=1; $bytes / 1048576" | bc)"
    else
        printf "%.2fGB" "$(echo "scale=2; $bytes / 1073741824" | bc)"
    fi
}

# Function to format speed
format_speed() {
    local bytes_per_sec=$1
    if [ $bytes_per_sec -lt 1024 ]; then
        printf "%.0fB/s" "$bytes_per_sec"
    elif [ $bytes_per_sec -lt 1048576 ]; then
        printf "%.1fKB/s" "$(echo "scale=1; $bytes_per_sec / 1024" | bc)"
    else
        printf "%.2fMB/s" "$(echo "scale=2; $bytes_per_sec / 1048576" | bc)"
    fi
}

# Function to draw progress bar
draw_progress_bar() {
    local percent=$1
    local width=50
    local filled=$(( width * percent / 100 ))
    local empty=$(( width - filled ))

    # Color based on progress
    local bar_color=$CYAN
    if [ $percent -ge 100 ]; then
        bar_color=$GREEN
    elif [ $percent -ge 75 ]; then
        bar_color=$BLUE
    elif [ $percent -ge 25 ]; then
        bar_color=$YELLOW
    fi

    printf "${bar_color}["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] ${BOLD}%3d%%${NC}" "$percent"
}

# Function to calculate ETA
calculate_eta() {
    local current_size=$1
    local total_size=$2
    local speed=$3

    if [ $speed -eq 0 ]; then
        echo "∞"
        return
    fi

    local remaining=$(( total_size - current_size ))
    local eta_sec=$(( remaining / speed ))

    if [ $eta_sec -lt 60 ]; then
        echo "${eta_sec}s"
    elif [ $eta_sec -lt 3600 ]; then
        echo "$(( eta_sec / 60 ))m $(( eta_sec % 60 ))s"
    else
        echo "$(( eta_sec / 3600 ))h $(( (eta_sec % 3600) / 60 ))m"
    fi
}

# Function to get file size safely
get_size() {
    local file=$1
    if [ -f "$file" ]; then
        stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

# Function to check if process is running
is_downloading() {
    local accession=$1
    pgrep -f "fasterq-dump $accession" > /dev/null 2>&1 || pgrep -f "prefetch $accession" > /dev/null 2>&1
}

# Function to find .sra file in multiple locations
find_sra_file() {
    local accession=$1

    # Check multiple possible locations
    local locations=(
        "$SRA_CACHE_DIR/${accession}.sra"
        "${accession}/${accession}.sra"
        "${accession}/${accession}.sra.tmp"
        "./${accession}/${accession}.sra"
        "./${accession}/${accession}.sra.tmp"
    )

    for location in "${locations[@]}"; do
        if [ -f "$location" ]; then
            echo "$location"
            return 0
        fi
    done

    return 1
}

# Function to count reference dependencies downloaded
count_reference_deps() {
    local accession=$1
    local ref_dir="${accession}"

    if [ -d "$ref_dir" ]; then
        # Count non-.sra files (these are reference dependencies)
        find "$ref_dir" -type f ! -name "*.sra*" ! -name "*.lock" ! -name "*.prf" ! -name "*.tmp" 2>/dev/null | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

# Function to detect download phase and get progress
get_download_phase() {
    local accession=$1

    # Check for final FASTQ files
    local fastq_r1="$FASTQ_DIR/${accession}_1.fastq"
    local fastq_r2="$FASTQ_DIR/${accession}_2.fastq"
    local fastq_r1_gz="$FASTQ_DIR/${accession}_1.fastq.gz"
    local fastq_r2_gz="$FASTQ_DIR/${accession}_2.fastq.gz"

    # Check if compressing (gzip/pigz is running OR uncompressed files still exist)
    if pgrep -f "gzip.*$accession" > /dev/null 2>&1 || \
       pgrep -f "pigz.*$accession" > /dev/null 2>&1 || \
       [ -f "$fastq_r1" ] || [ -f "$fastq_r2" ]; then
        echo "compressing"
        return
    fi

    # Both .gz files exist and no compression running = complete
    if [ -f "$fastq_r1_gz" ] && [ -f "$fastq_r2_gz" ]; then
        echo "complete"
        return
    fi

    # Uncompressed files exist but no compression process = still writing
    if [ -f "$fastq_r1" ] || [ -f "$fastq_r2" ]; then
        echo "writing"
        return
    fi

    # Check if fasterq-dump is running (extraction phase)
    if pgrep -f "fasterq-dump $accession" > /dev/null 2>&1; then
        echo "extracting"
        return
    fi

    # Check for temp files (extraction phase)
    local temp_files=$(find "$TEMP_DIR" -name "${accession}*" 2>/dev/null | wc -l)
    if [ $temp_files -gt 0 ]; then
        echo "extracting"
        return
    fi

    # Check for .sra file in multiple locations
    local sra_file=$(find_sra_file "$accession")
    if [ $? -eq 0 ] && [ -f "$sra_file" ]; then
        local sra_size=$(get_size "$sra_file")

        # Check if prefetch is still running
        if pgrep -f "prefetch $accession" > /dev/null 2>&1; then
            # .sra file exists and is large - check if downloading dependencies
            if [ $sra_size -gt 1000000000 ]; then
                # Large .sra file exists, prefetch still running = downloading dependencies
                local ref_count=$(count_reference_deps "$accession")
                if [ $ref_count -gt 0 ]; then
                    echo "dependencies"
                    return
                fi
            fi
            # Still downloading main .sra file
            echo "prefetching"
            return
        else
            # Prefetch process ended and file exists - it's complete
            if [ $sra_size -gt 1000000 ]; then
                echo "prefetched"
                return
            fi
        fi
    fi

    # Check if prefetch is running but file not visible yet
    if pgrep -f "prefetch $accession" > /dev/null 2>&1; then
        echo "prefetching"
        return
    fi

    # Check if fasterq-dump is running but no files yet
    if pgrep -f "fasterq-dump $accession" > /dev/null 2>&1; then
        echo "extracting"
        return
    fi

    echo "pending"
}

# Function to get total size of temp files for an accession
get_temp_size() {
    local accession=$1
    local total=0

    if [ -d "$TEMP_DIR" ]; then
        while IFS= read -r file; do
            local size=$(get_size "$file")
            total=$((total + size))
        done < <(find "$TEMP_DIR" -name "${accession}*" 2>/dev/null)
    fi

    echo $total
}

# Function to display sample info
display_sample() {
    local index=$1
    local accession=$2
    local population=$3
    local current_size=$4
    local expected_bytes=$5

    # Detect current phase
    local phase=$(get_download_phase "$accession")

    # Display header
    echo -e "${BOLD}${index}. ${population}${NC} (${accession})"

    case $phase in
        complete)
            # Already downloaded
            local r1_size=$(get_size "${FASTQ_DIR}/${accession}_1.fastq.gz")
            local r2_size=$(get_size "${FASTQ_DIR}/${accession}_2.fastq.gz")
            local total_size=$((r1_size + r2_size))
            echo -n "   "
            draw_progress_bar 100
            echo ""
            printf "   ${GREEN}✓ Complete${NC}  ${CYAN}Size:${NC} %s (R1: %s, R2: %s)\n" \
                "$(format_bytes $total_size)" \
                "$(format_bytes $r1_size)" \
                "$(format_bytes $r2_size)"
            ;;

        writing)
            # Writing final FASTQ files
            local percent=0
            if [ $current_size -gt 0 ] && [ $expected_bytes -gt 0 ]; then
                percent=$(( current_size * 100 / expected_bytes ))
                if [ $percent -gt 100 ]; then percent=100; fi
            fi

            echo -n "   "
            draw_progress_bar $percent
            echo ""
            printf "   ${GREEN}Phase 3/4: Writing FASTQ files${NC}  ${CYAN}Size:${NC} %s\n" \
                "$(format_bytes $current_size)"
            ;;

        compressing)
            # Compressing FASTQ files with gzip/pigz
            local fastq_r1="$FASTQ_DIR/${accession}_1.fastq"
            local fastq_r2="$FASTQ_DIR/${accession}_2.fastq"
            local fastq_r1_gz="$FASTQ_DIR/${accession}_1.fastq.gz"
            local fastq_r2_gz="$FASTQ_DIR/${accession}_2.fastq.gz"

            local uncompressed_size=0
            local compressed_size=0

            if [ -f "$fastq_r1" ]; then
                uncompressed_size=$(( uncompressed_size + $(get_size "$fastq_r1") ))
            fi
            if [ -f "$fastq_r2" ]; then
                uncompressed_size=$(( uncompressed_size + $(get_size "$fastq_r2") ))
            fi

            if [ -f "$fastq_r1_gz" ]; then
                compressed_size=$(( compressed_size + $(get_size "$fastq_r1_gz") ))
            fi
            if [ -f "$fastq_r2_gz" ]; then
                compressed_size=$(( compressed_size + $(get_size "$fastq_r2_gz") ))
            fi

            local expected_compressed=$(( uncompressed_size / 10 ))  # ~10:1 compression
            local percent=0

            if [ $compressed_size -gt 0 ] && [ $expected_compressed -gt 0 ]; then
                percent=$(( compressed_size * 100 / expected_compressed ))
                if [ $percent -gt 100 ]; then percent=100; fi
            fi

            echo -n "   "
            draw_progress_bar $percent
            echo ""
            printf "   ${YELLOW}Phase 4/4: Compressing FASTQ files${NC}  ${CYAN}Progress:${NC} %s / ~%s\n" \
                "$(format_bytes $compressed_size)" \
                "$(format_bytes $expected_compressed)"
            ;;

        extracting)
            # Extracting from .sra to temp files
            local temp_size=$(get_temp_size "$accession")
            local expected_temp_bytes=$(( EXPECTED_TEMP_SIZE_GB * 1073741824 ))
            local percent=0

            if [ $temp_size -gt 0 ] && [ $expected_temp_bytes -gt 0 ]; then
                percent=$(( temp_size * 100 / expected_temp_bytes ))
                if [ $percent -gt 100 ]; then percent=100; fi
            fi

            echo -n "   "
            draw_progress_bar $percent
            echo ""
            printf "   ${YELLOW}Phase 2/4: Extracting to temp files${NC}  ${CYAN}Temp:${NC} %s / ~%s\n" \
                "$(format_bytes $temp_size)" \
                "$(format_bytes $expected_temp_bytes)"
            ;;

        dependencies)
            # Downloading reference dependencies (Phase 1b)
            local ref_count=$(count_reference_deps "$accession")
            local total_deps=781  # Known number of reference dependencies
            local percent=0

            if [ $ref_count -gt 0 ] && [ $total_deps -gt 0 ]; then
                percent=$(( ref_count * 100 / total_deps ))
                if [ $percent -gt 100 ]; then percent=100; fi
            fi

            echo -n "   "
            draw_progress_bar $percent
            echo ""
            printf "   ${YELLOW}Phase 1b/3: Downloading reference dependencies${NC}  ${CYAN}Progress:${NC} %s/%s\n" \
                "$ref_count" "$total_deps"
            ;;

        prefetched)
            # .sra file downloaded, waiting for extraction
            local sra_file=$(find_sra_file "$accession")
            local sra_size=0
            if [ $? -eq 0 ] && [ -f "$sra_file" ]; then
                sra_size=$(get_size "$sra_file")
            fi

            echo -n "   "
            draw_progress_bar 100
            echo ""
            printf "   ${GREEN}Phase 1/3: Downloaded .sra file + dependencies${NC}  ${CYAN}Size:${NC} %s\n" \
                "$(format_bytes $sra_size)"
            echo -e "   ${YELLOW}⏳ Waiting for extraction to start...${NC}"
            ;;

        prefetching)
            # Downloading .sra file from NCBI
            local sra_file=$(find_sra_file "$accession")
            local sra_size=0
            if [ $? -eq 0 ] && [ -f "$sra_file" ]; then
                sra_size=$(get_size "$sra_file")
            fi
            local expected_sra_bytes=$(get_expected_sra_size "$accession")
            local percent=0

            if [ $sra_size -gt 0 ] && [ $expected_sra_bytes -gt 0 ]; then
                percent=$(( sra_size * 100 / expected_sra_bytes ))
                if [ $percent -gt 100 ]; then percent=100; fi
            fi

            echo -n "   "
            draw_progress_bar $percent
            echo ""
            printf "   ${YELLOW}Phase 1/3: Downloading .sra file${NC}  ${CYAN}Size:${NC} %s / ~%s\n" \
                "$(format_bytes $sra_size)" \
                "$(format_bytes $expected_sra_bytes)"
            ;;

        pending)
            # Not started yet
            echo -n "   "
            draw_progress_bar 0
            echo ""

            if is_downloading "$accession"; then
                echo -e "   ${YELLOW}⏳ Starting download...${NC}"
            else
                echo -e "   ${CYAN}Queued (waiting for previous download)${NC}"
            fi
            ;;
    esac

    echo ""
}

# Main display function
display_tracker() {
    clear

    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                  ${BOLD}GenomeVault Whole Genome Download Tracker${NC}${CYAN}                   ║${NC}"
    echo -e "${CYAN}║                        ${MAGENTA}Auto-Updating Every ${REFRESH_INTERVAL}s${NC}${CYAN}                              ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    local expected_bytes=$(( EXPECTED_SIZE_GB * 1073741824 ))

    # Sample 1: ERR3239334 (East Asian)
    local ACC1="ERR3239334"
    local FILE1_R1="$FASTQ_DIR/${ACC1}_1.fastq"
    local FILE1_R2="$FASTQ_DIR/${ACC1}_2.fastq"
    local SIZE1_R1=$(get_size "$FILE1_R1")
    local SIZE1_R2=$(get_size "$FILE1_R2")
    local TOTAL1=$(( SIZE1_R1 + SIZE1_R2 ))

    display_sample 1 "$ACC1" "East Asian (CHS)" $TOTAL1 $expected_bytes

    # Sample 2: ERR3239276 (South Asian)
    local ACC2="ERR3239276"
    local FILE2_R1="$FASTQ_DIR/${ACC2}_1.fastq"
    local FILE2_R2="$FASTQ_DIR/${ACC2}_2.fastq"
    local SIZE2_R1=$(get_size "$FILE2_R1")
    local SIZE2_R2=$(get_size "$FILE2_R2")
    local TOTAL2=$(( SIZE2_R1 + SIZE2_R2 ))

    display_sample 2 "$ACC2" "South Asian (ITU)" $TOTAL2 $expected_bytes

    # Sample 3: ERR3239454 (European)
    local ACC3="ERR3239454"
    local FILE3_R1="$FASTQ_DIR/${ACC3}_1.fastq"
    local FILE3_R2="$FASTQ_DIR/${ACC3}_2.fastq"
    local SIZE3_R1=$(get_size "$FILE3_R1")
    local SIZE3_R2=$(get_size "$FILE3_R2")
    local TOTAL3=$(( SIZE3_R1 + SIZE3_R2 ))

    display_sample 3 "$ACC3" "European (CEU)" $TOTAL3 $expected_bytes

    # Sample 4: ERR3239475 (African)
    local ACC4="ERR3239475"
    local FILE4_R1="$FASTQ_DIR/${ACC4}_1.fastq"
    local FILE4_R2="$FASTQ_DIR/${ACC4}_2.fastq"
    local SIZE4_R1=$(get_size "$FILE4_R1")
    local SIZE4_R2=$(get_size "$FILE4_R2")
    local TOTAL4=$(( SIZE4_R1 + SIZE4_R2 ))

    display_sample 4 "$ACC4" "African (YRI)" $TOTAL4 $expected_bytes

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}                                    Summary${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Count completed downloads and phases
    local COMPLETED=0
    local PHASE1=0  # Prefetching
    local PHASE1B=0 # Dependencies
    local PHASE2=0  # Extracting
    local PHASE3=0  # Writing
    local PHASE4=0  # Compressing

    for ACC in "$ACC1" "$ACC2" "$ACC3" "$ACC4"; do
        local phase=$(get_download_phase "$ACC")
        case $phase in
            complete) COMPLETED=$((COMPLETED + 1)) ;;
            compressing) PHASE4=$((PHASE4 + 1)) ;;
            writing) PHASE3=$((PHASE3 + 1)) ;;
            extracting) PHASE2=$((PHASE2 + 1)) ;;
            prefetching|prefetched) PHASE1=$((PHASE1 + 1)) ;;
            dependencies) PHASE1B=$((PHASE1B + 1)) ;;
        esac
    done

    local TOTAL_SIZE=$(( TOTAL1 + TOTAL2 + TOTAL3 + TOTAL4 ))
    local EXPECTED1=$(get_expected_fastq_size "$ACC1")
    local EXPECTED2=$(get_expected_fastq_size "$ACC2")
    local EXPECTED3=$(get_expected_fastq_size "$ACC3")
    local EXPECTED4=$(get_expected_fastq_size "$ACC4")
    local TOTAL_EXPECTED=$(( EXPECTED1 + EXPECTED2 + EXPECTED3 + EXPECTED4 ))
    local OVERALL_PERCENT=0
    if [ $TOTAL_EXPECTED -gt 0 ]; then
        OVERALL_PERCENT=$(( TOTAL_SIZE * 100 / TOTAL_EXPECTED ))
    fi

    echo -e "  ${BOLD}Overall Progress:${NC} "
    echo -n "  "
    draw_progress_bar $OVERALL_PERCENT
    echo ""
    echo ""

    printf "  ${CYAN}Completed:${NC} ${GREEN}$COMPLETED${NC}/4 genomes\n"

    if [ $PHASE1 -gt 0 ]; then
        printf "  ${CYAN}Phase 1 (Prefetch):${NC} ${YELLOW}$PHASE1${NC} genome(s)\n"
    fi
    if [ $PHASE1B -gt 0 ]; then
        printf "  ${CYAN}Phase 1b (Refs):${NC} ${YELLOW}$PHASE1B${NC} genome(s)\n"
    fi
    if [ $PHASE2 -gt 0 ]; then
        printf "  ${CYAN}Phase 2 (Extract):${NC} ${YELLOW}$PHASE2${NC} genome(s)\n"
    fi
    if [ $PHASE3 -gt 0 ]; then
        printf "  ${CYAN}Phase 3 (Write):${NC} ${YELLOW}$PHASE3${NC} genome(s)\n"
    fi
    if [ $PHASE4 -gt 0 ]; then
        printf "  ${CYAN}Phase 4 (Compress):${NC} ${YELLOW}$PHASE4${NC} genome(s)\n"
    fi

    printf "  ${CYAN}Final FASTQ size:${NC} $(format_bytes $TOTAL_SIZE) / $(format_bytes $TOTAL_EXPECTED)\n"
    echo ""

    if [ $COMPLETED -eq 4 ]; then
        echo -e "${GREEN}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${NC}"
        echo -e "${GREEN}┃                      🎉 All Downloads Complete! 🎉                       ┃${NC}"
        echo -e "${GREEN}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${NC}"
        echo ""
        echo -e "${YELLOW}Next steps:${NC}"
        echo -e "  1. Compress files (saves 70-80%): ${CYAN}./scripts/compress_fastq.sh${NC}"
        echo -e "  2. View data manifest: ${CYAN}cat data/downloaded/DATA_MANIFEST.md${NC}"
        echo -e "  3. Run GenomeVault pipeline: ${CYAN}python benchmarks/run_alignment_optimized_pipeline.py${NC}"
        echo ""
        echo -e "  ${MAGENTA}Press Ctrl+C to exit${NC}"
    else
        echo -e "${YELLOW}Downloads in progress...${NC}"
        echo ""

        # Show status explanation
        echo -e "  ${CYAN}ℹ${NC}  ${YELLOW}Download Process (5 phases):${NC}"
        echo -e "     ${BOLD}Phase 1:  Prefetch${NC}     - Download .sra file (~10 GB, 30-60 min)"
        echo -e "     ${BOLD}Phase 1b: References${NC}   - Download 781 reference chromosomes (10-15 min)"
        echo -e "     ${BOLD}Phase 2:  Extract${NC}      - Convert to temp files (~150 GB temp, 60-90 min)"
        echo -e "     ${BOLD}Phase 3:  Write${NC}        - Write final FASTQ files (~270 GB, 10-20 min)"
        echo -e "     ${BOLD}Phase 4:  Compress${NC}     - Compress to .gz (~25 GB final, 10-15 min)"
        echo ""
        echo -e "  ${CYAN}ℹ${NC}  ${MAGENTA}Sequential Downloads:${NC}"
        echo -e "     • Only 1 genome downloads at a time (prevents RAM exhaustion)"
        echo -e "     • Each genome takes ~2-4 hours total"
        echo -e "     • Progress updates every ${REFRESH_INTERVAL} seconds"
        echo ""

        echo -e "  ${MAGENTA}Press Ctrl+C to exit this tracker${NC}"
    fi

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    printf "${CYAN}Last updated: %-30s${NC}  ${CYAN}Next refresh in: ${REFRESH_INTERVAL}s${NC}\n" "$(date '+%Y-%m-%d %H:%M:%S')"
}

# Trap Ctrl+C to exit gracefully and clean up
trap 'echo ""; echo "Download tracker stopped."; rm -f "$SPEED_CACHE" "$SPEED_CACHE.bak"; exit 0' INT

# Main loop - auto-update
echo ""
echo -e "${CYAN}Starting auto-updating tracker...${NC}"
echo -e "${MAGENTA}Press Ctrl+C to exit${NC}"
sleep 2

while true; do
    display_tracker
    sleep $REFRESH_INTERVAL
done
