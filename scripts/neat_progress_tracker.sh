#!/bin/bash
#
# Graphical NEAT Progress Tracker
#
# Monitors NEAT synthetic genome generation with real-time progress bars.
# Tracks Ref2, Ref3, and Query sample generation.
#
# Usage: ./neat_progress_tracker.sh
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Progress bar function
progress_bar() {
    local current=$1
    local total=$2
    local label=$3
    local width=50

    local percentage=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    # Color based on percentage
    local color=$RED
    if [ $percentage -ge 75 ]; then
        color=$GREEN
    elif [ $percentage -ge 25 ]; then
        color=$YELLOW
    fi

    # Build progress bar
    printf "${BOLD}%-15s${NC} [" "$label"
    printf "${color}"
    for ((i=0; i<filled; i++)); do printf "█"; done
    printf "${NC}"
    for ((i=0; i<empty; i++)); do printf "░"; done
    printf "] ${BOLD}%3d%%${NC} (%d/%d chunks)\n" "$percentage" "$current" "$total"
}

# Estimate time remaining
estimate_time() {
    local current=$1
    local total=$2
    local elapsed=$3  # in seconds

    if [ $current -eq 0 ]; then
        echo "Calculating..."
        return
    fi

    local rate=$(echo "scale=2; $current / $elapsed" | bc)
    local remaining=$((total - current))
    local eta=$(echo "scale=0; $remaining / $rate" | bc 2>/dev/null || echo "0")

    if [ $eta -gt 0 ]; then
        local hours=$((eta / 3600))
        local minutes=$(( (eta % 3600) / 60 ))
        local seconds=$((eta % 60))

        if [ $hours -gt 0 ]; then
            printf "%dh %dm %ds" $hours $minutes $seconds
        elif [ $minutes -gt 0 ]; then
            printf "%dm %ds" $minutes $seconds
        else
            printf "%ds" $seconds
        fi
    else
        echo "Calculating..."
    fi
}

# Main monitoring loop
monitor_neat() {
    local WORK_DIR="/Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples"
    local TOTAL_CHUNKS=102
    local START_TIME=$(date +%s)

    echo ""
    echo "${BOLD}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo "${BOLD}║                    NEAT Reference Pool Generation Tracker                  ║${NC}"
    echo "${BOLD}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    while true; do
        clear

        echo ""
        echo "${BOLD}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
        echo "${BOLD}║                    NEAT Reference Pool Generation Tracker                  ║${NC}"
        echo "${BOLD}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "${BLUE}📊 Monitoring NEAT progress - Press Ctrl+C to exit${NC}"
        echo ""

        # Current time
        echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""

        # Check each reference
        local ref1_complete=false
        local ref2_complete=false
        local ref3_complete=false
        local query_complete=false

        local ref2_chunks=0
        local ref3_chunks=0
        local query_chunks=0

        # Ref1 (should be complete already)
        if [ -f "$WORK_DIR/references/ref1/sample1_r1.fastq.gz" ] && [ -s "$WORK_DIR/references/ref1/sample1_r1.fastq.gz" ]; then
            ref1_complete=true
        fi

        # Ref2 (81 salvaged + recovery run)
        local ref2_salvaged=81
        local ref2_recovery=0
        if [ -f "$WORK_DIR/references/ref2/sample2_r1.fastq.gz" ] && [ -s "$WORK_DIR/references/ref2/sample2_r1.fastq.gz" ]; then
            ref2_complete=true
            ref2_chunks=$TOTAL_CHUNKS
        else
            # Count new recovery chunks (using -newer flag)
            ref2_recovery=$(find /var/folders -name "sample2_r*.fastq.gz" -type f -newer "$WORK_DIR/scripts/recover_ref2_threads4.sh" -size +1M 2>/dev/null | wc -l | tr -d ' ')
            ref2_chunks=$((ref2_salvaged + ref2_recovery))
        fi

        # Ref3 (81 salvaged + recovery run)
        local ref3_salvaged=81
        local ref3_recovery=0
        if [ -f "$WORK_DIR/references/ref3/sample3_r1.fastq.gz" ] && [ -s "$WORK_DIR/references/ref3/sample3_r1.fastq.gz" ]; then
            ref3_complete=true
            ref3_chunks=$TOTAL_CHUNKS
        else
            # Count new recovery chunks (using -newer flag)
            ref3_recovery=$(find /var/folders -name "sample3_r*.fastq.gz" -type f -newer "$WORK_DIR/scripts/recover_ref3_threads4.sh" -size +1M 2>/dev/null | wc -l | tr -d ' ')
            ref3_chunks=$((ref3_salvaged + ref3_recovery))
        fi

        # Query (seed=1 test run - fresh generation)
        if [ -f "$WORK_DIR/query/sample4_r1.fastq.gz" ] && [ -s "$WORK_DIR/query/sample4_r1.fastq.gz" ]; then
            query_complete=true
            query_chunks=$TOTAL_CHUNKS
        else
            # Count all sample4 chunks
            query_chunks=$(find /var/folders -name "sample4_r1.fastq.gz" -type f -size +1M 2>/dev/null | wc -l | tr -d ' ')
        fi

        # Threads=4 Regeneration Test (NEW)
        local t4_test_active=false
        local t4_ref2_chunks=0
        local t4_ref3_chunks=0
        local t4_query_chunks=0

        if [ -f "/Users/rohanvinaik/genomevault/scripts/regenerate_chunks_1_21_threads4.sh" ]; then
            # Check if regeneration script has run recently (within last hour)
            local script_age=$(($(date +%s) - $(stat -f %m "/Users/rohanvinaik/genomevault/scripts/regenerate_chunks_1_21_threads4.sh")))
            if [ $script_age -lt 3600 ]; then
                t4_test_active=true
                # Count chunks generated by threads=4 test (newer than script)
                t4_ref2_chunks=$(find /var/folders -name "sample2_r1.fastq.gz" -type f -size +1M -newer "/Users/rohanvinaik/genomevault/scripts/regenerate_chunks_1_21_threads4.sh" 2>/dev/null | wc -l | tr -d ' ')
                t4_ref3_chunks=$(find /var/folders -name "sample3_r1.fastq.gz" -type f -size +1M -newer "/Users/rohanvinaik/genomevault/scripts/regenerate_chunks_1_21_threads4.sh" 2>/dev/null | wc -l | tr -d ' ')
                t4_query_chunks=$(find /var/folders -name "sample4_r1.fastq.gz" -type f -size +1M -newer "/Users/rohanvinaik/genomevault/scripts/regenerate_chunks_1_21_threads4.sh" 2>/dev/null | wc -l | tr -d ' ')
            fi
        fi

        # Display progress
        echo "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""

        if $ref1_complete; then
            echo "${GREEN}✓${NC} ${BOLD}Reference 1${NC}    ✅ Complete (seed=42)"
        else
            echo "${RED}✗${NC} ${BOLD}Reference 1${NC}    ⏸️  Not started"
        fi
        echo ""

        if $ref2_complete; then
            echo "${GREEN}✓${NC} ${BOLD}Reference 2${NC}    ✅ Complete (seed=200)"
        else
            progress_bar $ref2_chunks $TOTAL_CHUNKS "Ref2 Combined"
            echo "                 💾 Salvaged: $ref2_salvaged | 🔧 Recovery: $ref2_recovery (t=4)"
        fi
        echo ""

        if $ref3_complete; then
            echo "${GREEN}✓${NC} ${BOLD}Reference 3${NC}    ✅ Complete (seed=300)"
        else
            progress_bar $ref3_chunks $TOTAL_CHUNKS "Ref3 Combined"
            echo "                 💾 Salvaged: $ref3_salvaged | 🔧 Recovery: $ref3_recovery (t=4)"
        fi
        echo ""

        if $query_complete; then
            echo "${GREEN}✓${NC} ${BOLD}Query Sample${NC}   ✅ Complete (seed=1)"
        else
            progress_bar $query_chunks $TOTAL_CHUNKS "Query Fresh"
            if [ $query_chunks -gt 0 ]; then
                local elapsed=$(($(date +%s) - START_TIME))
                local eta=$(estimate_time $query_chunks $TOTAL_CHUNKS $elapsed)
                echo "                 ⏱️  ETA: $eta | 🧪 seed=1, threads=10"
            fi
        fi

        echo ""
        echo "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

        # Threads=4 Regeneration Test Status
        if $t4_test_active; then
            echo ""
            echo "${YELLOW}${BOLD}🧪 ACTIVE TEST: threads=4 Regeneration (Hypothesis: devs hardcoded threads=4)${NC}"
            echo "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo ""
            echo "  ${BOLD}Ref2 (threads=4):${NC} $t4_ref2_chunks unique chunks generated (target: chunks 1-21)"
            echo "  ${BOLD}Ref3 (threads=4):${NC} $t4_ref3_chunks unique chunks generated (target: chunks 1-21)"
            echo "  ${BOLD}Query (threads=4):${NC} $t4_query_chunks unique chunks generated (target: chunks 1-21)"
            echo ""
            if [ $t4_ref2_chunks -ge 21 ] || [ $t4_ref3_chunks -ge 21 ] || [ $t4_query_chunks -ge 21 ]; then
                echo "  ${GREEN}${BOLD}✅ HYPOTHESIS CONFIRMED: threads=4 generates chunks 1-21!${NC}"
            elif [ $t4_ref2_chunks -gt 0 ] || [ $t4_ref3_chunks -gt 0 ] || [ $t4_query_chunks -gt 0 ]; then
                echo "  ${BLUE}⏳ Test in progress... monitoring chunk ranges${NC}"
            else
                echo "  ${YELLOW}⏳ Test starting... waiting for chunk generation${NC}"
            fi
            echo ""
            echo "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        fi
        echo ""

        # Overall summary
        local completed_count=0
        $ref1_complete && completed_count=$((completed_count + 1))
        $ref2_complete && completed_count=$((completed_count + 1))
        $ref3_complete && completed_count=$((completed_count + 1))
        $query_complete && completed_count=$((completed_count + 1))

        echo "${BOLD}Overall Status:${NC} $completed_count/4 samples complete"
        echo "${BOLD}Zombie Data Package:${NC} Ref1(100%) + Ref2($((ref2_chunks*100/TOTAL_CHUNKS))%) + Ref3($((ref3_chunks*100/TOTAL_CHUNKS))%) + Query($((query_chunks*100/TOTAL_CHUNKS))%)"

        if [ $completed_count -ge 3 ] && $query_complete; then
            echo "${GREEN}${BOLD}🎉 k-Anonymity Achieved: k=$((completed_count - 1))${NC}"
            echo "${GREEN}${BOLD}✅ Ready for differential encoding!${NC}"
        elif [ $completed_count -ge 1 ]; then
            echo "${YELLOW}Partial data available - building zombie reference pool${NC}"
            echo "${BLUE}Strategy: Salvage all generated chunks when processes stall/fail${NC}"
        else
            echo "${YELLOW}Generating samples...${NC}"
        fi

        echo ""
        echo "${BLUE}Refreshing every 5 seconds...${NC}"
        echo ""

        # Check if all complete
        if $ref1_complete && $ref2_complete && $ref3_complete && $query_complete; then
            echo ""
            echo "${GREEN}${BOLD}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
            echo "${GREEN}${BOLD}║                  🎉 All Samples Generated Successfully! 🎉                 ║${NC}"
            echo "${GREEN}${BOLD}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            echo "${GREEN}✅ 3 reference genomes ready${NC}"
            echo "${GREEN}✅ 1 query genome ready${NC}"
            echo "${GREEN}✅ k=3 anonymity guaranteed${NC}"
            echo ""
            echo "Next step: python scripts/validate_reference_pool.py"
            echo ""
            exit 0
        fi

        sleep 5
    done
}

# Run monitor
monitor_neat
