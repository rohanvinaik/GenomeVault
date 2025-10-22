#!/bin/bash
#
# GenomeVault Full Pipeline Progress Tracker
#
# Monitors the complete pipeline benchmark running in the background
#
# Usage: ./scripts/pipeline_progress_tracker.sh
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Pipeline details
PIPELINE_PID=11614
LOG_FILE="benchmark_results/full_pipeline_run_*.log"
OUTPUT_DIR="benchmark_results/full_pipeline_results/pipeline_run_20251021_180403"

# Main monitoring loop
monitor_pipeline() {
    local START_TIME=$(date +%s)

    while true; do
        clear

        echo ""
        echo "${BOLD}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
        echo "${BOLD}║              GenomeVault Full Pipeline Progress Tracker                   ║${NC}"
        echo "${BOLD}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "${BLUE}📊 Press Ctrl+C to exit${NC}"
        echo ""

        # Current time and runtime
        local NOW=$(date +%s)
        local RUNTIME=$((NOW - START_TIME))
        local HOURS=$((RUNTIME / 3600))
        local MINUTES=$(((RUNTIME % 3600) / 60))
        local SECONDS=$((RUNTIME % 60))

        echo "⏰ $(date '+%Y-%m-%d %H:%M:%S') | Runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
        echo ""

        # Check if process is running
        if ps -p $PIPELINE_PID > /dev/null 2>&1; then
            echo "${GREEN}✓${NC} ${BOLD}Pipeline Process:${NC} Running (PID: $PIPELINE_PID)"
        else
            echo "${RED}✗${NC} ${BOLD}Pipeline Process:${NC} Completed or stopped"
        fi
        echo ""

        echo "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""

        # Latest log entries (last 15 lines, filtered for important info)
        echo "${BOLD}📋 Recent Activity:${NC}"
        echo ""
        if [ -f $(ls -t $LOG_FILE 2>/dev/null | head -1) ]; then
            tail -15 $(ls -t $LOG_FILE 2>/dev/null | head -1) | grep -E "(INFO|WARNING|ERROR|===|Step|Stage|Completed|Processing)" | tail -10 | while read line; do
                if echo "$line" | grep -q "ERROR"; then
                    echo "${RED}  $line${NC}"
                elif echo "$line" | grep -q "WARNING"; then
                    echo "${YELLOW}  $line${NC}"
                elif echo "$line" | grep -q "==="; then
                    echo "${CYAN}  $line${NC}"
                elif echo "$line" | grep -q "Completed"; then
                    echo "${GREEN}  $line${NC}"
                else
                    echo "  $line"
                fi
            done
        else
            echo "  ${YELLOW}Log file not found${NC}"
        fi
        echo ""

        echo "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""

        # Pipeline stages
        echo "${BOLD}🔄 Pipeline Stages:${NC}"
        echo ""

        local LOG_CONTENT=""
        if [ -f $(ls -t $LOG_FILE 2>/dev/null | head -1) ]; then
            LOG_CONTENT=$(cat $(ls -t $LOG_FILE 2>/dev/null | head -1))
        fi

        # Check each stage
        if echo "$LOG_CONTENT" | grep -q "FASTQ Processing"; then
            if echo "$LOG_CONTENT" | grep -q "Completed FASTQ Processing"; then
                echo "  ${GREEN}✓${NC} Stage 1: FASTQ Processing (Alignment, Region Detection, Variant Calling)"
            else
                echo "  ${YELLOW}⏳${NC} Stage 1: FASTQ Processing (Alignment, Region Detection, Variant Calling) ${YELLOW}[IN PROGRESS]${NC}"
            fi
        else
            echo "  ${BLUE}○${NC} Stage 1: FASTQ Processing"
        fi

        if echo "$LOG_CONTENT" | grep -q "Differential Encoding"; then
            if echo "$LOG_CONTENT" | grep -q "Completed Differential Encoding"; then
                echo "  ${GREEN}✓${NC} Stage 2: Differential Encoding (k=3 Anonymity)"
            else
                echo "  ${YELLOW}⏳${NC} Stage 2: Differential Encoding (k=3 Anonymity) ${YELLOW}[IN PROGRESS]${NC}"
            fi
        else
            echo "  ${BLUE}○${NC} Stage 2: Differential Encoding (k=3 Anonymity)"
        fi

        if echo "$LOG_CONTENT" | grep -q "HDC Integration"; then
            if echo "$LOG_CONTENT" | grep -q "Completed HDC"; then
                echo "  ${GREEN}✓${NC} Stage 3: HDC Integration (10,000D Hypervectors)"
            else
                echo "  ${YELLOW}⏳${NC} Stage 3: HDC Integration (10,000D Hypervectors) ${YELLOW}[IN PROGRESS]${NC}"
            fi
        else
            echo "  ${BLUE}○${NC} Stage 3: HDC Integration (10,000D Hypervectors)"
        fi

        if echo "$LOG_CONTENT" | grep -q "ZK Proof"; then
            if echo "$LOG_CONTENT" | grep -q "Completed ZK"; then
                echo "  ${GREEN}✓${NC} Stage 4: ZK Proof Generation (Privacy Verification)"
            else
                echo "  ${YELLOW}⏳${NC} Stage 4: ZK Proof Generation (Privacy Verification) ${YELLOW}[IN PROGRESS]${NC}"
            fi
        else
            echo "  ${BLUE}○${NC} Stage 4: ZK Proof Generation (Privacy Verification)"
        fi

        if echo "$LOG_CONTENT" | grep -q "PIR Query"; then
            if echo "$LOG_CONTENT" | grep -q "Completed PIR"; then
                echo "  ${GREEN}✓${NC} Stage 5: PIR Query (Privacy-Preserving Retrieval)"
            else
                echo "  ${YELLOW}⏳${NC} Stage 5: PIR Query (Privacy-Preserving Retrieval) ${YELLOW}[IN PROGRESS]${NC}"
            fi
        else
            echo "  ${BLUE}○${NC} Stage 5: PIR Query (Privacy-Preserving Retrieval)"
        fi

        echo ""
        echo "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""

        # Output files
        echo "${BOLD}📁 Output Files:${NC}"
        echo ""
        if [ -d "$OUTPUT_DIR" ]; then
            local FILE_COUNT=$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
            local DIR_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
            echo "  Directory: $OUTPUT_DIR"
            echo "  Files: $FILE_COUNT | Size: $DIR_SIZE"
        else
            echo "  ${YELLOW}Output directory not yet created${NC}"
        fi
        echo ""

        # Log file info
        echo "${BOLD}📝 Log File:${NC}"
        echo ""
        if [ -f $(ls -t $LOG_FILE 2>/dev/null | head -1) ]; then
            local LOG_SIZE=$(ls -lh $(ls -t $LOG_FILE 2>/dev/null | head -1) | awk '{print $5}')
            local LOG_LINES=$(wc -l < $(ls -t $LOG_FILE 2>/dev/null | head -1) | tr -d ' ')
            echo "  File: $(ls -t $LOG_FILE 2>/dev/null | head -1)"
            echo "  Size: $LOG_SIZE | Lines: $LOG_LINES"
            echo ""
            echo "  ${CYAN}View full log: tail -f $LOG_FILE${NC}"
        else
            echo "  ${YELLOW}Log file not found${NC}"
        fi

        echo ""
        echo "${BLUE}Refreshing every 5 seconds...${NC}"
        echo ""

        # Check if pipeline completed
        if ! ps -p $PIPELINE_PID > /dev/null 2>&1; then
            echo ""
            echo "${GREEN}${BOLD}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
            echo "${GREEN}${BOLD}║                    Pipeline Execution Completed! 🎉                       ║${NC}"
            echo "${GREEN}${BOLD}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            echo "${GREEN}View results:${NC}"
            echo "  Results directory: $OUTPUT_DIR"
            echo "  Full log: cat $LOG_FILE"
            echo ""
            exit 0
        fi

        sleep 5
    done
}

# Run monitor
monitor_pipeline
