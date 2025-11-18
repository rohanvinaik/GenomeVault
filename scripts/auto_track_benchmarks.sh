#!/bin/bash
#
# Auto-updating benchmark tracker
# Runs in background and updates benchmarks every 30 seconds
#

LOG_DIR="logs"
BENCHMARK_LOG="$LOG_DIR/benchmark_tracking.log"

mkdir -p "$LOG_DIR"

echo "Starting auto-benchmark tracker..."
echo "Updating every 30 seconds"
echo "Logs: $BENCHMARK_LOG"
echo "Press Ctrl+C to stop"
echo ""

# Log start
echo "=== Benchmark Tracking Started: $(date) ===" >> "$BENCHMARK_LOG"

# Counter for updates
UPDATE_COUNT=0

while true; do
    UPDATE_COUNT=$((UPDATE_COUNT + 1))

    echo "" >> "$BENCHMARK_LOG"
    echo "--- Update #$UPDATE_COUNT at $(date) ---" >> "$BENCHMARK_LOG"

    # Run benchmark tracker
    python3 scripts/track_pipeline_benchmarks.py >> "$BENCHMARK_LOG" 2>&1

    # Show brief status on screen
    clear
    echo "========================================================================"
    echo "Auto-Benchmark Tracker - Update #$UPDATE_COUNT"
    echo "Last updated: $(date)"
    echo "========================================================================"
    echo ""

    # Show latest benchmark summary
    tail -40 "$BENCHMARK_LOG" | grep -A 30 "BENCHMARK SUMMARY" || echo "Waiting for data..."

    echo ""
    echo "========================================================================"
    echo "Updating every 30 seconds. Press Ctrl+C to stop."
    echo "Full log: $BENCHMARK_LOG"
    echo "========================================================================"

    # Wait 30 seconds
    sleep 30
done
