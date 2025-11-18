#!/bin/bash
# Auto-monitor k=3 benchmark every 30 minutes and log progress

MONITOR_LOG="benchmark_results/k3_benchmark_progress.log"
MONITOR_SCRIPT="scripts/monitor_k3_gdiff_benchmark.py"

echo "=== Auto-monitor started at $(date '+%Y-%m-%d %H:%M:%S') ===" >> $MONITOR_LOG
echo "Using graphical monitor: $MONITOR_SCRIPT" >> $MONITOR_LOG
echo "" >> $MONITOR_LOG

while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> $MONITOR_LOG

    # Use new graphical monitor (non-watch mode for logging)
    python3 $MONITOR_SCRIPT 2>&1 | sed 's/\x1b\[[0-9;]*m//g' >> $MONITOR_LOG
    echo "" >> $MONITOR_LOG

    # Check if still running
    if ! ps aux | grep "run_k3_whole_genome_benchmark.py" | grep -v grep > /dev/null; then
        echo "Benchmark completed or stopped!" >> $MONITOR_LOG
        break
    fi

    # Wait 30 minutes
    sleep 1800
done

echo "=== Auto-monitor ended at $(date '+%Y-%m-%d %H:%M:%S') ===" >> $MONITOR_LOG
