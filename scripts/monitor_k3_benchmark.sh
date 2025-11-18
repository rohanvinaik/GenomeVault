#!/bin/bash
# Monitor k=3 whole genome GDiff benchmark progress

LOGFILE="benchmark_results/k3_whole_genome_benchmark_DEBUG.log"
PID=$(ps aux | grep "run_k3_whole_genome_benchmark.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "❌ Benchmark not running!"
    exit 1
fi

echo "========================================="
echo "k=3 Whole Genome Benchmark Monitor"
echo "========================================="
echo "PID: $PID"
echo "Started: $(stat -f %Sm -t '%Y-%m-%d %H:%M:%S' $LOGFILE)"
RUNTIME_SEC=$(($(date +%s) - $(stat -f %B $LOGFILE)))
RUNTIME_MIN=$((RUNTIME_SEC / 60))
RUNTIME_HR=$((RUNTIME_MIN / 60))
echo "Runtime: ${RUNTIME_SEC} seconds (${RUNTIME_MIN} min / ${RUNTIME_HR} hr)"
echo ""

# Worker status
echo "Worker Status:"
ps -p $PID -o pid,ppid,%cpu,%mem,etime,command | grep -v COMMAND
ps aux | grep -E "multiprocessing.*spawn_main" | grep $PID | awk '{printf "  Worker %s: CPU=%s%% MEM=%s%% TIME=%s\n", NR, $3, $4, $10}'
echo ""

# System resources
echo "System Resources:"
echo "  RAM: $(ps -p $PID -o %mem | tail -1)% (main process)"
echo "  CPU: $(ps -p $PID -o %cpu | tail -1)% (main process)"
TOTAL_CPU=$(ps aux | grep $PID | grep -v grep | awk '{sum+=$3} END {print sum}')
echo "  Total CPU (all workers): ${TOTAL_CPU}%"
echo ""

# Log analysis
echo "Last 10 lines of log:"
tail -10 $LOGFILE | grep -v "^$"
echo ""

# Estimated completion
RUNTIME_MIN=$((($(date +%s) - $(stat -f %B $LOGFILE))/60))
if [ $RUNTIME_MIN -gt 60 ]; then
    echo "⚠️  Running for ${RUNTIME_MIN} minutes (~$((RUNTIME_MIN/60)) hours)"
    echo "Estimated completion: 12-48 hours for whole genome"
else
    echo "⏱️  Running for ${RUNTIME_MIN} minutes"
    echo "Estimated completion: 12-48 hours for whole genome"
fi

# Check for output file
if [ -f "benchmark_results/k3_whole_genome_benchmark/experimental.gdiff.gz" ]; then
    SIZE=$(ls -lh benchmark_results/k3_whole_genome_benchmark/experimental.gdiff.gz | awk '{print $5}')
    echo "📊 Output file size: $SIZE (expected: ~150MB)"
fi

echo "========================================="
