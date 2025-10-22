#!/bin/bash
#
# Monitor Chunk Regeneration Progress
#
# Quick status checker for the background regeneration process
#

echo "========================================================================"
echo "Chunk Regeneration Monitor"
echo "========================================================================"
echo ""

# Check if regeneration is running
if ps aux | grep -q "[r]egenerate_missing_chunks.sh"; then
    echo "✓ Regeneration script is running"
else
    echo "⚠️  Regeneration script is not running"
fi

# Check NEAT worker
NEAT_PID=$(ps aux | grep "[n]eat read-simulator.*chunk_0" | awk '{print $2}')
if [ -n "$NEAT_PID" ]; then
    echo "✓ NEAT worker active (PID: $NEAT_PID)"
    ps aux | grep "[n]eat read-simulator.*chunk_0" | awk '{print "  CPU: " $3 "% | MEM: " $4 "% | TIME: " $10}'
else
    echo "⚠️  No NEAT worker found (may be between chunks)"
fi

echo ""
echo "Progress:"

# Count completed chunks
COMPLETED=$(find /Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples/temp/chunks_1-21_regenerated -name "sample2_r*.fastq.gz" 2>/dev/null | wc -l | tr -d ' ')
echo "  Completed files: $COMPLETED/42 (21 chunks × 2 files)"

# Calculate chunks completed
CHUNKS_DONE=$((COMPLETED / 2))
echo "  Chunks completed: $CHUNKS_DONE/21"

# Progress bar
PERCENT=$((CHUNKS_DONE * 100 / 21))
BARS=$((PERCENT / 5))
printf "  ["
for i in $(seq 1 20); do
    if [ $i -le $BARS ]; then
        printf "="
    else
        printf " "
    fi
done
printf "] %d%%\n" $PERCENT

echo ""
echo "Recent log (last 10 lines):"
if [ -f /Users/rohanvinaik/genomevault/benchmark_results/chunk_regeneration.log ]; then
    tail -10 /Users/rohanvinaik/genomevault/benchmark_results/chunk_regeneration.log | grep -E "Processing|complete|error|ERROR" || echo "  (Processing chunk - no status updates yet)"
else
    echo "  (Log file not found)"
fi

echo ""
echo "========================================================================"
echo "To watch live: tail -f /Users/rohanvinaik/genomevault/benchmark_results/chunk_regeneration.log"
echo "========================================================================"
