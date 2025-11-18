#!/bin/bash
# Monitor k=11 experimental BAM creation progress

K11_DIR="data/experimental_strands/ERR3239334/alignment/k11_bams"
TOTAL_REFS=11
EXPECTED_SIZE_GB=2.5

echo "=========================================="
echo "k=11 Experimental BAM Alignment Monitor"
echo "=========================================="
echo ""

# Check if process is running
if ps aux | grep -q "[c]reate_k11_experimental_bams.sh"; then
    echo "✓ Alignment process is RUNNING"
    PID=$(ps aux | grep "[c]reate_k11_experimental_bams.sh" | awk '{print $2}' | head -1)
    echo "  PID: $PID"
    RUNTIME=$(ps -p $PID -o etime= | xargs)
    echo "  Runtime: $RUNTIME"
else
    echo "✗ Alignment process is NOT RUNNING"
fi

echo ""
echo "==========================================
"
echo "BAM Files Progress:"
echo "=========================================="

completed=0
in_progress=0
pending=0

for i in {1..11}; do
    BAM_FILE="${K11_DIR}/experimental_vs_ref${i}.sorted.bam"
    BAI_FILE="${BAM_FILE}.bai"
    ALIGN_LOG="${K11_DIR}/experimental_vs_ref${i}.align.log"

    if [ -f "$BAI_FILE" ]; then
        # Complete (has index)
        SIZE=$(du -h "$BAM_FILE" | awk '{print $1}')
        echo "✓ ref${i}: COMPLETE ($SIZE)"
        ((completed++))
    elif [ -f "$BAM_FILE" ]; then
        # In progress (BAM exists but no index)
        SIZE=$(du -h "$BAM_FILE" | awk '{print $1}')
        echo "⏳ ref${i}: SORTING ($SIZE)"
        ((in_progress++))
    elif [ -f "$ALIGN_LOG" ]; then
        # Alignment in progress (log exists)
        if [ -s "$ALIGN_LOG" ]; then
            READS=$(tail -1 "$ALIGN_LOG" | grep -o 'mapped [0-9]*' | awk '{print $2}' || echo "0")
            if [ "$READS" = "0" ] || [ -z "$READS" ]; then
                READS=$(tail -10 "$ALIGN_LOG" | grep -o 'mapped [0-9]*' | tail -1 | awk '{print $2}' || echo "0")
            fi
            READS_M=$(echo "scale=1; $READS / 1000000" | bc 2>/dev/null || echo "0")
            PROGRESS=$(echo "scale=1; $READS / 379 * 100" | bc 2>/dev/null || echo "0")
            echo "⏳ ref${i}: ALIGNING (${READS_M}M reads, ${PROGRESS}%)"
            ((in_progress++))
        else
            echo "⏳ ref${i}: STARTING..."
            ((in_progress++))
        fi
    else
        # Not started
        echo "⏹ ref${i}: PENDING"
        ((pending++))
    fi
done

echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="
echo "  ✓ Complete: $completed / $TOTAL_REFS"
echo "  ⏳ In Progress: $in_progress"
echo "  ⏹ Pending: $pending"

if [ $completed -eq $TOTAL_REFS ]; then
    echo ""
    echo "🎉 ALL BAMs COMPLETE!"
    echo ""
    echo "Ready to run GDiff encoding:"
    echo "  python3 benchmarks/run_k12_gdiff_pipeline.py"
else
    # Estimate time remaining
    if [ $completed -gt 0 ]; then
        # Calculate average time per BAM from runtime
        if ps aux | grep -q "[c]reate_k11_experimental_bams.sh"; then
            PID=$(ps aux | grep "[c]reate_k11_experimental_bams.sh" | awk '{print $2}' | head -1)
            # macOS ps doesn't have etimes, use etime and parse it
            RUNTIME_STR=$(ps -p $PID -o etime= | xargs)

            # Convert etime format (HH:MM:SS or MM:SS or DD-HH:MM:SS) to seconds
            # For simplicity, estimate based on completed BAMs (5 hours each)
            AVG_TIME_PER_BAM=$((5 * 3600))  # 5 hours in seconds
            REMAINING_BAMS=$((TOTAL_REFS - completed))
            EST_REMAINING_SEC=$((AVG_TIME_PER_BAM * REMAINING_BAMS))
            EST_REMAINING_HOURS=$(echo "scale=1; $EST_REMAINING_SEC / 3600" | bc)

            echo ""
            echo "Estimated time remaining: ${EST_REMAINING_HOURS} hours"
        fi
    else
        echo ""
        echo "Estimated total time: ~55 hours (~5 hours per BAM)"
    fi
fi

echo ""
echo "Total disk usage:"
du -sh "$K11_DIR" 2>/dev/null || echo "  0 B"

echo ""
echo "To check again, run:"
echo "  bash benchmarks/monitor_k11_alignment.sh"
echo ""
