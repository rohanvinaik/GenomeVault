#!/bin/bash
#
# Auto-updating monitor for Phase 1-3 pipeline
# Runs in background and updates status every 10 seconds
#

OUTPUT_FILE="benchmark_results/phase123_pipeline_status.txt"
LOG_FILE="logs/phase123_optimized_deployment.log"

echo "Starting auto-monitor (updating every 10 seconds)..."
echo "Status file: $OUTPUT_FILE"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Clear and write new status
    {
        echo "========================================================================"
        echo "Phase 1-3 Optimized Pipeline - Live Status"
        echo "Last updated: $(date)"
        echo "========================================================================"
        echo ""

        # Check if pipeline is running
        PID=$(ps aux | grep "deploy_phase123_optimized_pipeline.py" | grep -v grep | awk '{print $2}')

        if [ -z "$PID" ]; then
            echo "⚠️  Pipeline not running (may have completed or crashed)"
        else
            echo "✅ Pipeline running (PID: $PID)"

            # Show CPU/memory usage
            ps -p $PID -o %cpu,%mem,etime | tail -1 | awk '{printf "   CPU: %s%%  Memory: %s%%  Runtime: %s\n", $1, $2, $3}'
        fi

        echo ""
        echo "------------------------------------------------------------------------"
        echo "Progress"
        echo "------------------------------------------------------------------------"

        # Count completed references
        LAYER2_DIR="benchmark_results/enhanced_privacy_k13_phase123_optimized/layer2_reference_pool"
        if [ -d "$LAYER2_DIR" ]; then
            COMPLETED=$(ls -1 "$LAYER2_DIR"/*.vcf.gz 2>/dev/null | wc -l | tr -d ' ')
            echo "References completed: $COMPLETED / 12"

            if [ $COMPLETED -gt 0 ]; then
                echo ""
                echo "Completed:"
                ls -lh "$LAYER2_DIR"/*.vcf.gz 2>/dev/null | tail -3 | awk '{print "  " $9 " (" $5 ")"}'
            fi

            # Current work
            CURRENT_BAM=$(ls -t "$LAYER2_DIR"/*.bam 2>/dev/null | head -1)
            if [ -n "$CURRENT_BAM" ]; then
                echo ""
                echo "Currently processing:"
                ls -lh "$CURRENT_BAM" | awk '{print "  " $9 " (" $5 ")"}'
            fi
        fi

        echo ""
        echo "------------------------------------------------------------------------"
        echo "Performance"
        echo "------------------------------------------------------------------------"

        # Extract timing from log
        if [ -f "$LOG_FILE" ]; then
            # Index build
            INDEX_TIME=$(grep "Index built in" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
            [ -n "$INDEX_TIME" ] && echo "Minimap2 index: ${INDEX_TIME}s (built once, reused)"

            # Alignment times
            ALIGN_TIMES=$(grep "Alignment complete in" "$LOG_FILE" | grep -oE "[0-9]+\.[0-9]+" || true)
            if [ -n "$ALIGN_TIMES" ]; then
                COUNT=$(echo "$ALIGN_TIMES" | wc -l | tr -d ' ')
                AVG=$(echo "$ALIGN_TIMES" | awk '{s+=$1; c++} END {if(c>0) printf "%.1f", s/c}')
                MIN=$(echo "$ALIGN_TIMES" | sort -n | head -1)
                MAX=$(echo "$ALIGN_TIMES" | sort -n | tail -1)
                [ -n "$AVG" ] && echo "Alignment+sort: ${AVG}s avg (${MIN}s min, ${MAX}s max, n=$COUNT)"
            fi

            # Variant calling times
            VARIANT_TIMES=$(grep "Variant calling complete in" "$LOG_FILE" | grep -oE "[0-9]+\.[0-9]+" || true)
            if [ -n "$VARIANT_TIMES" ]; then
                COUNT=$(echo "$VARIANT_TIMES" | wc -l | tr -d ' ')
                AVG=$(echo "$VARIANT_TIMES" | awk '{s+=$1; c++} END {if(c>0) printf "%.1f", s/c}')
                MIN=$(echo "$VARIANT_TIMES" | sort -n | head -1)
                MAX=$(echo "$VARIANT_TIMES" | sort -n | tail -1)
                [ -n "$AVG" ] && echo "Variant calling: ${AVG}s avg (${MIN}s min, ${MAX}s max, n=$COUNT)"
            fi
        fi

        echo ""
        echo "------------------------------------------------------------------------"
        echo "Latest Log (last 5 lines)"
        echo "------------------------------------------------------------------------"
        [ -f "$LOG_FILE" ] && tail -5 "$LOG_FILE"

        echo ""
        echo "========================================================================"
        echo "Auto-updating every 10 seconds. View this file: cat $OUTPUT_FILE"
        echo "========================================================================"

    } > "$OUTPUT_FILE"

    # Show on screen too
    clear
    cat "$OUTPUT_FILE"

    # Wait 10 seconds
    sleep 10
done
