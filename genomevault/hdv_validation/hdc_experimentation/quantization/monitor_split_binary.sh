#!/bin/bash

# Monitor split binary quantization progress

OUTPUT_FILE="genomevault/hdv_validation/hdc_experimentation/output/encoded_genome_6banks_split_binary.h5"
PID=43085

echo "=========================================="
echo "Split Binary Quantization Monitor"
echo "=========================================="
echo ""
echo "Output file: $OUTPUT_FILE"
echo "Process PID: $PID"
echo ""

# Check if process is running
if ! ps -p $PID > /dev/null 2>&1; then
    echo "✅ Process completed!"
    echo ""
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Final file size:"
        ls -lh "$OUTPUT_FILE"
        echo ""
        echo "Running validation..."
        python3 genomevault/hdv_validation/hdc_experimentation/quantization/validate_split_binary.py
    else
        echo "❌ Output file not found!"
    fi
    exit 0
fi

echo "Process is running. Monitoring file size growth..."
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

# Track previous size for rate calculation
PREV_SIZE=0
START_TIME=$(date +%s)

while ps -p $PID > /dev/null 2>&1; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ -f "$OUTPUT_FILE" ]; then
        # Get file size in bytes
        SIZE_BYTES=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null)
        SIZE_MB=$((SIZE_BYTES / 1048576))

        # Calculate growth rate
        if [ $PREV_SIZE -gt 0 ]; then
            GROWTH=$((SIZE_BYTES - PREV_SIZE))
            GROWTH_MB=$((GROWTH / 1048576))
            RATE_MB_PER_SEC=$((GROWTH / 30))
            RATE_MB_PER_MIN=$((RATE_MB_PER_SEC * 60 / 1048576))

            # Estimate completion (assuming ~450-500MB final size)
            TARGET_SIZE=$((475 * 1048576))
            REMAINING=$((TARGET_SIZE - SIZE_BYTES))

            if [ $RATE_MB_PER_SEC -gt 0 ]; then
                ETA_SEC=$((REMAINING / RATE_MB_PER_SEC))
                ETA_MIN=$((ETA_SEC / 60))

                echo "$(date '+%H:%M:%S') | Size: ${SIZE_MB}M | Growth: +${GROWTH_MB}M/30s (~${RATE_MB_PER_MIN}M/min) | ETA: ~${ETA_MIN} min"
            else
                echo "$(date '+%H:%M:%S') | Size: ${SIZE_MB}M | Growth: +${GROWTH_MB}M/30s"
            fi
        else
            echo "$(date '+%H:%M:%S') | Size: ${SIZE_MB}M | Monitoring..."
        fi

        PREV_SIZE=$SIZE_BYTES
    else
        echo "$(date '+%H:%M:%S') | File not yet created..."
    fi

    sleep 30
done

echo ""
echo "✅ Process completed!"
echo ""

if [ -f "$OUTPUT_FILE" ]; then
    echo "Final file size:"
    ls -lh "$OUTPUT_FILE"
    echo ""
    echo "Total time: $((ELAPSED / 60)) minutes"
    echo ""
    read -p "Run validation now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 genomevault/hdv_validation/hdc_experimentation/quantization/validate_split_binary.py
    fi
else
    echo "❌ Output file not found!"
fi
