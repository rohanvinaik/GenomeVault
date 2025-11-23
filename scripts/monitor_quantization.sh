#!/bin/bash
# Monitor quantization file creation progress

INT8_FILE="data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses_3d_int8.h5"
INT4_FILE="data/experimental_strands/ERR3239334/hdv_encoding/encoded_genome_5lenses_3d_int4.h5"
EXPECTED_INT8_SIZE_GB=54
EXPECTED_INT4_SIZE_GB=30

echo "================================================================================"
echo "QUANTIZATION PROGRESS MONITOR"
echo "================================================================================"
echo ""
date
echo ""

# Check process
PID=$(ps aux | grep create_proper_quantized_files.py | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    CPU=$(ps aux | grep $PID | grep -v grep | awk '{print $3}')
    MEM=$(ps aux | grep $PID | grep -v grep | awk '{print $4}')
    echo "✓ Process running: PID $PID (CPU: ${CPU}%, MEM: ${MEM}%)"
else
    echo "⚠️  Process not found - may have completed or crashed"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo "INT8 FILE"
echo "--------------------------------------------------------------------------------"

if [ -f "$INT8_FILE" ]; then
    # Get current size
    CURRENT_SIZE=$(stat -f%z "$INT8_FILE")
    CURRENT_GB=$(echo "scale=2; $CURRENT_SIZE / 1024 / 1024 / 1024" | bc)
    EXPECTED_BYTES=$(echo "$EXPECTED_INT8_SIZE_GB * 1024 * 1024 * 1024" | bc)
    PROGRESS=$(echo "scale=1; ($CURRENT_SIZE / $EXPECTED_BYTES) * 100" | bc)

    echo "Status: ⏳ Creating..."
    echo "Size:   ${CURRENT_GB} GB / ~${EXPECTED_INT8_SIZE_GB} GB"
    echo "Progress: ${PROGRESS}%"

    # Calculate speed and ETA
    if [ -f /tmp/int8_prev_size ]; then
        PREV_SIZE=$(cat /tmp/int8_prev_size)
        PREV_TIME=$(cat /tmp/int8_prev_time)
        CURRENT_TIME=$(date +%s)

        TIME_DIFF=$((CURRENT_TIME - PREV_TIME))
        SIZE_DIFF=$((CURRENT_SIZE - PREV_SIZE))

        if [ $TIME_DIFF -gt 0 ] && [ $SIZE_DIFF -gt 0 ]; then
            SPEED_BPS=$(echo "scale=2; $SIZE_DIFF / $TIME_DIFF" | bc)
            SPEED_MBPS=$(echo "scale=2; $SPEED_BPS / 1024 / 1024" | bc)

            REMAINING_BYTES=$((EXPECTED_BYTES - CURRENT_SIZE))
            if [ $(echo "$SPEED_BPS > 0" | bc) -eq 1 ]; then
                ETA_SECONDS=$(echo "scale=0; $REMAINING_BYTES / $SPEED_BPS" | bc)
                ETA_MINUTES=$(echo "scale=1; $ETA_SECONDS / 60" | bc)

                echo "Speed:  ${SPEED_MBPS} MB/s"
                echo "ETA:    ~${ETA_MINUTES} minutes"
            fi
        fi
    fi

    # Save current state for next check
    echo "$CURRENT_SIZE" > /tmp/int8_prev_size
    date +%s > /tmp/int8_prev_time
else
    echo "Status: ⏸️  Not started yet"
fi

echo ""
echo "--------------------------------------------------------------------------------"
echo "INT4 FILE"
echo "--------------------------------------------------------------------------------"

if [ -f "$INT4_FILE" ]; then
    CURRENT_SIZE=$(stat -f%z "$INT4_FILE")
    CURRENT_GB=$(echo "scale=2; $CURRENT_SIZE / 1024 / 1024 / 1024" | bc)
    EXPECTED_BYTES=$(echo "$EXPECTED_INT4_SIZE_GB * 1024 * 1024 * 1024" | bc)
    PROGRESS=$(echo "scale=1; ($CURRENT_SIZE / $EXPECTED_BYTES) * 100" | bc)

    echo "Status: ⏳ Creating..."
    echo "Size:   ${CURRENT_GB} GB / ~${EXPECTED_INT4_SIZE_GB} GB"
    echo "Progress: ${PROGRESS}%"
else
    echo "Status: ⏸️  Not started yet (waiting for int8 to complete)"
fi

echo ""
echo "================================================================================"
echo "Run this script again to update progress"
echo "================================================================================"
