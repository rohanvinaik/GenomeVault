#!/bin/bash
# Monitor parallel quantization progress

TARGET_DIR="/Users/rohanvinaik/genomevault/data/experimental_strands/ERR3239334/hdv_encoding"

echo "================================================================================"
echo "PARALLEL QUANTIZATION MONITOR"
echo "================================================================================"
date
echo ""

# Check processes
echo "Active Python processes:"
ps aux | grep "create_quantized_files_parallel" | grep -v grep | awk '{print "  PID " $2 ": CPU=" $3 "%, MEM=" $4 "%"}'
echo ""

# Check files
echo "================================================================================"
echo "FILE STATUS"
echo "================================================================================"

for file in int8 int4 ternary; do
    FILEPATH="$TARGET_DIR/encoded_genome_5lenses_3d_${file}.h5"
    echo ""
    echo "--- ${file^^} ---"

    if [ -f "$FILEPATH" ]; then
        SIZE_BYTES=$(stat -f%z "$FILEPATH")
        SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc)
        echo "Status: ⏳ Creating..."
        echo "Size:   ${SIZE_GB} GB"

        # Show modification time (how recently updated)
        echo "Last modified: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$FILEPATH")"
    else
        echo "Status: ⏸️  Not started yet"
    fi
done

echo ""
echo "================================================================================"
echo "Run this script again to update progress"
echo "================================================================================"
