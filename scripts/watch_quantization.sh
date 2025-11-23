#!/bin/bash
# Auto-updating quantization monitor

while true; do
    clear
    ./scripts/monitor_quantization.sh

    # Check if process is still running
    if ! ps aux | grep create_proper_quantized_files.py | grep -v grep > /dev/null; then
        echo ""
        echo "================================================================================"
        echo "✓ QUANTIZATION COMPLETE (process finished)"
        echo "================================================================================"
        break
    fi

    echo ""
    echo "Auto-refreshing in 30 seconds... (Ctrl+C to stop monitoring)"
    sleep 30
done
