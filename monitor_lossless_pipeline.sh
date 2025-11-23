#!/bin/bash
while true; do
  clear
  echo "================================================================================  "
  echo "LOSSLESS k=11 GDiff Encoding Pipeline Monitor"
  echo "================================================================================"
  echo ""
  
  # Check if process is running
  if ps aux | grep "run_k12_gdiff_pipeline.py" | grep -v grep > /dev/null; then
    echo "✓ Pipeline RUNNING"
    
    # Get latest log
    LOG=$(ls -t k11_FIXED_LOSSLESS_*.log 2>/dev/null | head -1)
    if [ -n "$LOG" ]; then
      echo ""
      echo "Latest log: $LOG"
      echo "---"
      tail -15 "$LOG"
    fi
  else
    echo "✗ Pipeline STOPPED"
    echo ""
    echo "Last 20 lines of most recent log:"
    LOG=$(ls -t k11_FIXED_LOSSLESS_*.log 2>/dev/null | head -1)
    if [ -n "$LOG" ]; then
      tail -20 "$LOG"
    fi
  fi
  
  echo ""
  echo "---"
  echo "Refreshing in 30 seconds... (Ctrl+C to stop)"
  sleep 30
done
