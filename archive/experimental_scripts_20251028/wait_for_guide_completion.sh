#!/bin/bash
echo "=== Waiting for Guide Extraction to Complete ===" echo "Started: $(date +%H:%M:%S)"
echo ""

# PIDs to monitor
PIDS="13066 16170 16226"

while true; do
  sleep 60
  
  # Check if any samtools processes are still running
  running=0
  for pid in $PIDS; do
    if ps -p $pid > /dev/null 2>&1; then
      running=$((running + 1))
    fi
  done
  
  if [ $running -eq 0 ]; then
    echo ""
    echo "✅ ALL EXTRACTION PROCESSES COMPLETE at $(date +%H:%M:%S)!"
    echo ""
    ls -lh benchmark_results/k3_whole_genome_benchmark/guide_sequences/*.fa.gz
    echo ""
    echo "Testing file integrity:"
    for f in benchmark_results/k3_whole_genome_benchmark/guide_sequences/*.fa.gz; do
      echo -n "  $(basename $f): "
      gunzip -t "$f" 2>&1 && echo "✓ OK" || echo "✗ CORRUPTED"
    done
    break
  else
    echo "$(date +%H:%M:%S): $running/3 extraction processes still running"
    ls -lh benchmark_results/k3_whole_genome_benchmark/guide_sequences/*.fa.gz | awk '{print "  " $9 ": " $5}'
  fi
done
