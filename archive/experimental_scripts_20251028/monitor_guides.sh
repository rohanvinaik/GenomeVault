#!/bin/bash
echo "=== Guide Sequence Extraction Monitor ==="
for i in {1..20}; do
  sleep 60
  echo ""
  echo "Check $i at $(date +%H:%M:%S):"
  count=$(ls benchmark_results/k3_whole_genome_benchmark/guide_sequences/*.fa.gz 2>/dev/null | wc -l | tr -d ' ')
  
  if [ "$count" -eq "3" ]; then
    sizes=$(ls -lh benchmark_results/k3_whole_genome_benchmark/guide_sequences/*.fa.gz 2>/dev/null | awk '{if ($5 ~ /M/) print $5}' | wc -l | tr -d ' ')
    if [ "$sizes" -eq "3" ]; then
      echo "  ✅ ALL 3 GUIDES EXTRACTED!"
      ls -lh benchmark_results/k3_whole_genome_benchmark/guide_sequences/
      break
    fi
  fi
  
  echo "  Guides extracted: $count / 3"
  ls -lh benchmark_results/k3_whole_genome_benchmark/guide_sequences/*.fa.gz 2>/dev/null | head -5
done
