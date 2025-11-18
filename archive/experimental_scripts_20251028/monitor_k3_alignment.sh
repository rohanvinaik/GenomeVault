#!/bin/bash
echo "=== k=3 Experimental Alignment Monitor ==="
echo "Started: $(date +%H:%M:%S)"
echo "PID: 53349"
echo ""

for i in {1..60}; do
  sleep 30
  
  echo "=== Check $i at $(date +%H:%M:%S) ==="
  
  # Check if process is still running
  if ! ps -p 53349 > /dev/null 2>&1; then
    echo "✅ Process completed!"
    echo ""
    echo "Final log (last 50 lines):"
    tail -50 logs/k3_experimental_alignment_fixed.log
    
    # Check if output VCF was created
    if [ -f benchmark_results/k3_whole_genome_benchmark/experimental.vcf.gz ]; then
      echo ""
      echo "✅ OUTPUT VCF CREATED!"
      ls -lh benchmark_results/k3_whole_genome_benchmark/experimental.vcf.gz
    fi
    break
  fi
  
  # Show latest log output
  echo "Process still running. Latest output:"
  tail -5 logs/k3_experimental_alignment_fixed.log
  echo ""
done
