#!/bin/bash
# Wrapper script to run benchmark with correct PYTHONPATH

export PYTHONPATH="/Users/rohanvinaik/genomevault:$PYTHONPATH"
cd "/Users/rohanvinaik/genomevault"

echo "Running benchmark with PYTHONPATH set..."
python3 benchmarks/benchmark_packed_hypervector.py "$@"
