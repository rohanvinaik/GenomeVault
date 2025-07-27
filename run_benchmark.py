#!/usr/bin/env python3
import sys

sys.path.insert(0, "/Users/rohanvinaik/genomevault")

# Now run the benchmark
exec(open("benchmarks/benchmark_packed_hypervector.py").read())
