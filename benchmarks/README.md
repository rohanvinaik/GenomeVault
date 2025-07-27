# GenomeVault Benchmarks

This directory contains performance benchmark results for all GenomeVault lanes (ZK, PIR, HDC).

## Directory Structure

```
benchmarks/
├── hdc/          # Hyperdimensional Computing benchmarks
├── pir/          # Private Information Retrieval benchmarks
├── zk/           # Zero-Knowledge proof benchmarks
└── README.md     # This file
```

## Running Benchmarks

### Manual Execution

Run benchmarks for a specific lane:

```bash
# HDC benchmarks
make bench-hdc

# PIR benchmarks
make bench-pir

# ZK benchmarks
make bench-zk

# All benchmarks
make bench
```

Or use the script directly:

```bash
python scripts/bench.py --lane hdc --output benchmarks
```

### Automated Execution

Benchmarks run automatically:
- Daily at 2 AM UTC via GitHub Actions
- On every push to main that modifies relevant code
- Can be manually triggered from GitHub Actions UI

## Benchmark Output

Each benchmark run produces a timestamped JSON file:

```
benchmarks/hdc/20250124_143052.json
```

### HDC Benchmark Metrics

- **Encoding throughput**: Operations per second for different dimensions
- **Memory usage**: KB per hypervector for each compression tier
- **Compression ratio**: Data reduction achieved
- **Batch scalability**: Speedup with batch processing
- **Binding operations**: Performance of different binding types

### PIR Benchmark Metrics

- **Query generation**: Time to create PIR queries
- **Server response**: Computation time per server
- **Reconstruction**: Time to reconstruct data from responses
- **Scaling characteristics**: Communication vs database size
- **Batch query performance**: Throughput for multiple queries

### ZK Benchmark Metrics

- **Proof generation time**: Time to create proofs
- **Proof size**: Bytes per proof
- **Verification time**: Time to verify proofs

## Performance Reports

Generate a comprehensive performance report:

```bash
make perf-report
```

Or directly:

```bash
python scripts/generate_perf_report.py --input benchmarks --output docs/perf
```

This creates:
- `docs/perf/performance_report.html` - Visual report with charts
- `docs/perf/performance_summary.json` - Summary metrics
- `docs/perf/*.png` - Performance charts

## CI/CD Integration

The `benchmarks.yml` workflow:
1. Runs benchmarks for each lane
2. Uploads results as artifacts (retained for 90 days)
3. Generates performance reports
4. Comments on PRs with performance summary

## Adding New Benchmarks

To add benchmarks for a new component:

1. Create benchmark class in `scripts/bench.py`:
   ```python
   class MyComponentBenchmark:
       def __init__(self, harness: BenchmarkHarness):
           self.harness = harness

       async def run_all(self):
           # Run benchmarks
           results = {"metric": value}
           self.harness.add_result("test_name", results)
   ```

2. Add to lane selection in `run_benchmarks()`:
   ```python
   elif lane == "mycomponent":
       benchmark = MyComponentBenchmark(harness)
   ```

3. Update Makefile:
   ```makefile
   bench-mycomponent:
       @echo "Running MyComponent benchmarks..."
       $(PYTHON) $(SCRIPTS_DIR)/bench.py --lane mycomponent --output $(BENCH_DIR)
   ```

## Performance Targets

### HDC Lane
- Encoding: > 1000 ops/sec for clinical tier (10K dimensions)
- Memory: < 20 KB per hypervector
- Compression: > 500x ratio

### PIR Lane
- Query generation: < 10 ms
- Server response: < 100 ms for 1M items
- Reconstruction: < 5 ms

### ZK Lane
- Proof generation: < 200 ms
- Proof size: < 500 bytes
- Verification: < 30 ms

## Troubleshooting

If benchmarks fail:

1. Check dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. Verify module imports:
   ```python
   python -c "from genomevault.hypervector_transform.hdc_encoder import create_encoder"
   ```

3. Run with verbose logging:
   ```bash
   python scripts/bench.py --lane hdc --output benchmarks --verbose
   ```

4. Check individual benchmark scripts:
   ```bash
   python scripts/bench_hdc.py --quick
   ```
