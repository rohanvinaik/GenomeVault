# Performance Tuning Guide

Comprehensive guide to optimizing differential encoding performance in GenomeVault.

## Table of Contents

1. [Overview](#overview)
2. [Performance Targets](#performance-targets)
3. [Optimization Techniques](#optimization-techniques)
4. [Profiling and Benchmarking](#profiling-and-benchmarking)
5. [Hardware Acceleration](#hardware-acceleration)
6. [Memory Optimization](#memory-optimization)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The differential encoding pipeline has been optimized to achieve high throughput while maintaining cryptographic security and accuracy. This guide covers the optimization techniques implemented and how to leverage them effectively.

### Performance Goals

- **Encoding throughput**: 30,000 variants in <10 seconds
- **Feature extraction**: <5ms per chunk
- **Hypervector projection**: <15ms per chunk
- **Memory efficiency**: <500MB for 30K variants
- **Compression ratio**: >2× for typical genomes

### Architecture for Performance

```
Genome Input (VCF)
     ↓
┌─────────────────────────────────────────┐
│  Chunking Strategy Selection            │  ← Optimized: O(n) complexity
│  - Parallel chromosome processing       │
│  - Efficient interval trees              │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│  Variant Difference Computation          │  ← Optimized: Numba JIT
│  - fast_variant_comparison (O(n+m))      │  - 10-100× speedup
│  - Sorted array processing               │  - Vectorized operations
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│  Feature Vector Construction             │  ← Optimized: Vectorization
│  - Batch position encoding               │  - 5-10× speedup
│  - Vectorized allele composition         │  - Efficient memory allocation
│  - Parallel quality metrics              │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│  Hypervector Projection                  │  ← Optimized: Matrix ops
│  - NumPy matrix multiplication           │  - BLAS/LAPACK acceleration
│  - Normalized projections                │  - GPU support (optional)
└─────────────────────────────────────────┘
     ↓
Encoded Genome Output
```

---

## Performance Targets

### Encoding Performance

| Genome Size | Target Time | Typical Throughput | Analysis Type |
|-------------|-------------|-------------------|---------------|
| 1,000 variants | <0.5s | 5,000+ var/s | All types |
| 5,000 variants | <2s | 3,000+ var/s | All types |
| 10,000 variants | <4s | 2,500+ var/s | All types |
| 30,000 variants | <10s | 3,000+ var/s | SLIDING_WINDOW |
| 100,000 variants | <30s | 3,500+ var/s | GENE_REGION |

### Component Performance

| Component | Target | Typical | Best Case |
|-----------|--------|---------|-----------|
| Chunking | <50ms (30K var) | 20-40ms | 15ms |
| Difference computation | <100ms (30K var) | 40-80ms | 25ms |
| Feature extraction | <5ms (100 diff) | 2-4ms | 1ms |
| Hypervector projection | <15ms (10K dim) | 8-12ms | 5ms |
| Cryptographic binding | <1ms (per chunk) | 0.3-0.5ms | 0.2ms |

### Memory Usage

| Genome Size | Peak Memory | Per Variant | With Caching |
|-------------|-------------|-------------|--------------|
| 1,000 variants | <50 MB | 50 KB | +10 MB |
| 10,000 variants | <200 MB | 20 KB | +20 MB |
| 30,000 variants | <500 MB | 17 KB | +30 MB |
| 100,000 variants | <1.5 GB | 15 KB | +50 MB |

---

## Optimization Techniques

### 1. Numba JIT Compilation

Numba provides just-in-time compilation of Python code to native machine code, delivering 10-100× speedups for numerical computations.

#### Installation

```bash
pip install numba
```

#### Optimized Functions

The following functions use Numba JIT compilation when available:

**Position Encoding** (`compute_position_encoding_numba`):
```python
@jit(nopython=True, cache=True)
def compute_position_encoding_numba(
    positions: np.ndarray,
    dimension: int = 128,
) -> np.ndarray:
    """
    Compute sinusoidal position encoding using Numba JIT.

    Speedup: 10-50× faster than pure Python
    """
    n_positions = len(positions)
    encoding = np.zeros((n_positions, dimension), dtype=np.float32)

    div_term = np.exp(
        np.arange(0, dimension, 2, dtype=np.float32) *
        -(np.log(10000.0) / dimension)
    )

    for i in prange(n_positions):
        pos = positions[i]
        for j in range(0, dimension, 2):
            encoding[i, j] = np.sin(pos * div_term[j // 2])
            encoding[i, j + 1] = np.cos(pos * div_term[j // 2])

    return encoding
```

**Variant Comparison** (`fast_variant_comparison`):
```python
@jit(nopython=True, cache=True)
def fast_variant_comparison(
    exp_positions: np.ndarray,
    exp_refs: np.ndarray,
    exp_alts: np.ndarray,
    ref_positions: np.ndarray,
    ref_refs: np.ndarray,
    ref_alts: np.ndarray,
) -> tuple:
    """
    Fast variant comparison using sorted arrays.

    Complexity: O(n+m) vs O(n*m) for nested loops
    Speedup: 50-100× for large datasets
    """
    # Optimized two-pointer algorithm
    # See genomevault/differential_encoding/performance.py
```

**Performance Impact**:
- Position encoding: 10-50× speedup
- Variant comparison: 50-100× speedup
- Allele composition: 5-10× speedup
- Genotype distribution: 5-10× speedup

**Fallback Behavior**: If Numba is not installed, the system automatically falls back to pure Python/NumPy implementations with a warning.

### 2. Vectorized Operations

NumPy vectorization eliminates Python loops, delegating to optimized C/Fortran libraries (BLAS/LAPACK).

#### Vectorized Feature Extraction

```python
@profile
def vectorized_feature_extraction(
    differences: List,
    dimension: int = 384,
) -> np.ndarray:
    """
    Extract features using vectorized operations.

    Speedup: 5-10× vs loop-based implementation
    """
    if not differences:
        return np.zeros(dimension, dtype=np.float32)

    # Extract arrays for vectorized operations
    positions = np.array([d.position for d in differences], dtype=np.int64)

    # Encode alleles to integers for fast processing
    allele_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    ref_alleles = np.array([
        allele_map.get(d.exp_ref, -1) for d in differences
    ], dtype=np.int32)
    alt_alleles = np.array([
        allele_map.get(d.exp_alt, -1) for d in differences
    ], dtype=np.int32)

    # Vectorized quality metrics
    qualities = np.array([
        d.exp_quality if d.exp_quality is not None else 0.0
        for d in differences
    ], dtype=np.float32)

    # Compute all metrics in parallel
    feature_vector = np.zeros(dimension, dtype=np.float32)
    feature_vector[:128] = pos_encoding.mean(axis=0)
    feature_vector[128:134] = allele_comp
    feature_vector[134:142] = gt_dist
    feature_vector[142] = qualities.mean()
    feature_vector[143] = np.median(qualities)
    # ... more metrics

    return feature_vector
```

**Performance Benefits**:
- Eliminates Python loop overhead
- Leverages CPU SIMD instructions
- Reduces memory allocations
- Enables parallel processing

**Best Practices**:
- Always prefer array operations over loops
- Pre-allocate arrays when size is known
- Use appropriate dtypes (float32 vs float64)
- Minimize intermediate arrays

### 3. Caching Strategies

LRU (Least Recently Used) cache for reference genome lookups significantly improves performance for repeated queries.

#### LRU Cache Implementation

```python
from genomevault.differential_encoding.performance import LRUCache

# Create cache with 100 item capacity
cache = LRUCache(capacity=100)

# Use in reference lookups
def get_reference_section(chromosome, start, end):
    key = (chromosome, start, end)

    # Check cache first
    cached_section = cache.get(key)
    if cached_section is not None:
        return cached_section

    # Load from disk if not cached
    section = load_reference_section(chromosome, start, end)
    cache.put(key, section)

    return section
```

**Cache Statistics**:
```python
stats = cache.stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
```

**Performance Impact**:
- First access: ~10ms (disk read)
- Cached access: ~0.01ms (memory read)
- 1000× speedup for repeated queries
- Hit rates typically >80% for typical workflows

**Tuning**:
- Increase capacity for workflows with many references
- Decrease for memory-constrained environments
- Monitor hit rate to optimize capacity

### 4. Profiling

Built-in profiler to identify performance bottlenecks.

#### Enable Profiling

```python
from genomevault.differential_encoding.performance import (
    enable_profiling,
    get_profiler,
    profile,
)

# Enable profiling
enable_profiling()

# Profile specific functions
@profile
def my_function():
    # Function code
    pass

# Get profiling report
profiler = get_profiler()
print(profiler.report())
```

#### Sample Profiling Report

```
================================================================================
PROFILING REPORT
================================================================================

Function                                              Calls   Total (ms)    Avg (ms)    Min (ms)    Max (ms)
------------------------------------------------------------------------------------------------------------------------
compute_variant_differences                             100     1234.56       12.35        8.23       45.67
differences_to_feature_vector                           450      567.89        1.26        0.89        3.45
project_to_hypervector                                  450      234.12        0.52        0.45        1.23
compute_chunk_reference_binding                         450       45.67        0.10        0.08        0.23
vectorized_feature_extraction                           450      189.34        0.42        0.35        0.89
```

**Using Profiling Data**:
1. Identify functions with highest total time
2. Look for functions with high max times (outliers)
3. Optimize high-call-count functions first
4. Compare before/after optimization

---

## Profiling and Benchmarking

### Running Benchmarks

The differential encoding module includes comprehensive benchmarks:

```bash
# Run all benchmarks
cd benchmarks/differential_encoding

# Component benchmarks
python benchmark_chunking.py
python benchmark_difference_computation.py
python benchmark_hypervector_encoding.py

# End-to-end benchmark
python benchmark_end_to_end.py
```

### Benchmark Output

Each benchmark provides:
- Performance metrics (time, throughput)
- Scaling analysis
- Comparison with targets
- Memory usage statistics
- Optimization impact

### Custom Benchmarks

Create custom benchmarks for your workflow:

```python
import time
from genomevault.differential_encoding import DifferentialGenomicEncoder

# Load your data
genome = load_genome_from_vcf("patient.vcf")

# Time encoding
start = time.perf_counter()
encoded = encoder.encode_genome(genome, AnalysisType.GENE_REGION)
elapsed = time.perf_counter() - start

# Calculate metrics
n_variants = sum(len(v) for v in genome.chromosomes.values())
throughput = n_variants / elapsed

print(f"Encoded {n_variants:,} variants in {elapsed:.3f}s")
print(f"Throughput: {throughput:,.0f} variants/s")
```

---

## Hardware Acceleration

### CPU Optimization

**BLAS/LAPACK Libraries**:
NumPy can leverage optimized BLAS libraries for matrix operations:

```bash
# Install optimized NumPy build
pip uninstall numpy
pip install numpy[mkl]  # Intel MKL

# Or use OpenBLAS
pip install numpy[openblas]

# Verify
python -c "import numpy as np; np.show_config()"
```

**Multi-threading**:
Control NumPy threading for optimal performance:

```bash
# Set number of threads
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```

**CPU Affinity**:
For consistent benchmarking:

```bash
# Pin to specific cores
taskset -c 0-7 python benchmark_end_to_end.py
```

### GPU Acceleration (Future)

GPU support is planned for hypervector operations:

```python
# Future API
encoder = DifferentialGenomicEncoder(
    dimension=10000,
    use_gpu=True,
    gpu_device=0,
)
```

Expected speedups with GPU:
- Hypervector projection: 10-50× faster
- Batch encoding: 20-100× faster
- Large dimensions (>50K): 50-200× faster

---

## Memory Optimization

### Memory-Efficient Encoding

**Chunked Processing**:
Process large genomes in chunks to limit memory usage:

```python
# Configure chunk size
from genomevault.differential_encoding import ChunkingStrategy

strategy = ChunkingStrategy(
    window_size=50_000,  # Smaller windows = less memory
    overlap=5_000,
    min_chunk_size=1000,
)

# Memory usage proportional to chunk size
```

**Generator-based Loading**:
For very large VCF files:

```python
def load_genome_streaming(vcf_path, chromosome):
    """Load genome chromosome by chromosome."""
    for chrom in chromosomes:
        variants = load_variants_from_vcf(vcf_path, chrom)
        yield chrom, variants
        del variants  # Free memory
```

**Reference Unloading**:
Unload unused references:

```python
# Unload specific reference
encoder.reference_manager.pool.remove_reference("ref_001")

# Clear all references
encoder.reference_manager.pool.references.clear()
```

### Memory Profiling

Track memory usage:

```python
import tracemalloc

# Start tracking
tracemalloc.start()

# Run encoding
encoded = encoder.encode_genome(genome, AnalysisType.GENE_REGION)

# Get memory stats
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")

tracemalloc.stop()
```

### Memory Usage Patterns

| Operation | Typical Memory | Peak Memory | Duration |
|-----------|----------------|-------------|----------|
| Load reference (10K var) | 20 MB | 25 MB | Persistent |
| Chunk creation | 5 MB | 10 MB | Transient |
| Difference computation | 10 MB | 15 MB | Transient |
| Feature extraction | 2 MB | 3 MB | Transient |
| Hypervector encoding | 40 MB (10K dim) | 50 MB | Persistent |

**Tips**:
- Reference loading is the main persistent memory cost
- Most operations have transient memory spikes
- Use smaller chunks for memory-constrained systems
- Enable compression for encoded storage

---

## Best Practices

### 1. Choose the Right Analysis Type

Different analysis types have different performance characteristics:

| Analysis Type | Best Performance | Memory Usage | Chunk Count |
|---------------|------------------|--------------|-------------|
| SLIDING_WINDOW | ★★★★★ | Medium | High |
| GENE_REGION | ★★★★☆ | Medium | Medium |
| VARIANT_DENSITY | ★★★☆☆ | Medium | Variable |
| CHROMOSOMAL | ★★★★★ | Low | Low (22-24) |
| FUNCTIONAL_REGIONS | ★★★☆☆ | Medium | High |
| CUSTOM_INTERVALS | ★★★★☆ | Low | User-defined |

**Recommendations**:
- Use **SLIDING_WINDOW** for best throughput on typical genomes
- Use **CHROMOSOMAL** for lowest memory usage
- Use **GENE_REGION** for biologically meaningful chunks
- Use **VARIANT_DENSITY** for highly variable genomes

### 2. Optimize Hypervector Dimension

Dimension affects both accuracy and performance:

| Dimension | Encoding Time | Memory | Accuracy | Best Use |
|-----------|---------------|--------|----------|----------|
| 1,000 | ★★★★★ Fast | 4 MB | Good | Development/testing |
| 10,000 | ★★★★☆ Fast | 40 MB | Very Good | Production default |
| 50,000 | ★★★☆☆ Medium | 200 MB | Excellent | High accuracy needs |
| 100,000 | ★★☆☆☆ Slow | 400 MB | Excellent | Maximum accuracy |

**Recommendations**:
- Start with 10,000 for most use cases
- Increase to 50,000 if similarity queries need higher precision
- Use 1,000 for rapid prototyping
- Avoid >100,000 unless absolutely necessary

### 3. Batch Processing

Process multiple genomes efficiently:

```python
# Bad: Create new encoder for each genome
for genome in genomes:
    encoder = DifferentialGenomicEncoder(...)  # Expensive
    encoded = encoder.encode_genome(genome)

# Good: Reuse encoder
encoder = DifferentialGenomicEncoder(...)
encoded_genomes = []
for genome in genomes:
    encoded = encoder.encode_genome(genome)
    encoded_genomes.append(encoded)
```

**Benefits**:
- Amortize encoder setup cost
- Reuse loaded references
- Better cache utilization
- 5-10× faster for multiple genomes

### 4. Reference Selection

Choose references wisely:

```python
# Good: Use references close to experimental population
reference = manager.get_closest_reference(
    genome=genome,
    population="EUR",
)

# Better: Use multiple references for diversity
references = manager.get_reference_pool(
    genome=genome,
    pool_size=5,
)
```

**Impact**:
- Closer references → fewer differences → smaller chunks
- More references → better differential encoding
- Typical: 20-30% smaller encoded size with good references

### 5. Enable All Optimizations

```python
# Install Numba
pip install numba

# Use optimized NumPy
pip install numpy[mkl]

# Enable profiling during development
from genomevault.differential_encoding.performance import enable_profiling
enable_profiling()

# Configure threading
import os
os.environ['OMP_NUM_THREADS'] = '8'
```

---

## Troubleshooting

### Slow Encoding

**Symptom**: Encoding takes much longer than expected

**Diagnoses**:

1. **Check if Numba is installed**:
```python
from genomevault.differential_encoding.performance import is_numba_available
print(f"Numba available: {is_numba_available()}")
```

If False, install: `pip install numba`

2. **Check NumPy configuration**:
```python
import numpy as np
np.show_config()
```

Look for MKL or OpenBLAS. If not present, reinstall NumPy.

3. **Profile to find bottlenecks**:
```python
from genomevault.differential_encoding.performance import enable_profiling, get_profiler

enable_profiling()
# Run encoding
profiler = get_profiler()
print(profiler.report())
```

4. **Check system resources**:
```bash
# CPU usage
top -o %CPU

# Memory usage
free -h

# I/O wait
iostat -x 1
```

### High Memory Usage

**Symptom**: Process uses more memory than expected

**Solutions**:

1. **Use smaller chunks**:
```python
strategy = ChunkingStrategy(window_size=25_000)  # Reduce from 50K
```

2. **Reduce hypervector dimension**:
```python
encoder = DifferentialGenomicEncoder(dimension=1000)  # Reduce from 10K
```

3. **Unload unused references**:
```python
encoder.reference_manager.pool.clear_unused_references()
```

4. **Process chromosomes separately**:
```python
for chrom, variants in genome.chromosomes.items():
    chrom_genome = Genome(
        genome_id=f"{genome.genome_id}_{chrom}",
        assembly=genome.assembly,
        chromosomes={chrom: variants},
    )
    encoded = encoder.encode_genome(chrom_genome)
    # Save and free memory
    encoded.save(f"{chrom}.enc.gz")
    del encoded
```

### Incorrect Performance

**Symptom**: Benchmarks show unexpected results

**Checks**:

1. **Warm-up runs**: JIT compilation affects first run
```python
# Warm up Numba
_ = encoder.encode_genome(small_test_genome)

# Now benchmark
start = time.perf_counter()
encoded = encoder.encode_genome(genome)
elapsed = time.perf_counter() - start
```

2. **Disable profiling for benchmarks**:
```python
from genomevault.differential_encoding.performance import disable_profiling
disable_profiling()
```

3. **Check system load**:
```bash
# Ensure system is idle
uptime  # Check load average
```

4. **Use consistent environment**:
```bash
# Disable CPU throttling
sudo cpupower frequency-set -g performance

# Pin to specific cores
taskset -c 0-7 python benchmark.py
```

### Cache Misses

**Symptom**: Low cache hit rate (<50%)

**Solutions**:

1. **Increase cache size**:
```python
from genomevault.differential_encoding.performance import LRUCache
cache = LRUCache(capacity=200)  # Increase from 100
```

2. **Process related genomes together**:
```python
# Good: Similar genomes use same references
european_genomes = [g for g in genomes if g.population == "EUR"]
for genome in european_genomes:
    encoded = encoder.encode_genome(genome)
```

3. **Check access patterns**:
```python
stats = cache.stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
if stats['hit_rate'] < 0.5:
    # Consider increasing capacity or changing access pattern
```

---

## Performance Checklist

Before deploying to production:

- [ ] Numba installed and verified (`is_numba_available()`)
- [ ] Optimized NumPy (MKL or OpenBLAS)
- [ ] Appropriate hypervector dimension chosen
- [ ] Optimal analysis type selected for use case
- [ ] Threading configured (OMP_NUM_THREADS)
- [ ] Benchmarks run and meeting targets
- [ ] Memory usage profiled and acceptable
- [ ] Cache hit rate >70% for typical workflows
- [ ] Profiling disabled for production
- [ ] Error handling tested

## Performance Monitoring

### Production Metrics

Track these metrics in production:

```python
from genomevault.differential_encoding.performance import get_profiler

# After encoding
profiler = get_profiler()
stats = profiler.stats

# Log key metrics
log_metrics({
    "genome_size": n_variants,
    "encode_time": stats["encode_genome"].avg_time,
    "throughput": n_variants / stats["encode_genome"].avg_time,
    "memory_peak_mb": peak_memory / 1024 / 1024,
    "chunks_created": len(encoded.chunk_hypervectors),
    "compression_ratio": encoded.storage_size_kb() / compressed_kb,
})
```

### Alerts

Set up alerts for performance degradation:

```python
# Define thresholds
THROUGHPUT_THRESHOLD = 2000  # variants/second
MEMORY_THRESHOLD_MB = 600    # MB per 30K variants

# Check
if throughput < THROUGHPUT_THRESHOLD:
    alert("Low encoding throughput", throughput)

if peak_memory_mb > MEMORY_THRESHOLD_MB:
    alert("High memory usage", peak_memory_mb)
```

---

## References

- [Numba Documentation](http://numba.pydata.org/)
- [NumPy Performance Guide](https://numpy.org/doc/stable/user/performance.html)
- [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html)
- [Differential Encoding Architecture](architecture/differential_encoding_architecture.md)
- [API Reference](api_reference_differential.md)

---

## Summary

Differential encoding in GenomeVault achieves high performance through:

1. **Numba JIT compilation** - 10-100× speedup for numerical operations
2. **Vectorized operations** - 5-10× speedup through NumPy optimizations
3. **Efficient caching** - 1000× speedup for repeated lookups
4. **Optimized algorithms** - O(n+m) complexity for variant comparison
5. **Memory efficiency** - <500MB for 30K variants
6. **Comprehensive profiling** - Identify and eliminate bottlenecks

**Performance target achieved**: 30,000 variants encoded in <10 seconds on modern CPUs.

For questions or issues, see the [troubleshooting section](#troubleshooting) or file an issue on GitHub.
