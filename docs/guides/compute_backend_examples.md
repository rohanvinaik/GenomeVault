# GenomeVault Compute Backend - Usage Examples

Comprehensive examples for using the hardware abstraction layer across different deployment scenarios.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Production API Deployment](#production-api-deployment)
- [Research Batch Processing](#research-batch-processing)
- [Backend Selection Strategies](#backend-selection-strategies)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## Basic Usage

### Auto-Detect Best Backend

```python
from genomevault.compute import get_accelerator
import numpy as np

# Automatically detect and use best available backend (Metal > CUDA > CPU)
accelerator = get_accelerator()
print(f"Using: {accelerator.name}")

# Encode single sample
variants = np.random.randn(100, 10).astype(np.float32)
hypervector = accelerator.encode_single(variants)
print(f"Encoded to {hypervector.shape[0]}-dimensional vector")
```

### Explicit Backend Selection

```python
from genomevault.compute import initialize_backend, get_accelerator, ComputeBackend

# Force CPU backend (production default)
initialize_backend(ComputeBackend.CPU)
accelerator = get_accelerator()

# Force Metal backend (Apple Silicon)
initialize_backend(ComputeBackend.METAL)
accelerator = get_accelerator()

# Force CUDA backend (NVIDIA GPU)
initialize_backend(ComputeBackend.CUDA)
accelerator = get_accelerator()
```

---

## Production API Deployment

### API Endpoint with CPU-Only

```python
from fastapi import FastAPI, HTTPException
from genomevault.compute import initialize_backend, get_accelerator, ComputeBackend
import numpy as np

# Initialize CPU backend for predictable latency
initialize_backend(ComputeBackend.CPU)
accelerator = get_accelerator()

app = FastAPI()

@app.post("/encode")
async def encode_genome(variants: list[list[float]]):
    """
    Encode genomic variants to hypervector

    Target: <10ms latency for real-time clinical queries
    """
    try:
        # Convert to numpy
        variants_np = np.array(variants, dtype=np.float32)

        # Encode on CPU (optimized for latency)
        hypervector = accelerator.encode_single(variants_np)

        return {
            "hypervector": hypervector.tolist(),
            "dimension": len(hypervector),
            "backend": accelerator.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check with backend info"""
    return {
        "status": "healthy",
        "backend": accelerator.name,
        "latency_target": "<10ms"
    }
```

### Environment-Based Configuration

```python
import os
from genomevault.compute import initialize_backend, get_accelerator, ComputeBackend

# Read from environment variable
backend_name = os.getenv("GENOMEVAULT_BACKEND", "cpu").lower()

backend_map = {
    "cpu": ComputeBackend.CPU,
    "metal": ComputeBackend.METAL,
    "cuda": ComputeBackend.CUDA,
    "auto": ComputeBackend.AUTO,
}

# Initialize based on environment
backend = backend_map.get(backend_name, ComputeBackend.CPU)
initialize_backend(backend)
accelerator = get_accelerator()

print(f"✓ Production API using: {accelerator.name}")
```

---

## Research Batch Processing

### Batch Encoding with Auto GPU

```python
from genomevault.compute import get_accelerator, get_backend
import numpy as np
import time

# Auto-detect best backend (will use GPU if available)
accelerator = get_accelerator()
backend = get_backend()

print(f"Using backend: {backend.value}")

# Generate batch of 1000 samples
batch = [
    np.random.randn(100, 10).astype(np.float32)
    for _ in range(1000)
]

# Batch encode (GPU provides 50× speedup)
start = time.perf_counter()
hypervectors = accelerator.encode_batch(batch)
elapsed = time.perf_counter() - start

print(f"Encoded {len(batch)} samples in {elapsed:.2f}s")
print(f"Throughput: {len(batch)/elapsed:.0f} samples/sec")
print(f"Hypervectors shape: {hypervectors.shape}")

# Expected performance:
# CPU: ~5s (200 samples/sec)
# Metal: ~0.1s (10,000 samples/sec, 50× speedup)
# CUDA: ~0.15s (6,667 samples/sec, 33× speedup)
```

### Bulk Database Import

```python
from genomevault.compute import initialize_backend, get_accelerator, ComputeBackend
import numpy as np
from pathlib import Path

# Force GPU for bulk operations
try:
    # Try Metal first (Apple Silicon)
    initialize_backend(ComputeBackend.METAL)
except (ImportError, RuntimeError):
    try:
        # Fallback to CUDA
        initialize_backend(ComputeBackend.CUDA)
    except (ImportError, RuntimeError):
        # Final fallback to CPU (will be slow)
        initialize_backend(ComputeBackend.CPU)
        print("⚠️ Warning: Using CPU for bulk import (will be slow)")

accelerator = get_accelerator()
print(f"Bulk import using: {accelerator.name}")

def process_vcf_file(vcf_path: Path):
    """Process VCF file and return hypervectors"""
    # Parse VCF (implementation details omitted)
    variants_batch = parse_vcf(vcf_path)  # Returns list of variants

    # Batch encode all at once
    hypervectors = accelerator.encode_batch(variants_batch)

    return hypervectors

# Process multiple files
vcf_files = Path("data/vcf/").glob("*.vcf")
for vcf_file in vcf_files:
    hypervectors = process_vcf_file(vcf_file)
    print(f"Processed {vcf_file.name}: {len(hypervectors)} samples")
```

---

## Backend Selection Strategies

### Strategy 1: Latency-Optimized (Production API)

```python
from genomevault.compute import initialize_backend, ComputeBackend

# ALWAYS use CPU for production APIs
# Rationale:
# - Predictable latency (<10ms)
# - No GPU warmup overhead
# - Simpler deployment (no GPU drivers)
initialize_backend(ComputeBackend.CPU)
```

### Strategy 2: Throughput-Optimized (Research)

```python
from genomevault.compute import initialize_backend, ComputeBackend

# Use AUTO for research workloads
# Automatically selects best available GPU
initialize_backend(ComputeBackend.AUTO)

# This will select:
# 1. Metal if on Apple Silicon (best for unified memory)
# 2. CUDA if NVIDIA GPU present (best for discrete GPU)
# 3. CPU if no GPU (fallback)
```

### Strategy 3: Hybrid Approach

```python
from genomevault.compute import get_accelerator, get_backend, ComputeBackend
from genomevault.compute.cpu_backend import CPUBackend

# Use AUTO globally
accelerator = get_accelerator()

def encode_with_latency_priority(variants):
    """
    Single-sample encoding with explicit CPU path
    Bypasses GPU detection for minimal latency
    """
    cpu = CPUBackend()  # Direct CPU instance
    return cpu.encode_single(variants)

def encode_with_throughput_priority(variants_batch):
    """
    Batch encoding using auto-detected backend
    Uses GPU if available
    """
    return accelerator.encode_batch(variants_batch)

# Route based on use case
if is_real_time_query:
    result = encode_with_latency_priority(variants)
else:
    results = encode_with_throughput_priority(variants_batch)
```

---

## Performance Optimization

### Similarity Search Optimization

```python
from genomevault.compute import get_accelerator
import numpy as np

accelerator = get_accelerator()

# Small database (<100K): Direct computation
query = np.random.rand(8192).astype(np.float32)
small_db = np.random.rand(10_000, 8192).astype(np.float32)

indices, scores = accelerator.similarity_search(query, small_db, top_k=10)
print(f"Small DB search: {indices}")

# Large database (>100K): Automatic FAISS indexing
large_db = np.random.rand(1_000_000, 8192).astype(np.float32)

indices, scores = accelerator.similarity_search(query, large_db, top_k=10)
print(f"Large DB search (FAISS-accelerated): {indices}")
```

### Batch Size Selection

```python
from genomevault.compute import get_accelerator, get_backend, ComputeBackend

accelerator = get_accelerator()
backend = get_backend()

def optimal_batch_size(n_samples: int) -> int:
    """
    Determine optimal batch size based on backend

    Rules:
    - CPU: Process all at once (no transfer overhead)
    - CUDA: Batch > 100 to amortize transfer overhead
    - Metal: Any size OK (unified memory, no transfers)
    """
    if backend == ComputeBackend.CPU:
        return n_samples  # Process all at once

    elif backend == ComputeBackend.CUDA:
        # CUDA: Transfer overhead dominates for small batches
        if n_samples < 100:
            # Warn user
            print(f"⚠️ Small batch ({n_samples}) on CUDA: Consider using CPU")
        return max(100, n_samples)  # Minimum 100 for CUDA

    elif backend == ComputeBackend.METAL:
        # Metal: Unified memory, no transfer overhead
        return n_samples  # Any size is fine

    return n_samples

# Use optimal batch size
samples = [...]  # Your samples
batch_size = optimal_batch_size(len(samples))
hypervectors = accelerator.encode_batch(samples[:batch_size])
```

---

## Troubleshooting

### Check Backend Detection

```python
from genomevault.compute import initialize_backend, get_accelerator, get_backend, ComputeBackend

# Test all backends
print("Testing backend detection...")

backends_to_test = [
    ComputeBackend.AUTO,
    ComputeBackend.CPU,
    ComputeBackend.METAL,
    ComputeBackend.CUDA,
]

for backend_type in backends_to_test:
    try:
        initialize_backend(backend_type)
        accelerator = get_accelerator()
        print(f"✓ {backend_type.value}: {accelerator.name}")
    except (ImportError, RuntimeError) as e:
        print(f"✗ {backend_type.value}: {e}")
```

### Performance Comparison

```python
from genomevault.compute import initialize_backend, get_accelerator, ComputeBackend
import numpy as np
import time

test_data = [np.random.randn(100, 10).astype(np.float32) for _ in range(100)]

backends = [
    (ComputeBackend.CPU, "CPU"),
    (ComputeBackend.METAL, "Metal"),
    (ComputeBackend.CUDA, "CUDA"),
]

print("Performance comparison (100 samples):")
print("-" * 50)

for backend_type, name in backends:
    try:
        initialize_backend(backend_type)
        accelerator = get_accelerator()

        # Warm up
        _ = accelerator.encode_batch(test_data[:10])

        # Measure
        start = time.perf_counter()
        results = accelerator.encode_batch(test_data)
        elapsed = time.perf_counter() - start

        print(f"{name:10s}: {elapsed*1000:7.2f}ms ({len(test_data)/elapsed:6.0f} samples/sec)")

    except (ImportError, RuntimeError):
        print(f"{name:10s}: Not available")
```

### Memory Usage Monitoring

```python
from genomevault.compute import get_accelerator, get_backend, ComputeBackend
import psutil
import numpy as np

def monitor_memory(func):
    """Decorator to monitor memory usage"""
    def wrapper(*args, **kwargs):
        process = psutil.Process()

        # Before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Execute
        result = func(*args, **kwargs)

        # After
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_delta = mem_after - mem_before

        print(f"Memory: {mem_before:.1f}MB → {mem_after:.1f}MB (Δ{mem_delta:+.1f}MB)")

        return result
    return wrapper

@monitor_memory
def test_batch_encoding(batch_size: int):
    accelerator = get_accelerator()
    batch = [np.random.randn(100, 10).astype(np.float32) for _ in range(batch_size)]
    return accelerator.encode_batch(batch)

# Test different batch sizes
for size in [10, 100, 1000]:
    print(f"\nBatch size: {size}")
    test_batch_encoding(size)
```

---

## Configuration-Based Backend Selection

```python
import yaml
from pathlib import Path
from genomevault.compute import initialize_backend, ComputeBackend

# Load configuration
config_path = Path("genomevault/config/compute.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Get backend preference
backend_name = config['compute']['default_backend']

backend_map = {
    'auto': ComputeBackend.AUTO,
    'cpu': ComputeBackend.CPU,
    'metal': ComputeBackend.METAL,
    'cuda': ComputeBackend.CUDA,
}

backend = backend_map[backend_name]
initialize_backend(backend)

print(f"Initialized backend from config: {backend.value}")

# Apply component-specific settings
hdc_config = config['compute']['hdc_encoding']
batch_threshold = hdc_config['batch_threshold']

print(f"HDC batch threshold: {batch_threshold}")
print(f"GPU will be used for batch_size > {batch_threshold}")
```

---

## Best Practices Summary

### ✅ DO

- Use CPU for production API endpoints (predictable <10ms latency)
- Use AUTO for research batch processing (automatic GPU acceleration)
- Test backend selection in CI/CD pipeline
- Monitor performance metrics per backend
- Use FAISS for large-scale similarity search (>100K database)

### ❌ DON'T

- Use GPU for single-sample encoding (transfer overhead > compute savings)
- Use CUDA for small batches (<100 samples)
- Assume GPU is always faster (measure!)
- Force GPU in production without fallback to CPU
- Use GPU for ZK proofs or PIR (CPU-optimized workloads)

### 🎯 Performance Targets

| Operation | CPU | Metal | CUDA |
|-----------|-----|-------|------|
| Single encode | <10ms ✓ | <1ms ⚡ | ~2ms ⚠️ |
| Batch 100 | <1s ✓ | <10ms ⚡ | ~50ms ✓ |
| Batch 1K | <5s ✓ | <100ms ⚡ | <150ms ⚡ |
| Search 1M | <5s ✓ | <400ms ⚡ | <300ms ⚡ |

Legend:
- ✓ Good performance
- ⚡ Excellent performance
- ⚠️ Consider CPU instead (transfer overhead)
